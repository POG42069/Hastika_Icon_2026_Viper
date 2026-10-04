"""Khangtest Task B fixed-epoch training, adapted to Nhotest paired CSVs.

Reference: origin/Khangtest e27514b. Full-data class-weighted loss and soft
ensemble follow that branch; epoch checkpoints use Nhotest's resume format.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from llm_config import CHECKPOINT_ROOT, OUTPUT_ROOT, LLMTrainingConfig, TaskConfig
from src.data import labeled_input_filenames, load_epoch_validation_frame, load_task_frames, prompt_inputs, resolve_data_file
from src.distributed import RuntimeContext, initialize_runtime, cleanup_runtime, synchronize
from src.llm_data import ChatEncoder, CompletionDataset, CompletionCollator
from src.llm_training import (
    LineProgress, _autocast, completion_log_probs, compute_dtype, require_cuda,
    load_lora_model, save_adapter, restore_adapter, on_main, score_candidates,
    format_per_class_diagnostics, completion_token_diagnostics,
    log_completion_token_diagnostics, _training_state, _wrap_for_ddp,
    _input_hashes, _resume_signature, _save_epoch_resume, _latest_phase_checkpoint,
    _restore_epoch_resume, _clear_phase_resume, _write_json,
)
from src.resume import ResumeCatalog, prepare_resume_catalogs, read_catalog_manifest, write_catalog_manifest
from src.training import create_train_loader, log, set_global_seed, write_validation_confusion_matrix
from src.distributed import gather_python_objects
from src.submission import write_submission


class WeightedDataset(CompletionDataset):
    def __getitem__(self, index):
        feature = super().__getitem__(index)
        feature['class_id'] = self.labels[index]
        return feature


class WeightedCollator(CompletionCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        present = ['class_id' in feature for feature in features]
        if any(present) and not all(present):
            raise ValueError('A batch cannot mix labeled and candidate features.')
        if all(present):
            batch['class_ids'] = torch.tensor([feature['class_id'] for feature in features], dtype=torch.long)
        return batch


def compute_class_weights(
    label_ids: np.ndarray,
    n_classes: int,
    power: float,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return mild inverse-frequency weights with mean sample weight one."""

    labels = np.asarray(label_ids, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Class weighting requires a non-empty one-dimensional label array.")
    if np.any((labels < 0) | (labels >= n_classes)):
        raise ValueError("Class weighting received an invalid label index.")
    if not 0 < power <= 1:
        raise ValueError("class_weight_power must be in (0, 1].")
    if not 0 < minimum <= maximum:
        raise ValueError("Class-weight bounds must satisfy 0 < minimum <= maximum.")
    counts = np.bincount(labels, minlength=n_classes).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in a class-weighted training split.")

    balanced = labels.size / (n_classes * counts.astype(np.float64))
    weights = np.power(balanced, power)
    weights = np.clip(weights, minimum, maximum)
    sample_mean = float(np.dot(counts, weights) / labels.size)
    weights /= sample_mean
    return torch.tensor(weights, dtype=torch.float32), counts


def train_weighted_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    config,
    runtime,
    epoch,
    phase="LLM",
    class_weights: torch.Tensor | None = None,
):
    """Mean answer-token NLL per example; correctly scale even a partial group."""
    model.train()
    dtype = compute_dtype(config, runtime)
    device_class_weights = (
        class_weights.to(runtime.device, dtype=torch.float32)
        if class_weights is not None
        else None
    )
    total_loss, total_examples, successful_updates, skipped_updates = 0.0, 0, 0, 0
    iterator = iter(loader)
    progress = LineProgress(
        len(loader),
        f"{phase} | epoch {epoch}",
        "batches/GPU",
        runtime,
        config.progress_log_percent,
    )
    while group := list(islice(iterator, config.gradient_accumulation_steps)):
        group_size = sum(batch["input_ids"].shape[0] for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for index, cpu_batch in enumerate(group):
            batch = {key: value.to(runtime.device) for key, value in cpu_batch.items()}
            sync = (
                model.no_sync()
                if isinstance(model, DDP) and index < len(group) - 1
                else nullcontext()
            )
            with sync:
                with _autocast(runtime, dtype):
                    log_probs, counts = completion_log_probs(model, {key: value for key, value in batch.items() if key != "class_ids"})
                    example_losses = -log_probs / counts
                    if device_class_weights is None:
                        optimized_losses = example_losses
                    else:
                        if "class_ids" not in batch:
                            raise ValueError(
                                "Class-weighted training requires class IDs in every batch."
                            )
                        optimized_losses = example_losses * device_class_weights[
                            batch["class_ids"]
                        ]
                    loss = optimized_losses.sum() / group_size
                scaler.scale(loss).backward()
            total_loss += float(optimized_losses.detach().sum())
            total_examples += len(example_losses)
            progress.update(
                1, details=f"mean_loss={total_loss / total_examples:.4f}"
            )
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm, error_if_nonfinite=False
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # An FP16 overflow skips the optimizer update; don't advance LR then.
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            successful_updates += 1
        else:
            skipped_updates += 1
    if successful_updates == 0:
        raise RuntimeError(
            "No optimizer updates succeeded. Reduce initial_loss_scale or use BF16 if supported."
        )
    if skipped_updates:
        log(
            f"FP16 overflow: skipped {skipped_updates} updates; loss scale now {scaler.get_scale()}.",
            runtime,
        )
    gathered = gather_python_objects((total_loss, total_examples), runtime)
    return sum(item[0] for item in gathered) / sum(item[1] for item in gathered)


def _classification_metrics(
    task: TaskConfig,
    label_ids: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
) -> dict[str, object]:
    """Compute the fixed-label metrics used for aggregated OOF predictions."""

    labels = np.asarray(label_ids, dtype=np.int64)
    predictions = np.asarray(probabilities).argmax(axis=1)
    label_indices = list(range(len(task.labels)))
    per_class_scores = f1_score(
        labels,
        predictions,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=label_indices)
    result: dict[str, object] = {
        "loss": float(loss),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(task.labels, per_class_scores, strict=True)
        },
        "confusion_matrix": {
            "rows": "actual labels",
            "columns": "predicted labels",
            "label_order": list(task.labels),
            "values": matrix.astype(int).tolist(),
        },
    }
    if task.report_per_class_metrics:
        precision, recall, class_f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_indices,
            zero_division=0,
        )
        result["per_class"] = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(task.labels)
        }
    return result


def _class_weight_state(
    task: TaskConfig, label_ids: np.ndarray
) -> tuple[torch.Tensor | None, dict[str, object] | None]:
    """Compute task-configured class weights and a JSON-safe audit summary."""

    if task.class_weight_power is None:
        return None, None
    weights, counts = compute_class_weights(
        label_ids,
        len(task.labels),
        task.class_weight_power,
        task.class_weight_min,
        task.class_weight_max,
    )
    summary: dict[str, object] = {
        "formula": "clip((N / (K * n_c)) ** power, minimum, maximum), then normalize mean sample weight to 1",
        "power": task.class_weight_power,
        "pre_normalization_clip": [task.class_weight_min, task.class_weight_max],
        "counts": {
            label: int(count) for label, count in zip(task.labels, counts, strict=True)
        },
        "weights": {
            label: float(weight)
            for label, weight in zip(task.labels, weights.tolist(), strict=True)
        },
    }
    return weights, summary


def _write_prediction_diagnostics(
    frame: pd.DataFrame,
    task: TaskConfig,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    """Write labels, folds and probabilities for OOF or ensemble auditing."""

    label_ids = np.asarray(probabilities).argmax(axis=1)
    data: dict[str, object] = {
        "id": frame["id"].astype(str).tolist(),
    }
    if task.label_column in frame:
        data["true_label"] = frame[task.label_column].tolist()
    if "fold" in frame:
        data["fold"] = frame["fold"].astype(int).tolist()
    data["predicted_label"] = [task.labels[index] for index in label_ids]
    for label_index, label in enumerate(task.labels):
        safe_label = label.lower().replace("-", "_").replace(" ", "_")
        data[f"prob_{safe_label}"] = probabilities[:, label_index]
    pd.DataFrame(data).to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def format_confusion_matrix(metrics: dict) -> list[str]:
    """Render actual-label rows and predicted-label columns in Kaggle text logs."""

    confusion = metrics["confusion_matrix"]
    labels = confusion["label_order"]
    matrix = np.asarray(confusion["values"], dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix shape does not match its label order.")
    width = max(len("actual / predicted"), *(len(label) for label in labels)) + 2
    lines = [
        "Confusion matrix | rows=actual labels | columns=predicted labels",
        "actual / predicted".ljust(width)
        + "".join(label.rjust(width) for label in labels),
    ]
    lines.extend(
        label.ljust(width) + "".join(str(int(value)).rjust(width) for value in row)
        for label, row in zip(labels, matrix, strict=True)
    )
    return lines


def log_classification_metrics(metrics: dict, heading: str, runtime: RuntimeContext) -> None:
    """Print aggregate metrics, class metrics and both directions of confusion."""

    log(
        f"{heading}: val_loss={metrics['loss']:.5f}, "
        f"macro_f1={metrics['macro_f1']:.5f}, accuracy={metrics['accuracy']:.5f}",
        runtime,
    )
    for line in format_confusion_matrix(metrics) + format_per_class_diagnostics(metrics):
        log(line, runtime)


def public_validation_metrics(metrics: dict) -> dict:
    """Exclude per-example score arrays from JSON training history."""

    return {
        key: metrics[key]
        for key in (
            "loss", "macro_f1", "accuracy", "per_class_f1", "per_class",
            "confusion_matrix",
        )
        if key in metrics
    }


class TaskBValidationMonitor:
    """Report external gold each epoch without training on it or selecting epochs."""

    def __init__(self, task, config, runtime, training_frame, encoder, collator):
        self.task, self.config, self.runtime = task, config, runtime
        self.collator = collator
        self.frame = load_epoch_validation_frame(task)
        self.data = None
        self.summary = {'available': False, 'filename': task.epoch_validation_filename}
        if self.frame is None:
            log('Task B: external gold validation is absent; training and prediction continue without validation metrics.', runtime)
            return
        overlap = set(training_frame['id']) & set(self.frame['id'])
        if overlap:
            raise ValueError(f'Official validation IDs must not occur in training data: {sorted(overlap)[:10]}')
        self.data = CompletionDataset(prompt_inputs(self.frame, task), encoder, self.frame['label_id'].tolist())
        self.summary = {
            'available': True, 'filename': task.epoch_validation_filename,
            'sha256': hashlib.sha256(resolve_data_file(task.epoch_validation_filename).read_bytes()).hexdigest(),
            'input_sha256': hashlib.sha256(resolve_data_file(task.epoch_validation_input_filename or task.prediction_filename).read_bytes()).hexdigest(),
            'rows': len(self.frame), 'label_distribution': self.frame[task.label_column].value_counts().to_dict(),
            'truncated_comments': self.data.truncated_count,
            'purpose': 'reporting only; never used for training or epoch selection',
            'raw_column': task.original_comment_column, 'english_column': 'Comment',
        }
        log(f'Task B per-epoch validation | rows={len(self.frame)} | gold matched by ID | reporting only', runtime)

    def evaluate(self, model, seed, epoch, output_dir):
        if self.data is None:
            return None
        heading = f'Official validation | seed={seed} | epoch={epoch}'
        devices = [self.runtime.device.index] if self.runtime.device.type == 'cuda' else []
        was_training = model.training
        try:
            with torch.random.fork_rng(devices=devices):
                metrics = score_candidates(model, self.data, self.collator, self.config, self.runtime, heading)
        finally:
            model.train(was_training)
        log_classification_metrics(metrics, heading, self.runtime)

        def export():
            directory = output_dir / 'official_validation'
            directory.mkdir(parents=True, exist_ok=True)
            write_validation_confusion_matrix(metrics['confusion_matrix'], epoch, directory)
            _write_prediction_diagnostics(self.frame, self.task, metrics['probabilities'], directory / f'predictions_epoch_{epoch:02d}.csv')
            _write_json(directory / f'metrics_epoch_{epoch:02d}.json', {'seed': seed, 'epoch': epoch, **public_validation_metrics(metrics)})

        on_main(export, self.runtime)
        return metrics

    def evaluate_ensemble(self, results, output_dir):
        if self.data is None:
            return None
        probabilities = np.mean(np.stack([result['validation_probabilities'] for result in results]), axis=0)
        metrics = _classification_metrics(self.task, self.frame['label_id'].to_numpy(), probabilities,
                                          float(np.mean([result['validation_metrics']['loss'] for result in results])))
        log_classification_metrics(metrics, 'Official validation | FINAL SOFT ENSEMBLE', self.runtime)

        def export():
            _write_json(output_dir / 'official_validation_metrics.json', {'dataset': self.summary, **public_validation_metrics(metrics)})
            _write_prediction_diagnostics(self.frame, self.task, probabilities, output_dir / 'official_validation_predictions.csv')

        on_main(export, self.runtime)
        return public_validation_metrics(metrics)


def _finished_seed(catalogs, seed, signature):
    for catalog in catalogs:
        output, checkpoint = catalog.seed_output(seed), catalog.seed_checkpoint(seed)
        if output is None or checkpoint is None:
            continue
        try:
            summary = json.loads((output / 'fixed_task_b_summary.json').read_text())
        except (OSError, ValueError):
            continue
        adapter = checkpoint / 'final_full_data_adapter'
        if summary.get('resume_signature') == signature and (adapter / 'adapter_config.json').is_file() and (
            (adapter / 'adapter_model.safetensors').is_file() or (adapter / 'adapter_model.bin').is_file()
        ):
            return catalog, summary, adapter
    return None


def _cached_predictions(output, summary, prediction_frame, task, prediction_hash, monitor):
    if summary.get('prediction_sha256') != prediction_hash or summary.get('validation_dataset') != monitor.summary:
        return None
    try:
        with np.load(output / 'prediction_scores.npz', allow_pickle=False) as values:
            if values['ids'].tolist() != prediction_frame['id'].tolist() or values['labels'].tolist() != list(task.labels):
                return None
            probabilities = values['probabilities'].copy()
            validation = values['validation_probabilities'].copy()
        rows = 0 if monitor.data is None else len(monitor.data)
        for scores, size in ((probabilities, len(prediction_frame)), (validation, rows)):
            if scores.shape != (size, len(task.labels)) or not np.isfinite(scores).all() or (scores < 0).any():
                return None
            if size and not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
                return None
        if monitor.data is not None and not isinstance(summary.get('validation_metrics'), dict):
            return None
        return {'probabilities': probabilities, 'validation_probabilities': validation,
                'validation_metrics': summary.get('validation_metrics'), 'summary': summary}
    except (OSError, ValueError, KeyError):
        return None


def _copy_tree(source, destination):
    if source.resolve() != destination.resolve():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def run_fixed_task_b(task, config, resume_sources=(), auto_resume=True):
    """Run Khangtest's current Task B strategy with epoch and completed-seed resume."""
    epochs = task.fixed_full_retrain_epochs
    seeds = task.selection_seeds or (config.seed,)
    if epochs is None or epochs < 1 or not task.full_retrain_soft_ensemble:
        raise ValueError('Fixed Task B requires positive epochs and a soft ensemble.')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Task B soft ensemble requires at least two unique nonnegative seeds.')
    if task.selection_train_label_caps or task.use_augmentation or task.class_weight_power is None:
        raise ValueError('Fixed Task B uses original-only full data, class-weighted loss and no label caps.')
    runtime = initialize_runtime()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        require_cuda(runtime)
        set_global_seed(config.seed)
        frame, prediction_frame = load_task_frames(task)
        if frame.empty or prediction_frame.empty:
            raise ValueError('Training and prediction inputs must not be empty.')
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        encoder = ChatEncoder(tokenizer, task, config.max_length)
        collator = WeightedCollator(tokenizer.pad_token_id)
        training = WeightedDataset(prompt_inputs(frame, task), encoder, frame['label_id'].tolist())
        prediction = CompletionDataset(prompt_inputs(prediction_frame, task), encoder)
        weights, weight_summary = _class_weight_state(task, frame['label_id'].to_numpy())
        monitor = TaskBValidationMonitor(task, config, runtime, frame, encoder, collator)
        output_dir, checkpoint_dir = OUTPUT_ROOT / task.output_subdir, CHECKPOINT_ROOT / task.checkpoint_subdir
        on_main(lambda: output_dir.mkdir(parents=True, exist_ok=True), runtime)
        on_main(lambda: checkpoint_dir.mkdir(parents=True, exist_ok=True), runtime)
        training_hash, prediction_hash = _input_hashes(task)
        signature = _resume_signature(task, config, training_hash, runtime)
        signature.update({'pipeline': 'khangtest-fixed-soft-v1', 'fixed_epochs': epochs,
                          'prompt_style': task.prompt_style, 'class_weight_power': task.class_weight_power,
                          'class_weight_min': task.class_weight_min, 'class_weight_max': task.class_weight_max})
        sources = list(resume_sources)
        if auto_resume and Path('/kaggle/input').is_dir():
            sources.append(Path('/kaggle/input'))
        sources = list(dict.fromkeys(sources))
        manifest = checkpoint_dir / 'resume_catalog.json'

        def prepare():
            scratch = Path(os.getenv('HASTIKA_RESUME_CACHE_DIR', f'/tmp/hastika_resume_cache/{task.checkpoint_subdir}'))
            if scratch.exists():
                shutil.rmtree(scratch)
            write_catalog_manifest(manifest, prepare_resume_catalogs(sources, scratch, task.output_subdir, task.checkpoint_subdir))

        on_main(prepare, runtime)
        synchronize(runtime)
        catalogs = [ResumeCatalog('current run', output_dir, checkpoint_dir), *read_catalog_manifest(manifest)]
        diagnostics = completion_token_diagnostics(encoder)
        log_completion_token_diagnostics(diagnostics, runtime)
        log(f'{task.task_name} | {config.model_name} | Khangtest fixed full-data epochs={epochs} | seeds={seeds} | rows={len(frame)} | max_length={config.max_length} | world_size={runtime.world_size}', runtime)
        log('Class weights: ' + ' | '.join(f'{label}={weight_summary["weights"][label]:.4f}' for label in task.labels), runtime)
        results = []
        for seed in seeds:
            seed_output, seed_checkpoint = output_dir / f'seed_{seed}', checkpoint_dir / f'seed_{seed}'
            adapter = seed_checkpoint / 'final_full_data_adapter'
            on_main(lambda: seed_output.mkdir(parents=True, exist_ok=True), runtime)
            on_main(lambda: seed_checkpoint.mkdir(parents=True, exist_ok=True), runtime)
            finished = _finished_seed(catalogs, seed, signature)
            if finished is not None:
                catalog, summary, source_adapter = finished
                cached = _cached_predictions(catalog.seed_output(seed), summary, prediction_frame, task, prediction_hash, monitor)
                if cached is not None:
                    on_main(lambda: _copy_tree(catalog.seed_output(seed), seed_output), runtime)
                    on_main(lambda: _copy_tree(source_adapter, adapter), runtime)
                    cached['summary'] = {**cached['summary'], 'checkpoint': str(adapter)}
                    on_main(lambda: _write_json(seed_output / 'fixed_task_b_summary.json', cached['summary']), runtime)
                    on_main(lambda: _write_json(seed_output / 'seed_summary.json', cached['summary']), runtime)
                    cached.update({'seed': seed, 'label_ids': cached['probabilities'].argmax(axis=1)})
                    results.append(cached)
                    log(f'Seed {seed}: completed full-data adapter and predictions reused; no training.', runtime)
                    continue
            set_global_seed(seed)
            loader, sampler = create_train_loader(training, collator, config, runtime, seed)
            loaded = load_lora_model(config, runtime)
            model = _wrap_for_ddp(loaded, runtime)
            history, completed_epoch = [], 0
            if finished is not None:
                restore_adapter(model, finished[2])
                completed_epoch = epochs
                log(f'Seed {seed}: completed adapter reused; refreshing prediction or validation.', runtime)
            else:
                optimizer, scheduler, scaler = _training_state(model, loader, epochs, config, runtime)
                latest = _latest_phase_checkpoint(catalogs, seed, 'full_retrain', signature, epochs, runtime.world_size)
                if latest is not None:
                    completed_epoch, history = _restore_epoch_resume(
                        checkpoint=latest[0], expected_signature=signature, expected_total_epochs=epochs,
                        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, loader=loader, runtime=runtime,
                    )
                    log(f'Seed {seed}: resume after completed epoch {completed_epoch}.', runtime)
                for epoch in range(completed_epoch + 1, epochs + 1):
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    train_loss = train_weighted_epoch(model, loader, optimizer, scheduler, scaler, config, runtime, epoch,
                                                      f'Fixed full-data train seed {seed}', class_weights=weights)
                    metrics = monitor.evaluate(model, seed, epoch, seed_output)
                    history.append({'epoch': epoch, 'train_weighted_loss': train_loss,
                                    'official_validation': None if metrics is None else public_validation_metrics(metrics)})
                    on_main(lambda: _write_json(seed_output / 'full_data_training_history.json',
                        {'seed': seed, 'epoch_selection': 'fixed_without_oof', 'validation_dataset': monitor.summary,
                         'class_weighting': weight_summary, 'epochs': history}), runtime)
                    _save_epoch_resume(phase='full_retrain', epoch=epoch, total_epochs=epochs, model=model, tokenizer=tokenizer,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler, loader=loader, history=list(history),
                        seed_checkpoint_dir=seed_checkpoint, signature=signature, runtime=runtime)
                del optimizer, scheduler, scaler
            on_main(partial(save_adapter, model, tokenizer, adapter), runtime)
            scores = score_candidates(model, prediction, collator, config, runtime, f'Development prediction seed {seed}')
            validation = monitor.evaluate(model, seed, epochs, seed_output) if finished is not None or completed_epoch == epochs else metrics
            probabilities = scores['probabilities']
            label_ids = probabilities.argmax(axis=1)
            validation_probabilities = np.empty((0, len(task.labels))) if validation is None else validation['probabilities']
            summary = {'seed': seed, 'epochs': epochs, 'training_rows': len(frame), 'checkpoint': str(adapter),
                       'resume_signature': signature, 'prediction_sha256': prediction_hash, 'validation_dataset': monitor.summary,
                       'validation_metrics': None if validation is None else public_validation_metrics(validation)}

            def export_seed():
                write_submission(prediction_frame['id'].tolist(), [task.labels[index] for index in label_ids],
                                 task.labels, seed_output, task.submission_zip_name)
                temporary = seed_output / '.prediction_scores.tmp.npz'
                np.savez_compressed(temporary, ids=prediction_frame['id'].to_numpy(dtype=str), labels=np.asarray(task.labels),
                                    probabilities=probabilities, validation_probabilities=validation_probabilities)
                temporary.replace(seed_output / 'prediction_scores.npz')
                _write_json(seed_output / 'seed_summary.json', summary)
                _write_json(seed_output / 'fixed_task_b_summary.json', summary)

            on_main(export_seed, runtime)
            _clear_phase_resume(seed_checkpoint, 'full_retrain', runtime)
            results.append({'seed': seed, 'probabilities': probabilities, 'label_ids': label_ids,
                            'validation_probabilities': validation_probabilities,
                            'validation_metrics': summary['validation_metrics'], 'summary': summary})
            del model, loaded, loader
            gc.collect()
            torch.cuda.empty_cache()
        validation_summary = monitor.evaluate_ensemble(results, output_dir)
        mean = np.mean(np.stack([result['probabilities'] for result in results]), axis=0)
        final_labels = [task.labels[index] for index in mean.argmax(axis=1)]

        def export_final():
            audit = {'id': prediction_frame['id'].tolist()}
            for result in results:
                audit[f'seed_{result["seed"]}'] = [task.labels[index] for index in result['label_ids']]
            for index, label in enumerate(task.labels):
                safe = label.lower().replace('-', '_').replace(' ', '_')
                audit[f'mean_prob_{safe}'] = mean[:, index]
                audit[f'votes_{safe}'] = np.sum(np.stack([result['label_ids'] for result in results]) == index, axis=0)
            audit['final_label'] = final_labels
            pd.DataFrame(audit).to_csv(output_dir / 'ensemble_predictions.csv', index=False)
            write_submission(prediction_frame['id'].tolist(), final_labels, task.labels, output_dir, task.submission_zip_name)
            summary = {'task': task.task_name, 'reference_branch': 'Khangtest', 'reference_commit': 'e27514b',
                       'model': config.model_name, 'training_config': asdict(config), 'world_size': runtime.world_size,
                       'epoch_selection': {'strategy': 'fixed_without_oof', 'oof_skipped': True, 'fixed_epoch': epochs},
                       'class_weighting': {'downsampling': None, 'full_data': weight_summary},
                       'ensemble': {'method': 'equal-weight mean of per-seed candidate probabilities', 'seeds': list(seeds)},
                       'full_data_retrains': [result['summary'] for result in results],
                       'official_validation': {'dataset': monitor.summary, 'ensemble_metrics': validation_summary},
                       'prompt_input': {'input_mode': task.input_mode, 'prompt_style': task.prompt_style,
                                        'raw_column': task.original_comment_column, 'english_column': 'Comment'},
                       'truncated_comments': {'full_train': training.truncated_count, 'prediction': prediction.truncated_count},
                       'completion_token_diagnostics': diagnostics, 'resume_signature': signature}
            _write_json(output_dir / 'run_summary.json', summary)
            _write_json(checkpoint_dir / 'metrics.json', summary)

        on_main(export_final, runtime)
        log(f'Task B complete | fixed epochs={epochs} | soft ensemble seeds={seeds} | {output_dir}', runtime)
    finally:
        cleanup_runtime(runtime)
