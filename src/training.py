"""Shared, explicit 5-fold training engine for HASTIKA Task A and Task B."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import (
    CHECKPOINT_ROOT,
    OUTPUT_ROOT,
    TaskConfig,
    TrainingConfig,
)
from src.data import (
    ClassificationCollator,
    DistributedSliceSampler,
    EncodedTextDataset,
    load_task_frames,
)
from src.distributed import (
    RuntimeContext,
    cleanup_runtime,
    gather_python_objects,
    initialize_runtime,
    synchronize,
)
from src.submission import write_submission


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch reproducibly on every DDP worker."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(message: str, runtime: RuntimeContext) -> None:
    """Print once instead of duplicating every message across DDP workers."""

    if runtime.is_main_process:
        print(message, flush=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original Hugging Face model beneath a DDP wrapper."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def move_model_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]:
    """Move token tensors to a GPU and separate metadata from model inputs."""

    sample_indices = batch.pop("sample_index")
    labels = batch.pop("labels", None)
    model_inputs = {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }
    moved_labels = None if labels is None else labels.to(device, non_blocking=True)
    return model_inputs, moved_labels, sample_indices


def build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW with no decay on bias and LayerNorm parameters."""

    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            no_decay_parameters
            if any(term in name for term in no_decay_terms)
            else decay_parameters
        )
        target.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return AdamW(parameter_groups, lr=config.learning_rate)


def compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    """Compute balanced weights N / (K * N_c) from one fold's train split."""

    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            f"At least one class is missing from a training fold: {counts.tolist()}"
        )
    weights = len(labels) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_train_loader(
    dataset: EncodedTextDataset,
    collator: ClassificationCollator,
    config: TrainingConfig,
    runtime: RuntimeContext,
    fold_seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create a shuffled loader with exactly one per-GPU batch per worker."""

    sampler: DistributedSampler | None = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=fold_seed,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(fold_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size_per_gpu,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )
    return loader, sampler


def create_eval_loader(
    dataset: EncodedTextDataset,
    collator: ClassificationCollator,
    config: TrainingConfig,
    runtime: RuntimeContext,
) -> DataLoader:
    """Create a duplicate-free evaluation/prediction loader for each worker."""

    sampler = DistributedSliceSampler(len(dataset), runtime.rank, runtime.world_size)
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    runtime: RuntimeContext,
    epoch_number: int,
) -> float:
    """Run one epoch and return the sample-weighted mean loss."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    amp_enabled = config.use_amp and runtime.device.type == "cuda"
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch_number} train",
        disable=not runtime.is_main_process,
        leave=False,
    )

    for step, batch in enumerate(progress):
        model_inputs, labels, _ = move_model_inputs(batch, runtime.device)
        if labels is None:
            raise RuntimeError("Training batches must contain labels.")
        is_last_step = step + 1 == len(loader)
        should_update = (
            step + 1
        ) % config.gradient_accumulation_steps == 0 or is_last_step
        sync_context = nullcontext()
        if isinstance(model, DistributedDataParallel) and not should_update:
            sync_context = model.no_sync()

        with sync_context:
            with torch.amp.autocast(
                device_type=runtime.device.type, enabled=amp_enabled
            ):
                logits = model(**model_inputs).logits
                raw_loss = criterion(logits, labels)
                loss = raw_loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

        batch_size = labels.size(0)
        total_loss += float(raw_loss.detach().item()) * batch_size
        total_examples += batch_size

        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if runtime.is_main_process:
            progress.set_postfix(loss=f"{raw_loss.detach().item():.4f}")

    gathered = gather_python_objects((total_loss, total_examples), runtime)
    global_loss = sum(float(item[0]) for item in gathered)
    global_examples = sum(int(item[1]) for item in gathered)
    return global_loss / max(global_examples, 1)


@torch.no_grad()
def predict_distributed(
    model: nn.Module,
    loader: DataLoader,
    runtime: RuntimeContext,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Predict unique shards on all GPUs, gather them, and restore row order."""

    model.eval()
    local_indices: list[int] = []
    local_probabilities: list[np.ndarray] = []
    local_labels: list[int] = []
    amp_enabled = runtime.device.type == "cuda"
    progress = tqdm(
        loader,
        desc=description,
        disable=not runtime.is_main_process,
        leave=False,
    )

    for batch in progress:
        model_inputs, labels, sample_indices = move_model_inputs(batch, runtime.device)
        with torch.amp.autocast(device_type=runtime.device.type, enabled=amp_enabled):
            logits = model(**model_inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        local_indices.extend(sample_indices.tolist())
        local_probabilities.append(probabilities)
        if labels is not None:
            local_labels.extend(labels.cpu().tolist())

    probability_width = unwrap_model(model).config.num_labels
    probability_array = (
        np.concatenate(local_probabilities, axis=0)
        if local_probabilities
        else np.empty((0, probability_width), dtype=np.float32)
    )
    payload = (
        np.asarray(local_indices, dtype=np.int64),
        probability_array,
        np.asarray(local_labels, dtype=np.int64) if local_labels else None,
    )
    gathered = gather_python_objects(payload, runtime)
    all_indices = np.concatenate([item[0] for item in gathered], axis=0)
    all_probabilities = np.concatenate([item[1] for item in gathered], axis=0)
    label_parts = [item[2] for item in gathered if item[2] is not None]
    all_labels = np.concatenate(label_parts, axis=0) if label_parts else None

    order = np.argsort(all_indices)
    ordered_indices = all_indices[order]
    expected_indices = np.arange(len(ordered_indices), dtype=np.int64)
    if not np.array_equal(ordered_indices, expected_indices):
        raise RuntimeError("Distributed prediction lost or duplicated dataset rows.")
    ordered_labels = None if all_labels is None else all_labels[order]
    return ordered_indices, all_probabilities[order], ordered_labels


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_labels: int,
    runtime: RuntimeContext,
    description: str,
) -> dict[str, Any]:
    """Calculate validation loss, accuracy and the official Macro-F1 metric."""

    model.eval()
    local_loss = 0.0
    local_examples = 0
    local_indices: list[int] = []
    local_probabilities: list[np.ndarray] = []
    local_labels: list[int] = []
    amp_enabled = runtime.device.type == "cuda"
    progress = tqdm(
        loader,
        desc=description,
        disable=not runtime.is_main_process,
        leave=False,
    )

    for batch in progress:
        model_inputs, labels, sample_indices = move_model_inputs(batch, runtime.device)
        if labels is None:
            raise RuntimeError("Validation batches must contain labels.")
        with torch.amp.autocast(device_type=runtime.device.type, enabled=amp_enabled):
            logits = model(**model_inputs).logits
            loss = criterion(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        local_loss += float(loss.item()) * labels.size(0)
        local_examples += labels.size(0)
        local_indices.extend(sample_indices.tolist())
        local_probabilities.append(probabilities)
        local_labels.extend(labels.cpu().tolist())

    payload = (
        local_loss,
        local_examples,
        np.asarray(local_indices, dtype=np.int64),
        np.concatenate(local_probabilities, axis=0),
        np.asarray(local_labels, dtype=np.int64),
    )
    gathered = gather_python_objects(payload, runtime)
    total_loss = sum(float(item[0]) for item in gathered)
    total_examples = sum(int(item[1]) for item in gathered)
    indices = np.concatenate([item[2] for item in gathered], axis=0)
    probabilities = np.concatenate([item[3] for item in gathered], axis=0)
    labels = np.concatenate([item[4] for item in gathered], axis=0)

    order = np.argsort(indices)
    indices = indices[order]
    probabilities = probabilities[order]
    labels = labels[order]
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise RuntimeError("Distributed validation lost or duplicated dataset rows.")

    predictions = probabilities.argmax(axis=1)
    macro_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(num_labels)),
        average="macro",
        zero_division=0,
    )
    return {
        "loss": total_loss / max(total_examples, 1),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "probabilities": probabilities,
        "labels": labels,
    }


def save_model_state(model: nn.Module, path: Path) -> None:
    """Save only model weights to a portable CPU checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu()
        for name, tensor in unwrap_model(model).state_dict().items()
    }
    torch.save(state, path)
    del state


def load_model_state(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load a best-fold checkpoint on every DDP worker."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch 2.x images.
        state = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(state)
    unwrap_model(model).to(device)
    del state


def write_oof_predictions(
    train_df: pd.DataFrame,
    task: TaskConfig,
    oof_probabilities: np.ndarray,
    output_dir: Path,
) -> Path:
    """Store out-of-fold probabilities for error analysis and reproducibility."""

    predicted_ids = oof_probabilities.argmax(axis=1)
    output = pd.DataFrame(
        {
            "id": train_df["id"].astype(str),
            "true_label": train_df[task.label_column],
            "predicted_label": [task.labels[index] for index in predicted_ids],
        }
    )
    for label_index, label in enumerate(task.labels):
        safe_name = label.lower().replace("-", "_").replace(" ", "_")
        output[f"prob_{safe_name}"] = oof_probabilities[:, label_index]
    path = output_dir / "oof_predictions.csv"
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def run_cross_validated_task(task: TaskConfig, config: TrainingConfig) -> None:
    """Execute preprocessing, 5-fold training, ensemble, and submission output."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime = initialize_runtime()
    try:
        set_global_seed(config.seed)
        log(f"\n=== {task.task_name} ===", runtime)
        log(
            f"Device={runtime.device}; world_size={runtime.world_size}; "
            f"batch/GPU={config.train_batch_size_per_gpu}; "
            f"effective batch={config.train_batch_size_per_gpu * runtime.world_size * config.gradient_accumulation_steps}",
            runtime,
        )

        train_df, test_df = load_task_frames(task, config.preprocess)
        log(
            f"Loaded {len(train_df)} training rows and {len(test_df)} prediction rows.",
            runtime,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        collator = ClassificationCollator(tokenizer, config.pad_to_multiple_of)
        labels = train_df["label_id"].to_numpy(dtype=np.int64)
        splitter = StratifiedKFold(
            n_splits=config.num_folds,
            shuffle=True,
            random_state=config.seed,
        )
        folds = list(splitter.split(np.zeros(len(labels)), labels))

        checkpoint_root = CHECKPOINT_ROOT / task.checkpoint_subdir
        output_dir = OUTPUT_ROOT / task.output_subdir
        if runtime.is_main_process:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
        synchronize(runtime)

        test_probability_sum = np.zeros(
            (len(test_df), len(task.labels)), dtype=np.float64
        )
        oof_probabilities = np.zeros(
            (len(train_df), len(task.labels)), dtype=np.float32
        )
        fold_summaries: list[dict[str, Any]] = []

        for fold_number, (train_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            fold_seed = config.seed + fold_number
            set_global_seed(fold_seed)
            log(
                f"\n--- Fold {fold_number}/{config.num_folds}: "
                f"train={len(train_indices)}, validation={len(validation_indices)} ---",
                runtime,
            )

            fold_train = train_df.iloc[train_indices]
            fold_validation = train_df.iloc[validation_indices]
            train_dataset = EncodedTextDataset(
                fold_train["clean_text"].tolist(),
                tokenizer,
                config.max_length,
                fold_train["label_id"].tolist(),
            )
            validation_dataset = EncodedTextDataset(
                fold_validation["clean_text"].tolist(),
                tokenizer,
                config.max_length,
                fold_validation["label_id"].tolist(),
            )
            test_dataset = EncodedTextDataset(
                test_df["clean_text"].tolist(),
                tokenizer,
                config.max_length,
            )
            train_loader, train_sampler = create_train_loader(
                train_dataset, collator, config, runtime, fold_seed
            )
            validation_loader = create_eval_loader(
                validation_dataset, collator, config, runtime
            )
            test_loader = create_eval_loader(test_dataset, collator, config, runtime)

            id_to_label = {index: label for index, label in enumerate(task.labels)}
            label_to_id = {label: index for index, label in enumerate(task.labels)}
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=len(task.labels),
                id2label=id_to_label,
                label2id=label_to_id,
            )
            model.to(runtime.device)
            if runtime.distributed:
                if runtime.device.type == "cuda":
                    model = DistributedDataParallel(
                        model,
                        device_ids=[runtime.local_rank],
                        output_device=runtime.local_rank,
                    )
                else:
                    model = DistributedDataParallel(model)

            optimizer = build_optimizer(model, config)
            updates_per_epoch = math.ceil(
                len(train_loader) / config.gradient_accumulation_steps
            )
            total_updates = updates_per_epoch * config.max_epochs
            warmup_steps = int(total_updates * config.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_updates,
            )
            amp_enabled = config.use_amp and runtime.device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

            class_weights = None
            if task.use_class_weights:
                class_weights = compute_class_weights(
                    fold_train["label_id"].to_numpy(dtype=np.int64),
                    len(task.labels),
                ).to(runtime.device)
                log(f"Fold class weights: {class_weights.cpu().tolist()}", runtime)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_metric = -math.inf
            best_epoch = 0
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_root / f"fold_{fold_number}" / "best_model.pt"

            for epoch in range(1, config.max_epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    runtime,
                    epoch,
                )
                validation_metrics = evaluate_model(
                    model,
                    validation_loader,
                    criterion,
                    len(task.labels),
                    runtime,
                    f"Epoch {epoch} validation",
                )
                current_metric = float(validation_metrics["macro_f1"])
                log(
                    f"Fold {fold_number} | epoch {epoch} | "
                    f"train_loss={train_loss:.5f} | "
                    f"val_loss={validation_metrics['loss']:.5f} | "
                    f"val_macro_f1={current_metric:.5f} | "
                    f"val_accuracy={validation_metrics['accuracy']:.5f}",
                    runtime,
                )

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if runtime.is_main_process:
                        save_model_state(model, checkpoint_path)
                else:
                    epochs_without_improvement += 1

                synchronize(runtime)
                if epochs_without_improvement >= config.early_stopping_patience:
                    log(
                        f"Early stopping fold {fold_number} after epoch {epoch}.",
                        runtime,
                    )
                    break

            del optimizer, scheduler, scaler
            synchronize(runtime)
            load_model_state(model, checkpoint_path, runtime.device)
            synchronize(runtime)

            best_validation = evaluate_model(
                model,
                validation_loader,
                criterion,
                len(task.labels),
                runtime,
                f"Fold {fold_number} best checkpoint",
            )
            _, fold_test_probabilities, _ = predict_distributed(
                model,
                test_loader,
                runtime,
                f"Fold {fold_number} prediction",
            )
            if runtime.is_main_process:
                oof_probabilities[validation_indices] = best_validation["probabilities"]
                test_probability_sum += fold_test_probabilities
                fold_summary = {
                    "fold": fold_number,
                    "best_epoch": best_epoch,
                    "best_macro_f1": float(best_validation["macro_f1"]),
                    "checkpoint": str(checkpoint_path),
                }
                fold_summaries.append(fold_summary)
                metadata_path = checkpoint_path.parent / "metrics.json"
                metadata_path.write_text(
                    json.dumps(fold_summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            del model, criterion, class_weights
            del train_loader, validation_loader, test_loader
            del train_dataset, validation_dataset, test_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            synchronize(runtime)

        if runtime.is_main_process:
            mean_test_probabilities = test_probability_sum / config.num_folds
            predicted_label_ids = mean_test_probabilities.argmax(axis=1)
            predicted_labels = [task.labels[index] for index in predicted_label_ids]
            predictions_path, zip_path = write_submission(
                ids=test_df["id"].tolist(),
                labels=predicted_labels,
                allowed_labels=task.labels,
                output_dir=output_dir,
                zip_name=task.submission_zip_name,
            )

            oof_predictions = oof_probabilities.argmax(axis=1)
            oof_macro_f1 = f1_score(
                labels,
                oof_predictions,
                labels=list(range(len(task.labels))),
                average="macro",
                zero_division=0,
            )
            oof_path = write_oof_predictions(
                train_df, task, oof_probabilities, output_dir
            )
            run_summary = {
                "task": task.task_name,
                "model": config.model_name,
                "world_size": runtime.world_size,
                "effective_train_batch_size": (
                    config.train_batch_size_per_gpu
                    * runtime.world_size
                    * config.gradient_accumulation_steps
                ),
                "oof_macro_f1": float(oof_macro_f1),
                "folds": fold_summaries,
                "training_config": asdict(config),
            }
            summary_path = output_dir / "run_summary.json"
            summary_path.write_text(
                json.dumps(run_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            log(f"\nOOF Macro-F1: {oof_macro_f1:.5f}", runtime)
            log(f"Predictions CSV: {predictions_path}", runtime)
            log(f"Submission ZIP:  {zip_path}", runtime)
            log(f"OOF diagnostics: {oof_path}", runtime)
            log(f"Run summary:     {summary_path}", runtime)
    finally:
        cleanup_runtime(runtime)
