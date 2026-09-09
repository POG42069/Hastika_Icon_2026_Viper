"""Answer-only QLoRA SFT, exact candidate scoring, and CodaBench export.

Production entry points require CUDA and never silently replace QLoRA with full
fine-tuning. The small pure-PyTorch helpers can also be tested on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from config import CHECKPOINT_ROOT, OUTPUT_ROOT, PreprocessConfig, TaskConfig
from llm_config import LLM_TRAINING, LLMTrainingConfig
from src.data import load_task_frames, resolve_data_file
from src.distributed import (
    RuntimeContext,
    cleanup_runtime,
    gather_python_objects,
    initialize_runtime,
    relaunch_with_torchrun_if_needed,
)
from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
from src.submission import write_submission
from src.training import (
    build_optimizer,
    create_train_loader,
    is_better_checkpoint,
    log,
    prepare_nltk_resources,
    set_global_seed,
    stratified_holdout_indices,
    unwrap_model,
    write_split_manifest,
    write_validation_predictions,
)


def validate_config(config: LLMTrainingConfig) -> None:
    for name in (
        "max_epochs",
        "max_length",
        "train_batch_size_per_gpu",
        "eval_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "lora_rank",
        "early_stopping_patience",
        "max_gpus",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be strictly between zero and one")
    if config.text_mode not in {"paper", "raw"}:
        raise ValueError("text_mode must be 'paper' or 'raw'")
    if config.compute_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("compute_dtype must be auto, float16 or bfloat16")
    if config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if config.initial_loss_scale <= 0:
        raise ValueError("initial_loss_scale must be positive")
    if not 0 <= config.warmup_ratio < 1 or not 0 <= config.lora_dropout < 1:
        raise ValueError("warmup_ratio and lora_dropout must be in [0, 1)")


def compute_dtype(config: LLMTrainingConfig, runtime: RuntimeContext) -> torch.dtype:
    if runtime.device.type != "cuda":
        return torch.float32
    # Recent PyTorch versions also count emulated BF16 as supported. Require
    # native NVIDIA BF16 (Ampere or newer), so T4 always selects FP16.
    bf16_supported = (
        torch.cuda.get_device_capability(runtime.device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    if config.compute_dtype == "bfloat16" and not bf16_supported:
        raise ValueError("This GPU cannot use BF16; select auto or float16 (T4).")
    return (
        torch.bfloat16
        if config.compute_dtype == "bfloat16"
        or (config.compute_dtype == "auto" and bf16_supported)
        else torch.float16
    )


def require_cuda(runtime: RuntimeContext) -> None:
    if runtime.device.type != "cuda":
        raise RuntimeError(
            "QLoRA requires CUDA. Select GPU T4 x2 on Kaggle; CPU is only for tests."
        )


def _autocast(runtime: RuntimeContext, dtype: torch.dtype):
    return (
        torch.autocast("cuda", dtype=dtype)
        if runtime.device.type == "cuda"
        else nullcontext()
    )


def on_main(action: Callable[[], None], runtime: RuntimeContext) -> None:
    """Broadcast rank-zero artifact errors, rather than stranding other ranks."""
    error = None
    if runtime.is_main_process:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - propagate any writer failure to every rank.
            error = f"{type(exc).__name__}: {exc}"
    errors = gather_python_objects(error, runtime)
    if any(item is not None for item in errors):
        raise RuntimeError(next(item for item in errors if item is not None))


def load_qlora_model(config: LLMTrainingConfig, runtime: RuntimeContext):
    """Put each worker's entire quantized model on its own GPU, never auto-shard."""
    require_cuda(runtime)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    dtype = compute_dtype(config, runtime)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        dtype=dtype,
        device_map={"": runtime.device.index},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError(
            "The backbone was not loaded in 4-bit; refusing a non-QLoRA run."
        )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        ),
    )
    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA adapters to be trainable.")
    if runtime.is_main_process:
        model.print_trainable_parameters()
    return model


def completion_log_probs(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log P(full answer | prompt) and supervised token counts per row.

    Supervision is shifted exactly once: logits BEFORE each answer token predict
    that token. Project only the short suffix to the large vocabulary where the
    architecture supports logits_to_keep (Qwen2.5); preserve the full attention
    context. Left padding lets different prompt lengths share a short suffix.
    """
    labels = batch["labels"]
    positions = torch.nonzero(labels != -100, as_tuple=False)
    if positions.numel() == 0 or int(positions[:, 1].min()) < 1:
        raise ValueError(
            "Every completion needs preceding context and supervised tokens."
        )
    keep = labels.shape[1] - int(positions[:, 1].min()) + 1
    base = unwrap_model(model)
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    kwargs = {key: value for key, value in batch.items() if key != "labels"}
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        kwargs["logits_to_keep"] = keep
    logits = model(**kwargs).logits
    targets = labels[:, -logits.shape[1] :][:, 1:]
    shifted_logits = logits[:, :-1]
    mask = targets != -100
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Each row must have at least one supervised answer token.")
    token_losses = F.cross_entropy(
        shifted_logits[mask].float(), targets[mask], reduction="none"
    )
    row_ids = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(
        mask
    )[mask]
    losses = token_losses.new_zeros(labels.shape[0]).scatter_add(
        0, row_ids, token_losses
    )
    return -losses, counts


def train_llm_epoch(
    model, loader, optimizer, scheduler, scaler, config, runtime, epoch
):
    """Mean answer-token NLL per example; correctly scale even a partial group."""
    model.train()
    dtype = compute_dtype(config, runtime)
    total_loss, total_examples, successful_updates, skipped_updates = 0.0, 0, 0, 0
    iterator = iter(loader)
    progress = tqdm(
        total=len(loader),
        desc=f"LLM epoch {epoch}",
        disable=not runtime.is_main_process,
        leave=False,
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
                    log_probs, counts = completion_log_probs(model, batch)
                    example_losses = -log_probs / counts
                    loss = example_losses.sum() / group_size
                scaler.scale(loss).backward()
            total_loss += float(example_losses.detach().sum())
            total_examples += len(example_losses)
            progress.update(1)
            progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")
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
    progress.close()
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


@torch.no_grad()
def score_candidates(model, dataset, collator, config, runtime, description):
    """Score every full legal answer, restore row order and compute fixed-class F1.

    Sum token log probabilities including the chat end suffix. Softmax produces
    probabilities conditional on this finite candidate set (not calibrated
    confidence). No sampling or free-text label parsing is used.
    """
    model = unwrap_model(model)  # No DDP forward collectives on uneven eval shards.
    model.eval()
    dtype = compute_dtype(config, runtime)
    n_labels = len(dataset.encoder.task.labels)
    candidates = (
        (row, label, dataset.encoder.feature(dataset.prompts[row], label))
        for row in range(runtime.rank, len(dataset), runtime.world_size)
        for label in range(n_labels)
    )
    local = []
    count = len(range(runtime.rank, len(dataset), runtime.world_size)) * n_labels
    progress = tqdm(
        total=count, desc=description, disable=not runtime.is_main_process, leave=False
    )
    while group := list(islice(candidates, config.eval_batch_size_per_gpu)):
        batch = {
            key: value.to(runtime.device)
            for key, value in collator([x[2] for x in group]).items()
        }
        with _autocast(runtime, dtype):
            scores, counts = completion_log_probs(model, batch)
        for (row, label, _), score, tokens in zip(
            group, scores.cpu().tolist(), counts.cpu().tolist(), strict=True
        ):
            local.append((row, label, score, tokens))
        progress.update(len(group))
    progress.close()
    gathered = [
        item for shard in gather_python_objects(local, runtime) for item in shard
    ]
    gathered.sort(key=lambda item: (item[0], item[1]))
    expected = [
        (row, label) for row in range(len(dataset)) for label in range(n_labels)
    ]
    if [(item[0], item[1]) for item in gathered] != expected:
        raise RuntimeError("Candidate scoring lost or duplicated rows/labels.")
    scores = np.asarray([item[2] for item in gathered], dtype=np.float64).reshape(
        -1, n_labels
    )
    counts = np.asarray([item[3] for item in gathered], dtype=np.float64).reshape(
        -1, n_labels
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite candidate scores; check AMP and training stability."
        )
    probabilities = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
    result = {"probabilities": probabilities, "sequence_log_probs": scores}
    if dataset.labels is not None:
        labels = np.asarray(dataset.labels, dtype=np.int64)
        predictions = probabilities.argmax(axis=1)
        row = np.arange(len(labels))
        result.update(
            loss=float(np.mean(-scores[row, labels] / counts[row, labels])),
            macro_f1=float(
                f1_score(
                    labels,
                    predictions,
                    labels=list(range(n_labels)),
                    average="macro",
                    zero_division=0,
                )
            ),
            accuracy=float(accuracy_score(labels, predictions)),
        )
    return result


def save_adapter(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def restore_adapter(model, path: Path) -> None:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(str(path), device="cpu")
    result = set_peft_model_state_dict(
        unwrap_model(model), weights, adapter_name="default"
    )
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys}")
    if any("lora_" in name for name in result.missing_keys):
        raise RuntimeError("The checkpoint is missing LoRA adapter weights.")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_llm_task(task: TaskConfig, config: LLMTrainingConfig) -> None:
    validate_config(config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime = initialize_runtime()
    try:
        require_cuda(runtime)
        set_global_seed(config.seed)
        log(
            f"\n{task.task_name} | QLoRA {config.model_name} | world_size={runtime.world_size}",
            runtime,
        )
        log(
            f"Compute={compute_dtype(config, runtime)}; effective batch="
            f"{config.train_batch_size_per_gpu * config.gradient_accumulation_steps * runtime.world_size}",
            runtime,
        )
        preprocess = config.preprocess
        if config.text_mode == "raw":
            preprocess = PreprocessConfig(
                **{
                    key: False
                    for key, value in asdict(preprocess).items()
                    if isinstance(value, bool)
                }
            )
        if preprocess.remove_stopwords or preprocess.lemmatize_english:
            prepare_nltk_resources(runtime)
        frame, prediction_frame = load_task_frames(task, preprocess)
        for name, data in (("train", frame), ("prediction", prediction_frame)):
            if data.empty or data["id"].isna().any():
                raise ValueError(
                    f"{name} input must contain rows with non-missing IDs."
                )
        train_indices, val_indices = stratified_holdout_indices(
            frame["label_id"].to_numpy(), config.validation_size, config.seed
        )
        train_frame, val_frame = frame.iloc[train_indices], frame.iloc[val_indices]
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.model_revision, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        encoder = ChatEncoder(tokenizer, task, config.max_length)
        train_data = CompletionDataset(
            train_frame["clean_text"].tolist(),
            encoder,
            train_frame["label_id"].tolist(),
        )
        val_data = CompletionDataset(
            val_frame["clean_text"].tolist(), encoder, val_frame["label_id"].tolist()
        )
        prediction_data = CompletionDataset(
            prediction_frame["clean_text"].tolist(), encoder
        )
        collator = CompletionCollator(tokenizer.pad_token_id)
        loader, sampler = create_train_loader(
            train_data, collator, config, runtime, config.seed
        )
        output_dir = OUTPUT_ROOT / task.output_subdir
        checkpoint_dir = CHECKPOINT_ROOT / task.checkpoint_subdir
        adapter_dir = checkpoint_dir / "best_adapter"

        def initialize_artifacts():
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            write_split_manifest(frame, train_indices, val_indices, output_dir)

        on_main(initialize_artifacts, runtime)
        log(
            f"Split: train={len(train_data)}, validation={len(val_data)}; prediction={len(prediction_data)}",
            runtime,
        )
        log(
            f"Truncated comments: train={train_data.truncated_count}, validation={val_data.truncated_count}, "
            f"prediction={prediction_data.truncated_count}",
            runtime,
        )
        model = load_qlora_model(config, runtime)
        base = model.get_base_model()
        resolved_revision = getattr(base.config, "_commit_hash", None)
        parameter_counts = {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "reported_total": model.num_parameters(),
        }
        if runtime.distributed:
            model = DDP(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
        optimizer = build_optimizer(model, config)
        total_updates = (
            math.ceil(len(loader) / config.gradient_accumulation_steps)
            * config.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_updates * config.warmup_ratio), total_updates
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=compute_dtype(config, runtime) == torch.float16,
            init_scale=config.initial_loss_scale,
        )
        best_metric, best_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                model, loader, optimizer, scheduler, scaler, config, runtime, epoch
            )
            metrics = score_candidates(
                model, val_data, collator, config, runtime, f"Epoch {epoch} validation"
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{key: metrics[key] for key in ("loss", "macro_f1", "accuracy")},
            }
            history.append(record)
            log(
                f"Epoch {epoch}: train_loss={train_loss:.5f}, val_loss={metrics['loss']:.5f}, "
                f"macro_f1={metrics['macro_f1']:.5f}, accuracy={metrics['accuracy']:.5f}",
                runtime,
            )
            if is_better_checkpoint(
                metrics["macro_f1"], metrics["loss"], best_metric, best_loss
            ):
                best_metric, best_loss, best_epoch, stale = (
                    metrics["macro_f1"],
                    metrics["loss"],
                    epoch,
                    0,
                )
                on_main(partial(save_adapter, model, tokenizer, adapter_dir), runtime)
            else:
                stale += 1
            on_main(
                lambda: _write_json(
                    output_dir / "training_history.json", {"epochs": history}
                ),
                runtime,
            )
            if stale >= config.early_stopping_patience:
                log(f"Early stopping at epoch {epoch}.", runtime)
                break
        del optimizer, scheduler, scaler
        restore_adapter(model, adapter_dir)
        best_validation = score_candidates(
            model, val_data, collator, config, runtime, "Best adapter validation"
        )
        predictions = score_candidates(
            model,
            prediction_data,
            collator,
            config,
            runtime,
            "External input prediction",
        )

        def export_artifacts():
            label_ids = predictions["probabilities"].argmax(axis=1)
            csv_path, zip_path = write_submission(
                prediction_frame["id"].tolist(),
                [task.labels[i] for i in label_ids],
                task.labels,
                output_dir,
                task.submission_zip_name,
            )
            write_validation_predictions(
                val_frame, task, best_validation["probabilities"], output_dir
            )
            summary = {
                "best_epoch": best_epoch,
                "checkpoint": str(adapter_dir),
                **{
                    key: best_validation[key]
                    for key in ("macro_f1", "accuracy", "loss")
                },
            }
            _write_json(checkpoint_dir / "metrics.json", summary)
            _write_json(
                output_dir / "run_summary.json",
                {
                    "task": task.task_name,
                    "method": "QLoRA answer-only causal LM",
                    "model": config.model_name,
                    "resolved_model_revision": resolved_revision,
                    "world_size": runtime.world_size,
                    "compute_dtype": str(compute_dtype(config, runtime)),
                    "effective_train_batch_size": config.train_batch_size_per_gpu
                    * config.gradient_accumulation_steps
                    * runtime.world_size,
                    "split": {
                        "seed": config.seed,
                        "validation_size": config.validation_size,
                        "train_rows": len(train_data),
                        "validation_rows": len(val_data),
                        "prediction_rows": len(prediction_data),
                        "train_label_distribution": train_frame[task.label_column]
                        .value_counts()
                        .to_dict(),
                        "validation_label_distribution": val_frame[task.label_column]
                        .value_counts()
                        .to_dict(),
                    },
                    "best_checkpoint": summary,
                    "training_config": asdict(config),
                    "effective_preprocess": asdict(preprocess),
                    "parameters": parameter_counts,
                    "system_prompt": encoder.system_text,
                    "label_order": list(task.labels),
                    "completion_token_ids": encoder.completions,
                    "inference": "argmax sum log P(answer tokens including chat end | prompt)",
                    "probabilities": "softmax over allowed full-answer log likelihoods; not calibrated",
                    "loss": "mean per-example answer-token NLL; not class cross-entropy",
                    "truncated_comments": {
                        "train": train_data.truncated_count,
                        "validation": val_data.truncated_count,
                        "prediction": prediction_data.truncated_count,
                    },
                    "input_sha256": {
                        name: hashlib.sha256(
                            resolve_data_file(name).read_bytes()
                        ).hexdigest()
                        for name in (task.train_filename, task.prediction_filename)
                    },
                    "packages": {
                        name: version(name)
                        for name in (
                            "torch",
                            "transformers",
                            "peft",
                            "accelerate",
                            "bitsandbytes",
                            "scikit-learn",
                        )
                    },
                },
            )
            log(
                f"Best validation Macro-F1: {best_validation['macro_f1']:.5f}\n"
                f"Predictions: {csv_path}\nSubmission: {zip_path}\nAdapter: {adapter_dir}",
                runtime,
            )

        on_main(export_artifacts, runtime)
        del model, base, loader
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        cleanup_runtime(runtime)


def run_llm_cli(
    task: TaskConfig, defaults: LLMTrainingConfig = LLM_TRAINING
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--revision", default=defaults.model_revision)
    parser.add_argument(
        "--text-mode", choices=("paper", "raw"), default=defaults.text_mode
    )
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.train_batch_size_per_gpu
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument(
        "--prediction-file",
        default=task.prediction_filename,
        help="CSV input filename, or full path (e.g. official test inputs).",
    )
    args = parser.parse_args()
    config = replace(
        defaults,
        model_name=args.model,
        model_revision=args.revision,
        text_mode=args.text_mode,
        max_epochs=args.max_epochs,
        max_length=args.max_length,
        train_batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_all_available_gpus=defaults.use_all_available_gpus
        and not args.single_gpu,
    )
    validate_config(config)
    relaunch_with_torchrun_if_needed(config)
    run_llm_task(replace(task, prediction_filename=args.prediction_file), config)
