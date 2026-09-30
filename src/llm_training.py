"""Lightly normalized LoRA SFT, holdout selection, retraining, and export."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from llm_config import (
    CHECKPOINT_ROOT,
    LLM_TRAINING,
    OUTPUT_ROOT,
    LLMTrainingConfig,
    TaskConfig,
)
from src.data import load_epoch_validation_frame, load_task_frames, resolve_data_file
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
    cap_training_labels,
    create_train_loader,
    is_better_checkpoint,
    log,
    predefined_fold_indices,
    set_global_seed,
    stratified_holdout_indices,
    unwrap_model,
    write_downsampled_rows,
    write_seed_selection_results,
    write_split_manifest,
    write_validation_confusion_history,
    write_validation_confusion_matrix,
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
        "progress_log_percent",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be strictly between zero and one")
    if config.progress_log_percent > 100:
        raise ValueError("progress_log_percent must be between 1 and 100")
    if config.compute_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("compute_dtype must be auto, float16 or bfloat16")
    if config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if config.initial_loss_scale <= 0:
        raise ValueError("initial_loss_scale must be positive")
    if not 0 <= config.warmup_ratio < 1 or not 0 <= config.lora_dropout < 1:
        raise ValueError("warmup_ratio and lora_dropout must be in [0, 1)")


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
            "LoRA training requires CUDA. Select a GPU accelerator on Kaggle; "
            "CPU is only supported by the tests."
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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def completion_token_diagnostics(encoder: ChatEncoder) -> list[dict[str, object]]:
    """Describe the exact assistant completion scored for every allowed label."""

    tokenizer = encoder.tokenizer
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    diagnostics = []
    for label, code, token_ids in zip(
        encoder.task.labels, encoder.task.label_codes, encoder.completions, strict=True
    ):
        ids = list(token_ids)
        token_pieces = (
            list(convert_ids_to_tokens(ids))
            if callable(convert_ids_to_tokens)
            else None
        )
        diagnostics.append(
            {
                "label": label,
                "code": code,
                "token_count": len(ids),
                "token_ids": ids,
                "token_pieces": token_pieces,
                "decoded_completion": tokenizer.decode(
                    ids, skip_special_tokens=False
                ),
            }
        )
    return diagnostics


def log_completion_token_diagnostics(
    diagnostics: list[dict[str, object]], runtime: RuntimeContext
) -> None:
    """Log per-label tokenization and whether all completion lengths match."""

    for item in diagnostics:
        log(
            f"Completion tokens | label={item['label']!r} | code={item['code']!r} | "
            f"count={item['token_count']} | ids={item['token_ids']} | "
            f"pieces={item['token_pieces']} | "
            f"decoded={item['decoded_completion']!r}",
            runtime,
        )
    token_counts = [int(item["token_count"]) for item in diagnostics]
    log(
        "Completion token lengths equal across labels: "
        f"{len(set(token_counts)) == 1} | counts={token_counts}",
        runtime,
    )


class LineProgress:
    """Print compact, stable progress lines that render cleanly in Kaggle logs."""

    def __init__(
        self,
        total: int,
        description: str,
        unit: str,
        runtime: RuntimeContext,
        log_every_percent: int,
    ) -> None:
        self.total = total
        self.description = description
        self.unit = unit
        self.runtime = runtime
        self.log_every_percent = log_every_percent
        self.completed = 0
        self.next_percent = log_every_percent
        self.started_at = time.monotonic()
        log(f"[{description}] START | total={total} {unit}", runtime)

    def update(self, amount: int, details: str = "") -> None:
        self.completed = min(self.total, self.completed + amount)
        percent = 100 if self.total == 0 else int(100 * self.completed / self.total)
        if self.completed < self.total and percent < self.next_percent:
            return

        elapsed = time.monotonic() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = (
            (self.total - self.completed) / rate
            if rate > 0 and self.completed < self.total
            else 0.0
        )
        suffix = f" | {details}" if details else ""
        log(
            f"[{self.description}] {percent:3d}% | "
            f"{self.completed}/{self.total} {self.unit} | "
            f"elapsed={_format_duration(elapsed)} | "
            f"ETA={_format_duration(remaining)} | "
            f"speed={rate:.2f} {self.unit}/s{suffix}",
            self.runtime,
        )
        while self.next_percent <= percent:
            self.next_percent += self.log_every_percent


def load_lora_model(config: LLMTrainingConfig, runtime: RuntimeContext):
    """Load an unquantized backbone and attach rank-r LoRA adapters."""
    require_cuda(runtime)
    from peft import LoraConfig, get_peft_model

    dtype = compute_dtype(config, runtime)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=dtype,
        device_map={"": runtime.device.index},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        raise RuntimeError("Expected a non-quantized backbone for standard LoRA.")
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
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
    kwargs = {
        key: value
        for key, value in batch.items()
        if key not in {"labels", "class_ids"}
    }
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
                    log_probs, counts = completion_log_probs(model, batch)
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


@torch.no_grad()
def score_candidates(model, dataset, collator, config, runtime, description):
    """Score every full legal answer, restore row order and compute fixed-class F1.

    Mean token log probabilities include the chat end suffix and avoid favoring
    shorter answers when official labels have different token counts. Softmax
    produces probabilities conditional on this finite candidate set (not
    calibrated confidence). No sampling or free-text label parsing is used.
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
    progress = LineProgress(
        count,
        description,
        "candidates/GPU",
        runtime,
        config.progress_log_percent,
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
    expected_counts = np.asarray(
        [len(token_ids) for token_ids in dataset.encoder.completions],
        dtype=np.float64,
    )
    if not np.all(counts == expected_counts[None, :]):
        raise RuntimeError(
            "Completion token counts changed between prediction rows."
        )
    log(
        f"{description} | completion token counts="
        f"{dict(zip(dataset.encoder.task.labels, expected_counts.astype(int), strict=True))} "
        "| "
        f"equal_across_labels={len(set(expected_counts.tolist())) == 1}",
        runtime,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite candidate scores; check AMP and training stability."
        )
    normalized_scores = scores / counts
    probabilities = torch.softmax(
        torch.from_numpy(normalized_scores), dim=1
    ).numpy()
    result = {
        "probabilities": probabilities,
        "sequence_log_probs": scores,
        "mean_token_log_probs": normalized_scores,
        "completion_token_counts": counts,
    }
    if dataset.labels is not None:
        labels = np.asarray(dataset.labels, dtype=np.int64)
        predictions = probabilities.argmax(axis=1)
        per_class_scores = f1_score(
            labels,
            predictions,
            labels=list(range(n_labels)),
            average=None,
            zero_division=0,
        )
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
            per_class_f1={
                label: float(score)
                for label, score in zip(
                    dataset.encoder.task.labels, per_class_scores, strict=True
                )
            },
        )
        matrix = confusion_matrix(
            labels,
            predictions,
            labels=list(range(n_labels)),
        )
        result["confusion_matrix"] = {
            "rows": "actual labels",
            "columns": "predicted labels",
            "label_order": list(dataset.encoder.task.labels),
            "values": matrix.astype(int).tolist(),
        }
        if dataset.encoder.task.report_per_class_metrics:
            precision, recall, per_class_f1, support = (
                precision_recall_fscore_support(
                    labels,
                    predictions,
                    labels=list(range(n_labels)),
                    zero_division=0,
                )
            )
            result["per_class"] = {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(per_class_f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(dataset.encoder.task.labels)
            }
    return result


def format_per_class_diagnostics(metrics: dict) -> list[str]:
    """Describe each class and both directions of its classification errors."""

    per_class = metrics.get("per_class")
    confusion = metrics.get("confusion_matrix")
    if not per_class or not confusion:
        return []
    labels = confusion["label_order"]
    matrix = np.asarray(confusion["values"], dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix shape does not match its label order.")

    lines = []
    for class_index, label in enumerate(labels):
        class_metrics = per_class[label]
        incoming = [
            f"{source_label}={int(matrix[source_index, class_index])}"
            for source_index, source_label in enumerate(labels)
            if source_index != class_index
        ]
        outgoing = [
            f"{predicted_label}={int(matrix[class_index, predicted_index])}"
            for predicted_index, predicted_label in enumerate(labels)
            if predicted_index != class_index
        ]
        incoming_total = int(
            matrix[:, class_index].sum() - matrix[class_index, class_index]
        )
        outgoing_total = int(
            matrix[class_index, :].sum() - matrix[class_index, class_index]
        )
        lines.extend(
            [
                (
                    f"  {label}: precision={class_metrics['precision']:.5f}, "
                    f"recall={class_metrics['recall']:.5f}, "
                    f"f1={class_metrics['f1']:.5f}, "
                    f"support={class_metrics['support']}"
                ),
                (
                    f"    False positives into {label} (actual label): "
                    f"{', '.join(incoming)} | total={incoming_total}"
                ),
                (
                    f"    False negatives from {label} (predicted label): "
                    f"{', '.join(outgoing)} | total={outgoing_total}"
                ),
            ]
        )
    return lines


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


def _wrap_for_ddp(model, runtime: RuntimeContext):
    if not runtime.distributed:
        return model
    return DDP(
        model,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


def _training_state(model, loader, epochs, config, runtime):
    optimizer = build_optimizer(model, config)
    total_updates = math.ceil(
        len(loader) / config.gradient_accumulation_steps
    ) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_updates * config.warmup_ratio),
        total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=compute_dtype(config, runtime) == torch.float16,
        init_scale=config.initial_loss_scale,
    )
    return optimizer, scheduler, scaler


def binary_majority_vote(
    seed_label_ids: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final binary labels and per-class votes for an odd seed ensemble."""

    values = np.asarray(seed_label_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("seed_label_ids must have shape (seeds, prediction rows)")
    if values.shape[0] % 2 == 0:
        raise ValueError("Binary majority voting requires an odd number of seeds")
    if set(labels) != {"Non-Hate", "Hate"} or len(labels) != 2:
        raise ValueError("Binary majority voting requires Non-Hate and Hate labels")
    if np.any((values < 0) | (values >= len(labels))):
        raise ValueError("seed_label_ids contains an invalid label index")

    hate_index = labels.index("Hate")
    nonhate_index = labels.index("Non-Hate")
    hate_votes = np.sum(values == hate_index, axis=0)
    nonhate_votes = np.sum(values == nonhate_index, axis=0)
    final_ids = np.where(hate_votes > nonhate_votes, hate_index, nonhate_index)
    return final_ids, hate_votes, nonhate_votes


def _best_validation_result(seed_results: list[dict]) -> dict:
    """Return the best holdout result for reporting, not final prediction."""

    best = seed_results[0]
    for candidate in seed_results[1:]:
        candidate_summary = candidate["summary"]
        best_summary = best["summary"]
        if is_better_checkpoint(
            candidate_summary["macro_f1"],
            candidate_summary["loss"],
            best_summary["macro_f1"],
            best_summary["loss"],
        ):
            best = candidate
    return best


def _run_seed_ensemble(
    *,
    task: TaskConfig,
    config: LLMTrainingConfig,
    runtime: RuntimeContext,
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    val_data: CompletionDataset,
    prediction_data: CompletionDataset,
    collator: CompletionCollator,
    encoder: ChatEncoder,
    tokenizer: object,
    seed_results: list[dict],
    selection_seeds: tuple[int, ...],
    output_dir: Path,
    checkpoint_dir: Path,
    split_strategy: str,
    resolved_revision: str | None,
    parameter_counts: dict[str, int] | None,
    token_diagnostics: list[dict[str, object]],
) -> None:
    """Retrain every selected seed, predict, and hard-vote Task A outputs."""

    if len(seed_results) != len(selection_seeds):
        raise RuntimeError("Every configured seed must finish before ensembling")
    if len(selection_seeds) % 2 == 0:
        raise ValueError("ensemble_all_seeds requires an odd number of seeds")

    reporting_best = _best_validation_result(seed_results)
    reporting_summary = reporting_best["summary"]
    reporting_seed = int(reporting_summary["seed"])
    scalar_results = [
        {
            "seed": result["summary"]["seed"],
            "best_epoch": result["summary"]["best_epoch"],
            "macro_f1": result["summary"]["macro_f1"],
            "loss": result["summary"]["loss"],
            "accuracy": result["summary"]["accuracy"],
            "selection_train_rows": result["summary"]["selection_train_rows"],
            "full_retrain_rows": result["summary"]["full_retrain_rows"],
            "downsampled_rows": result["summary"]["downsampled_rows"],
            "checkpoint": result["summary"]["checkpoint"],
            "selected": True,
        }
        for result in seed_results
    ]

    def export_selection_overview() -> None:
        write_split_manifest(
            frame,
            reporting_best["train_indices"],
            reporting_best["validation_indices"],
            output_dir,
            reporting_best["downsampled_indices"],
        )
        write_downsampled_rows(
            frame, task, reporting_best["downsampled_indices"], output_dir
        )
        write_validation_predictions(
            val_frame,
            task,
            reporting_best["validation_probabilities"],
            output_dir,
        )
        _write_json(
            output_dir / "selection_training_history.json",
            {
                "seed": reporting_seed,
                "purpose": "best holdout seed for reporting only; all seeds are retrained",
                "epochs": reporting_best["selection_history"],
            },
        )
        for record in reporting_best["selection_history"]:
            if "confusion_matrix" in record:
                write_validation_confusion_matrix(
                    record["confusion_matrix"], record["epoch"], output_dir
                )
        write_validation_confusion_history(
            reporting_best["selection_history"], output_dir
        )
        write_seed_selection_results(scalar_results, output_dir)
        macro_scores = [row["macro_f1"] for row in scalar_results]
        _write_json(
            output_dir / "seed_selection_summary.json",
            {
                "selection_rule": (
                    "best epoch per seed by highest macro_f1; lowest loss breaks "
                    "exact ties; every seed proceeds to full-data retraining"
                ),
                "validation_is_fixed_across_seeds": True,
                "split_seed": config.seed,
                "reporting_best_seed": reporting_seed,
                "reporting_best_seed_is_not_an_ensemble_winner": True,
                "all_seeds_retrained": True,
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_std": float(np.std(macro_scores)),
                "results": [result["summary"] for result in seed_results],
            },
        )

    on_main(export_selection_overview, runtime)
    log(
        "Seed holdout results: "
        + " | ".join(
            f"seed={result['summary']['seed']} "
            f"Macro-F1={result['summary']['macro_f1']:.5f} "
            f"best_epoch={result['summary']['best_epoch']}"
            for result in seed_results
        ),
        runtime,
    )

    retrained_results = []
    for seed_number, result in enumerate(seed_results, start=1):
        summary = result["summary"]
        seed = int(summary["seed"])
        best_epoch = int(summary["best_epoch"])
        full_train_indices = result["full_train_indices"]
        full_train_frame = frame.iloc[full_train_indices]
        full_train_data = CompletionDataset(
            full_train_frame["text"].tolist(),
            encoder,
            full_train_frame["label_id"].tolist(),
        )
        seed_output_dir = result["seed_output_dir"]
        seed_checkpoint_dir = result["seed_checkpoint_dir"]
        final_adapter_dir = seed_checkpoint_dir / "final_full_data_adapter"

        log(
            f"\n=== PHASE 2/2 | SEED {seed_number}/{len(seed_results)} "
            f"| seed={seed} | FULL-DATA RETRAIN "
            f"({len(full_train_data)} rows / {best_epoch} epoch(s)) ===",
            runtime,
        )
        set_global_seed(seed)
        full_loader, full_sampler = create_train_loader(
            full_train_data, collator, config, runtime, seed
        )
        final_model = _wrap_for_ddp(load_lora_model(config, runtime), runtime)
        optimizer, scheduler, scaler = _training_state(
            final_model, full_loader, best_epoch, config, runtime
        )
        full_history = []
        for epoch in range(1, best_epoch + 1):
            if full_sampler is not None:
                full_sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                final_model,
                full_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                runtime,
                epoch,
                f"Full-data retrain seed {seed}",
            )
            full_history.append({"epoch": epoch, "train_loss": train_loss})
            log(
                f"Seed {seed} full-data epoch {epoch}/{best_epoch}: "
                f"train_loss={train_loss:.5f}",
                runtime,
            )
            on_main(
                lambda seed_output_dir=seed_output_dir,
                seed=seed,
                full_history=list(full_history): _write_json(
                    seed_output_dir / "full_data_training_history.json",
                    {"seed": seed, "epochs": full_history},
                ),
                runtime,
            )

        del optimizer, scheduler, scaler
        on_main(
            partial(save_adapter, final_model, tokenizer, final_adapter_dir), runtime
        )
        restore_adapter(final_model, final_adapter_dir)
        predictions = score_candidates(
            final_model,
            prediction_data,
            collator,
            config,
            runtime,
            "Development prediction from full-data adapter",
        )
        label_ids = predictions["probabilities"].argmax(axis=1)
        predicted_labels = [task.labels[index] for index in label_ids]
        final_summary = {
            "seed": seed,
            "epochs": best_epoch,
            "training_rows": len(full_train_data),
            "loaded_rows": len(frame),
            "downsampled_rows_excluded": len(result["downsampled_indices"]),
            "label_distribution": full_train_frame[task.label_column]
            .value_counts()
            .to_dict(),
            "checkpoint": str(final_adapter_dir),
            "prediction_source": "seed_specific_final_full_data_adapter",
        }

        def export_seed_retrain_artifacts(
            seed_output_dir=seed_output_dir,
            predicted_labels=predicted_labels,
            summary=summary,
            final_summary=final_summary,
        ) -> None:
            write_submission(
                prediction_frame["id"].tolist(),
                predicted_labels,
                task.labels,
                seed_output_dir,
                task.submission_zip_name,
            )
            _write_json(
                seed_output_dir / "seed_summary.json",
                {**summary, "full_data_retrain": final_summary},
            )

        on_main(export_seed_retrain_artifacts, runtime)
        retrained_results.append(
            {
                "seed": seed,
                "label_ids": label_ids,
                "probabilities": predictions["probabilities"],
                "summary": final_summary,
                "full_train_truncated": full_train_data.truncated_count,
            }
        )
        del final_model, full_loader, full_train_data
        gc.collect()
        torch.cuda.empty_cache()

    if len(retrained_results) != len(selection_seeds):
        raise RuntimeError("Voting requires a completed prediction from every seed")
    stacked_label_ids = np.stack(
        [result["label_ids"] for result in retrained_results], axis=0
    )
    final_ids, hate_votes, nonhate_votes = binary_majority_vote(
        stacked_label_ids, task.labels
    )
    final_labels = [task.labels[index] for index in final_ids]
    mean_probabilities = np.mean(
        np.stack(
            [result["probabilities"] for result in retrained_results], axis=0
        ),
        axis=0,
    )

    voting_data: dict[str, object] = {
        "id": prediction_frame["id"].astype(str).tolist()
    }
    for row, seed_result in zip(
        stacked_label_ids, retrained_results, strict=True
    ):
        voting_data[f"seed_{seed_result['seed']}"] = [
            task.labels[index] for index in row
        ]
    voting_data["hate_votes"] = hate_votes
    voting_data["nonhate_votes"] = nonhate_votes
    for label_index, label in enumerate(task.labels):
        safe_label = label.lower().replace("-", "_").replace(" ", "_")
        voting_data[f"mean_prob_{safe_label}"] = mean_probabilities[:, label_index]
    voting_data["final_label"] = final_labels
    voting_frame = pd.DataFrame(voting_data)

    def export_ensemble_artifacts() -> None:
        voting_frame.to_csv(
            output_dir / "voting_predictions.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        csv_path, zip_path = write_submission(
            prediction_frame["id"].tolist(),
            final_labels,
            task.labels,
            output_dir,
            task.submission_zip_name,
        )
        ensemble_summary = {
            "method": "hard majority vote",
            "seeds": list(selection_seeds),
            "required_votes": len(selection_seeds) // 2 + 1,
            "tie_possible": False,
            "vote_rule": "Hate if hate_votes > nonhate_votes; otherwise Non-Hate",
            "prediction_file": str(csv_path),
            "voting_audit_file": str(output_dir / "voting_predictions.csv"),
            "submission": str(zip_path),
        }
        retrain_summaries = [result["summary"] for result in retrained_results]
        _write_json(
            checkpoint_dir / "metrics.json",
            {
                "selection": [result["summary"] for result in seed_results],
                "reporting_best_selection": reporting_summary,
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
            },
        )
        reporting_train_frame = frame.iloc[reporting_best["train_indices"]]
        _write_json(
            output_dir / "run_summary.json",
            {
                "task": task.task_name,
                "method": "LoRA answer-only causal LM with hard-vote seed ensemble",
                "model": config.model_name,
                "resolved_model_revision": resolved_revision,
                "world_size": runtime.world_size,
                "compute_dtype": str(compute_dtype(config, runtime)),
                "quantization": None,
                "text_input": (
                    "ftfy + Unicode NFKC + HTML entity decoding + whitespace "
                    "normalization; URLs, mentions, and HTML tags preserved"
                ),
                "effective_train_batch_size": config.train_batch_size_per_gpu
                * config.gradient_accumulation_steps
                * runtime.world_size,
                "selection_seeds": list(selection_seeds),
                "reporting_best_seed": reporting_seed,
                "reporting_best_seed_used_only_for_validation_summary": True,
                "selection_split": {
                    "strategy": split_strategy,
                    "split_seed": config.seed,
                    "validation_fold": task.selection_validation_fold,
                    "validation_data_origins": list(
                        task.selection_validation_data_origins
                    ),
                    "actual_validation_fraction": len(val_data) / len(frame),
                    "validation_rows": len(val_data),
                    "validation_label_distribution": val_frame[task.label_column]
                    .value_counts()
                    .to_dict(),
                    "reporting_seed_train_rows": len(
                        reporting_best["train_indices"]
                    ),
                    "reporting_seed_train_label_distribution": reporting_train_frame[
                        task.label_column
                    ]
                    .value_counts()
                    .to_dict(),
                    "per_seed_downsampled_rows": {
                        str(result["summary"]["seed"]): len(
                            result["downsampled_indices"]
                        )
                        for result in seed_results
                    },
                },
                "seed_selection": [
                    result["summary"] for result in seed_results
                ],
                "best_validation_checkpoint_for_reporting": reporting_summary,
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
                "development_rows": len(prediction_data),
                "training_config": asdict(config),
                "parameters": parameter_counts,
                "system_prompt": encoder.system_text,
                "label_order": list(task.labels),
                "completion_token_ids": encoder.completions,
                "completion_token_diagnostics": token_diagnostics,
                "completion_token_lengths_equal": len(
                    {int(item["token_count"]) for item in token_diagnostics}
                )
                == 1,
                "inference": (
                    "per-seed argmax mean token log P(answer including chat end | "
                    "prompt), followed by hard majority vote"
                ),
                "probabilities": (
                    "softmax over mean token log likelihoods; mean probabilities "
                    "are diagnostic only and do not determine the final vote"
                ),
                "loss": "mean per-example answer-token NLL; not class cross-entropy",
                "truncated_comments": {
                    "selection_train_by_seed": {
                        str(result["summary"]["seed"]): result[
                            "selection_train_truncated"
                        ]
                        for result in seed_results
                    },
                    "validation": val_data.truncated_count,
                    "full_train_by_seed": {
                        str(result["seed"]): result["full_train_truncated"]
                        for result in retrained_results
                    },
                    "development": prediction_data.truncated_count,
                },
                "input_sha256": {
                    name: hashlib.sha256(resolve_data_file(name).read_bytes()).hexdigest()
                    for name in (*task.labeled_filenames(), task.prediction_filename)
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "peft",
                        "accelerate",
                        "scikit-learn",
                    )
                },
            },
        )
        log(
            f"\n=== COMPLETE | {task.task_name} ===\n"
            f"Retrained seeds: {list(selection_seeds)}\n"
            f"Voting rule: at least {len(selection_seeds) // 2 + 1} votes.\n"
            f"Voting audit: {output_dir / 'voting_predictions.csv'}\n"
            f"Predictions: {csv_path}\nSubmission: {zip_path}",
            runtime,
        )

    on_main(export_ensemble_artifacts, runtime)


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
    """Report external validation every epoch without affecting epoch selection."""

    def __init__(self, task, config, runtime, training_frame, encoder, collator):
        self.task = task
        self.config = config
        self.runtime = runtime
        self.collator = collator
        self.frame = load_epoch_validation_frame(task)
        overlap = set(training_frame["id"]) & set(self.frame["id"])
        if overlap:
            raise ValueError(
                "Official validation IDs must not occur in training data: "
                f"{sorted(overlap)[:10]}"
            )
        self.data = CompletionDataset(
            self.frame["text"].tolist(), encoder, self.frame["label_id"].tolist()
        )
        path = resolve_data_file(task.epoch_validation_filename)
        self.summary = {
            "filename": task.epoch_validation_filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(self.frame),
            "label_distribution": self.frame[task.label_column].value_counts().to_dict(),
            "truncated_comments": self.data.truncated_count,
            "purpose": "reporting only; never used for training or epoch selection",
        }
        log(
            f"Task B per-epoch validation | file={task.epoch_validation_filename} "
            f"| rows={len(self.data)} | reporting only",
            runtime,
        )

    def evaluate(self, model, seed: int, epoch: int, output_dir: Path) -> dict:
        heading = f"Official validation | seed={seed} | epoch={epoch}"
        devices = (
            [self.runtime.device.index] if self.runtime.device.type == "cuda" else []
        )
        was_training = model.training
        # Diagnostic forwards must not advance the RNG used by subsequent training.
        try:
            with torch.random.fork_rng(devices=devices):
                metrics = score_candidates(
                    model, self.data, self.collator, self.config, self.runtime, heading
                )
        finally:
            model.train(was_training)
        log_classification_metrics(metrics, heading, self.runtime)

        def export() -> None:
            directory = output_dir / "official_validation"
            directory.mkdir(parents=True, exist_ok=True)
            write_validation_confusion_matrix(metrics["confusion_matrix"], epoch, directory)
            _write_prediction_diagnostics(
                self.frame, self.task, metrics["probabilities"],
                directory / f"predictions_epoch_{epoch:02d}.csv",
            )
            _write_json(
                directory / f"metrics_epoch_{epoch:02d}.json",
                {"seed": seed, "epoch": epoch, **public_validation_metrics(metrics)},
            )

        on_main(export, self.runtime)
        return metrics

    def evaluate_ensemble(self, results: list[dict], output_dir: Path) -> dict:
        probabilities = np.mean(
            np.stack([row["validation_probabilities"] for row in results]), axis=0
        )
        # Ensemble loss here is mean gold-answer NLL of the component models.
        metrics = _classification_metrics(
            self.task, self.frame["label_id"].to_numpy(), probabilities,
            float(np.mean([row["validation_loss"] for row in results])),
        )
        log_classification_metrics(metrics, "Official validation | FINAL SOFT ENSEMBLE", self.runtime)

        def export() -> None:
            _write_json(
                output_dir / "official_validation_metrics.json",
                {"dataset": self.summary, **public_validation_metrics(metrics)},
            )
            _write_prediction_diagnostics(
                self.frame, self.task, probabilities,
                output_dir / "official_validation_predictions.csv",
            )

        on_main(export, self.runtime)
        return public_validation_metrics(metrics)


def _run_fixed_epoch_full_retrain_ensemble(
    *,
    task: TaskConfig,
    config: LLMTrainingConfig,
    runtime: RuntimeContext,
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    prediction_data: CompletionDataset,
    collator: CompletionCollator,
    encoder: ChatEncoder,
    tokenizer: object,
    output_dir: Path,
    checkpoint_dir: Path,
    retrain_seeds: tuple[int, ...],
    token_diagnostics: list[dict[str, object]],
    fixed_epochs: int,
) -> None:
    """Skip epoch selection and soft-ensemble fixed-epoch full-data models."""

    if fixed_epochs < 1:
        raise ValueError("fixed_full_retrain_epochs must be positive.")
    if not task.full_retrain_soft_ensemble:
        raise ValueError("Fixed full-data training requires soft ensembling.")
    if len(retrain_seeds) < 2:
        raise ValueError("A full-data soft ensemble requires at least two seeds.")
    if task.selection_train_label_caps:
        raise ValueError("Fixed class-weighted training cannot downsample labels.")
    if task.class_weight_power is None:
        raise ValueError("Fixed Task B training requires class-weighted loss.")

    validation_monitor = TaskBValidationMonitor(
        task, config, runtime, frame, encoder, collator
    )

    full_train_data = CompletionDataset(
        frame["text"].tolist(), encoder, frame["label_id"].tolist()
    )
    full_class_weights, full_class_weight_summary = _class_weight_state(
        task, frame["label_id"].to_numpy()
    )
    log(
        f"\n=== FIXED-EPOCH FULL-DATA EXPERIMENT | OOF SKIPPED "
        f"| epochs={fixed_epochs} | models={len(retrain_seeds)} ===",
        runtime,
    )
    log(
        "Full-data class weights: "
        + " | ".join(
            f"{label}={full_class_weight_summary['weights'][label]:.4f}"
            for label in task.labels
        ),
        runtime,
    )

    retrained_results = []
    resolved_revision = None
    parameter_counts = None
    for seed_number, seed in enumerate(retrain_seeds, start=1):
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_checkpoint_dir = checkpoint_dir / f"seed_{seed}"
        final_adapter_dir = seed_checkpoint_dir / "final_full_data_adapter"
        on_main(
            partial(seed_output_dir.mkdir, parents=True, exist_ok=True), runtime
        )
        on_main(
            partial(seed_checkpoint_dir.mkdir, parents=True, exist_ok=True), runtime
        )
        log(
            f"\n=== FULL-DATA MODEL {seed_number}/{len(retrain_seeds)} "
            f"| seed={seed} | rows={len(full_train_data)} "
            f"| epochs={fixed_epochs} ===",
            runtime,
        )
        set_global_seed(seed)
        full_loader, full_sampler = create_train_loader(
            full_train_data, collator, config, runtime, seed
        )
        loaded_model = load_lora_model(config, runtime)
        if parameter_counts is None:
            base = loaded_model.get_base_model()
            resolved_revision = getattr(base.config, "_commit_hash", None)
            parameter_counts = {
                "trainable": sum(
                    parameter.numel()
                    for parameter in loaded_model.parameters()
                    if parameter.requires_grad
                ),
                "reported_total": loaded_model.num_parameters(),
            }
            del base
        model = _wrap_for_ddp(loaded_model, runtime)
        optimizer, scheduler, scaler = _training_state(
            model, full_loader, fixed_epochs, config, runtime
        )
        full_history = []
        for epoch in range(1, fixed_epochs + 1):
            if full_sampler is not None:
                full_sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                model,
                full_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                runtime,
                epoch,
                f"Fixed full-data train seed {seed}",
                class_weights=full_class_weights,
            )
            validation_metrics = validation_monitor.evaluate(
                model, seed, epoch, seed_output_dir
            )
            full_history.append(
                {
                    "epoch": epoch, "train_weighted_loss": train_loss,
                    "official_validation": public_validation_metrics(validation_metrics),
                }
            )
            log(
                f"Seed {seed} full-data epoch {epoch}/{fixed_epochs}: "
                f"train_weighted_loss={train_loss:.5f}",
                runtime,
            )
            on_main(
                lambda seed_output_dir=seed_output_dir,
                seed=seed,
                full_history=list(full_history): _write_json(
                    seed_output_dir / "full_data_training_history.json",
                    {
                        "seed": seed,
                        "epoch_selection": "fixed_without_oof",
                        "validation_dataset": validation_monitor.summary,
                        "class_weighting": full_class_weight_summary,
                        "epochs": full_history,
                    },
                ),
                runtime,
            )

        del optimizer, scheduler, scaler
        on_main(partial(save_adapter, model, tokenizer, final_adapter_dir), runtime)
        restore_adapter(model, final_adapter_dir)
        predictions = score_candidates(
            model,
            prediction_data,
            collator,
            config,
            runtime,
            f"Development prediction seed {seed}",
        )
        probabilities = predictions["probabilities"]
        label_ids = probabilities.argmax(axis=1)
        summary = {
            "seed": seed,
            "epochs": fixed_epochs,
            "training_rows": len(full_train_data),
            "checkpoint": str(final_adapter_dir),
            "prediction_source": "seed_specific_full_data_adapter",
        }

        def export_seed_predictions(
            seed_output_dir=seed_output_dir,
            label_ids=label_ids,
            summary=summary,
        ) -> None:
            write_submission(
                prediction_frame["id"].tolist(),
                [task.labels[index] for index in label_ids],
                task.labels,
                seed_output_dir,
                task.submission_zip_name,
            )
            _write_json(seed_output_dir / "seed_summary.json", summary)

        on_main(export_seed_predictions, runtime)
        retrained_results.append(
            {
                "seed": seed,
                "probabilities": probabilities,
                "label_ids": label_ids,
                "summary": summary,
                "validation_probabilities": validation_metrics["probabilities"],
                "validation_loss": validation_metrics["loss"],
            }
        )
        del model, full_loader
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_validation = validation_monitor.evaluate_ensemble(retrained_results, output_dir)
    stacked_probabilities = np.stack(
        [result["probabilities"] for result in retrained_results], axis=0
    )
    mean_probabilities = stacked_probabilities.mean(axis=0)
    final_ids = mean_probabilities.argmax(axis=1)
    final_labels = [task.labels[index] for index in final_ids]

    def export_final_ensemble() -> None:
        ensemble_data: dict[str, object] = {
            "id": prediction_frame["id"].astype(str).tolist()
        }
        for result in retrained_results:
            ensemble_data[f"seed_{result['seed']}"] = [
                task.labels[index] for index in result["label_ids"]
            ]
        for label_index, label in enumerate(task.labels):
            safe_label = label.lower().replace("-", "_").replace(" ", "_")
            ensemble_data[f"mean_prob_{safe_label}"] = mean_probabilities[
                :, label_index
            ]
            ensemble_data[f"votes_{safe_label}"] = np.sum(
                np.stack(
                    [result["label_ids"] for result in retrained_results], axis=0
                )
                == label_index,
                axis=0,
            )
        ensemble_data["final_label"] = final_labels
        pd.DataFrame(ensemble_data).to_csv(
            output_dir / "ensemble_predictions.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        csv_path, zip_path = write_submission(
            prediction_frame["id"].tolist(),
            final_labels,
            task.labels,
            output_dir,
            task.submission_zip_name,
        )
        ensemble_summary = {
            "method": "equal-weight mean of per-seed candidate probabilities",
            "seeds": list(retrain_seeds),
            "models": len(retrained_results),
            "prediction_file": str(csv_path),
            "audit_file": str(output_dir / "ensemble_predictions.csv"),
            "submission": str(zip_path),
        }
        epoch_selection = {
            "strategy": "fixed_without_oof",
            "oof_skipped": True,
            "fixed_epoch": fixed_epochs,
        }
        retrain_summaries = [result["summary"] for result in retrained_results]
        _write_json(
            checkpoint_dir / "metrics.json",
            {
                "epoch_selection": epoch_selection,
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
            },
        )
        _write_json(
            output_dir / "run_summary.json",
            {
                "task": task.task_name,
                "method": (
                    "LoRA answer-only causal LM with fixed "
                    f"{fixed_epochs}-epoch full-data soft seed ensemble"
                ),
                "model": config.model_name,
                "resolved_model_revision": resolved_revision,
                "world_size": runtime.world_size,
                "compute_dtype": str(compute_dtype(config, runtime)),
                "quantization": None,
                "effective_train_batch_size": config.train_batch_size_per_gpu
                * config.gradient_accumulation_steps
                * runtime.world_size,
                "epoch_selection": epoch_selection,
                "official_validation": {
                    "dataset": validation_monitor.summary,
                    "ensemble_metrics": ensemble_validation,
                },
                "class_weighting": {
                    "downsampling": None,
                    "full_data": full_class_weight_summary,
                },
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
                "development_rows": len(prediction_data),
                "training_config": asdict(config),
                "parameters": parameter_counts,
                "system_prompt": encoder.system_text,
                "label_order": list(task.labels),
                "completion_token_ids": encoder.completions,
                "completion_token_diagnostics": token_diagnostics,
                "inference": (
                    "mean token log likelihood per allowed label, softmax per "
                    "model, then equal-weight mean probabilities across seeds"
                ),
                "truncated_comments": {
                    "full_train": full_train_data.truncated_count,
                    "development": prediction_data.truncated_count,
                },
                "input_sha256": {
                    name: hashlib.sha256(resolve_data_file(name).read_bytes()).hexdigest()
                    for name in (*task.labeled_filenames(), task.prediction_filename)
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "peft",
                        "accelerate",
                        "scikit-learn",
                    )
                },
            },
        )
        log(
            f"\n=== COMPLETE | {task.task_name} ===\n"
            f"OOF selection skipped; fixed full-data epochs: {fixed_epochs}\n"
            f"Full-data seeds: {list(retrain_seeds)}\n"
            f"Ensemble: equal-weight soft probability mean\n"
            f"Predictions: {csv_path}\nSubmission: {zip_path}",
            runtime,
        )

    on_main(export_final_ensemble, runtime)


def _run_oof_full_retrain_ensemble(
    *,
    task: TaskConfig,
    config: LLMTrainingConfig,
    runtime: RuntimeContext,
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    prediction_data: CompletionDataset,
    collator: CompletionCollator,
    encoder: ChatEncoder,
    tokenizer: object,
    output_dir: Path,
    checkpoint_dir: Path,
    retrain_seeds: tuple[int, ...],
    token_diagnostics: list[dict[str, object]],
) -> None:
    """Select one epoch with five-fold OOF, then soft-ensemble full-data seeds."""

    folds = task.oof_validation_folds
    configured_folds = tuple(range(1, len(task.original_fold_filenames) + 1))
    if tuple(sorted(folds)) != configured_folds or len(set(folds)) != len(folds):
        raise ValueError("OOF validation folds must contain every configured fold once.")
    if not task.full_retrain_soft_ensemble:
        raise ValueError("OOF Task B requires full_retrain_soft_ensemble=True.")
    if len(retrain_seeds) < 2:
        raise ValueError("A full-data soft ensemble requires at least two seeds.")
    if task.selection_train_label_caps:
        raise ValueError("OOF class-weighted training cannot also downsample labels.")
    if task.class_weight_power is None:
        raise ValueError("OOF Task B requires class-weighted loss.")

    n_labels = len(task.labels)
    fold_epoch_outputs: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    resolved_revision = None
    parameter_counts = None

    log(
        f"\n=== PHASE 1/2 | {len(folds)}-FOLD OOF EPOCH SELECTION "
        f"| fixed selection seed={config.seed} ===",
        runtime,
    )
    for fold_number, validation_fold in enumerate(folds, start=1):
        train_indices, validation_indices = predefined_fold_indices(
            frame["fold"].to_numpy(),
            validation_fold,
            frame["data_origin"].to_numpy(),
            task.selection_validation_data_origins,
        )
        train_frame = frame.iloc[train_indices]
        validation_frame = frame.iloc[validation_indices]
        train_data = CompletionDataset(
            train_frame["text"].tolist(),
            encoder,
            train_frame["label_id"].tolist(),
        )
        validation_data = CompletionDataset(
            validation_frame["text"].tolist(),
            encoder,
            validation_frame["label_id"].tolist(),
        )
        class_weights, class_weight_summary = _class_weight_state(
            task, train_frame["label_id"].to_numpy()
        )
        if class_weight_summary is not None:
            log(
                f"OOF fold {validation_fold} class weights: "
                + " | ".join(
                    f"{label}={class_weight_summary['weights'][label]:.4f}"
                    for label in task.labels
                ),
                runtime,
            )

        set_global_seed(config.seed)
        train_loader, train_sampler = create_train_loader(
            train_data, collator, config, runtime, config.seed
        )
        model = load_lora_model(config, runtime)
        if parameter_counts is None:
            base = model.get_base_model()
            resolved_revision = getattr(base.config, "_commit_hash", None)
            parameter_counts = {
                "trainable": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "reported_total": model.num_parameters(),
            }
            del base
        model = _wrap_for_ddp(model, runtime)
        optimizer, scheduler, scaler = _training_state(
            model, train_loader, config.max_epochs, config, runtime
        )
        fold_history = []
        epoch_outputs = []
        log(
            f"\n--- OOF fold {fold_number}/{len(folds)} | validation fold="
            f"{validation_fold} | train={len(train_data)} | "
            f"validation={len(validation_data)} ---",
            runtime,
        )
        for epoch in range(1, config.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                runtime,
                epoch,
                f"OOF fold {validation_fold}",
                class_weights=class_weights,
            )
            metrics = score_candidates(
                model,
                validation_data,
                collator,
                config,
                runtime,
                f"OOF fold {validation_fold} epoch {epoch} validation",
            )
            public_record = {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                **{
                    key: metrics[key]
                    for key in (
                        "loss",
                        "macro_f1",
                        "accuracy",
                        "per_class_f1",
                        "confusion_matrix",
                    )
                },
            }
            if "per_class" in metrics:
                public_record["per_class"] = metrics["per_class"]
            fold_history.append(public_record)
            epoch_outputs.append(
                {
                    "epoch": epoch,
                    "validation_indices": validation_indices,
                    "probabilities": metrics["probabilities"],
                    "loss": metrics["loss"],
                }
            )
            log(
                f"OOF fold {validation_fold} epoch {epoch}: "
                f"train_weighted_loss={train_loss:.5f}, "
                f"val_loss={metrics['loss']:.5f}, "
                f"macro_f1={metrics['macro_f1']:.5f}, "
                f"accuracy={metrics['accuracy']:.5f}",
                runtime,
            )

        fold_output_dir = output_dir / f"oof_fold_{validation_fold}"

        def export_fold_history(
            fold_output_dir=fold_output_dir,
            validation_fold=validation_fold,
            fold_history=fold_history,
            class_weight_summary=class_weight_summary,
        ) -> None:
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                fold_output_dir / "selection_training_history.json",
                {
                    "validation_fold": validation_fold,
                    "selection_seed": config.seed,
                    "class_weighting": class_weight_summary,
                    "epochs": fold_history,
                },
            )

        on_main(export_fold_history, runtime)
        fold_epoch_outputs.extend(epoch_outputs)
        fold_summaries.append(
            {
                "validation_fold": validation_fold,
                "selection_seed": config.seed,
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
                "train_label_distribution": train_frame[task.label_column]
                .value_counts()
                .to_dict(),
                "validation_label_distribution": validation_frame[task.label_column]
                .value_counts()
                .to_dict(),
                "class_weighting": class_weight_summary,
                "truncated_train_comments": train_data.truncated_count,
                "truncated_validation_comments": validation_data.truncated_count,
            }
        )
        del model, optimizer, scheduler, scaler, train_loader, train_data, validation_data
        gc.collect()
        torch.cuda.empty_cache()

    original_indices = np.flatnonzero(
        frame["data_origin"].to_numpy() == "original"
    ).astype(np.int64)
    original_labels = frame.iloc[original_indices]["label_id"].to_numpy()
    oof_history = []
    oof_probabilities_by_epoch: dict[int, np.ndarray] = {}
    best_epoch, best_metric, best_loss = 0, -math.inf, math.inf
    for epoch in range(1, config.max_epochs + 1):
        probabilities_by_row = np.full((len(frame), n_labels), np.nan, dtype=np.float64)
        seen = np.zeros(len(frame), dtype=bool)
        loss_sum = 0.0
        validation_rows = 0
        for output in fold_epoch_outputs:
            if output["epoch"] != epoch:
                continue
            indices = output["validation_indices"]
            if seen[indices].any():
                raise RuntimeError("An original row received duplicate OOF predictions.")
            probabilities_by_row[indices] = output["probabilities"]
            seen[indices] = True
            loss_sum += float(output["loss"]) * len(indices)
            validation_rows += len(indices)
        if validation_rows != len(original_indices) or not seen[original_indices].all():
            raise RuntimeError("OOF predictions do not cover every original row exactly once.")
        probabilities = probabilities_by_row[original_indices]
        metrics = _classification_metrics(
            task,
            original_labels,
            probabilities,
            loss_sum / validation_rows,
        )
        record = {"epoch": epoch, **metrics}
        oof_history.append(record)
        oof_probabilities_by_epoch[epoch] = probabilities
        log(
            f"OOF aggregate epoch {epoch}: loss={metrics['loss']:.5f}, "
            f"macro_f1={metrics['macro_f1']:.5f}, "
            f"accuracy={metrics['accuracy']:.5f}",
            runtime,
        )
        if is_better_checkpoint(
            metrics["macro_f1"], metrics["loss"], best_metric, best_loss
        ):
            best_epoch = epoch
            best_metric = metrics["macro_f1"]
            best_loss = metrics["loss"]

    best_oof_record = oof_history[best_epoch - 1]
    original_frame = frame.iloc[original_indices]

    def export_oof_artifacts() -> None:
        scalar_rows = [
            {
                "epoch": record["epoch"],
                "loss": record["loss"],
                "macro_f1": record["macro_f1"],
                "accuracy": record["accuracy"],
                "selected": record["epoch"] == best_epoch,
            }
            for record in oof_history
        ]
        pd.DataFrame(scalar_rows).to_csv(
            output_dir / "oof_epoch_results.csv", index=False, encoding="utf-8"
        )
        _write_json(
            output_dir / "oof_selection_summary.json",
            {
                "selection_rule": (
                    "highest aggregate OOF macro_f1; lowest aggregate OOF loss "
                    "breaks exact ties"
                ),
                "selection_seed": config.seed,
                "validation_folds": list(folds),
                "best_epoch": best_epoch,
                "best_macro_f1": best_metric,
                "best_loss": best_loss,
                "folds": fold_summaries,
                "epochs": oof_history,
            },
        )
        write_validation_confusion_history(oof_history, output_dir)
        for record in oof_history:
            write_validation_confusion_matrix(
                record["confusion_matrix"], record["epoch"], output_dir
            )
            _write_prediction_diagnostics(
                original_frame,
                task,
                oof_probabilities_by_epoch[record["epoch"]],
                output_dir / f"oof_predictions_epoch_{record['epoch']}.csv",
            )
        _write_prediction_diagnostics(
            original_frame,
            task,
            oof_probabilities_by_epoch[best_epoch],
            output_dir / "oof_predictions.csv",
        )
        for validation_fold in folds:
            fold_indices = np.flatnonzero(
                original_frame["fold"].to_numpy() == validation_fold
            )
            _write_prediction_diagnostics(
                original_frame.iloc[fold_indices],
                task,
                oof_probabilities_by_epoch[best_epoch][fold_indices],
                output_dir
                / f"oof_fold_{validation_fold}"
                / "validation_predictions.csv",
            )

    on_main(export_oof_artifacts, runtime)
    log(
        f"\nOOF selected global epoch={best_epoch}, "
        f"Macro-F1={best_metric:.5f}, loss={best_loss:.5f}.",
        runtime,
    )

    full_train_data = CompletionDataset(
        frame["text"].tolist(), encoder, frame["label_id"].tolist()
    )
    full_class_weights, full_class_weight_summary = _class_weight_state(
        task, frame["label_id"].to_numpy()
    )
    log(
        "Full-data class weights: "
        + " | ".join(
            f"{label}={full_class_weight_summary['weights'][label]:.4f}"
            for label in task.labels
        ),
        runtime,
    )
    retrained_results = []
    for seed_number, seed in enumerate(retrain_seeds, start=1):
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_checkpoint_dir = checkpoint_dir / f"seed_{seed}"
        final_adapter_dir = seed_checkpoint_dir / "final_full_data_adapter"
        on_main(
            partial(seed_output_dir.mkdir, parents=True, exist_ok=True), runtime
        )
        on_main(
            partial(seed_checkpoint_dir.mkdir, parents=True, exist_ok=True), runtime
        )
        log(
            f"\n=== PHASE 2/2 | FULL-DATA MODEL {seed_number}/{len(retrain_seeds)} "
            f"| seed={seed} | rows={len(full_train_data)} | epochs={best_epoch} ===",
            runtime,
        )
        set_global_seed(seed)
        full_loader, full_sampler = create_train_loader(
            full_train_data, collator, config, runtime, seed
        )
        model = _wrap_for_ddp(load_lora_model(config, runtime), runtime)
        optimizer, scheduler, scaler = _training_state(
            model, full_loader, best_epoch, config, runtime
        )
        full_history = []
        for epoch in range(1, best_epoch + 1):
            if full_sampler is not None:
                full_sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                model,
                full_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                runtime,
                epoch,
                f"Full-data retrain seed {seed}",
                class_weights=full_class_weights,
            )
            full_history.append(
                {"epoch": epoch, "train_weighted_loss": train_loss}
            )
            log(
                f"Seed {seed} full-data epoch {epoch}/{best_epoch}: "
                f"train_weighted_loss={train_loss:.5f}",
                runtime,
            )
            on_main(
                lambda seed_output_dir=seed_output_dir,
                seed=seed,
                full_history=list(full_history): _write_json(
                    seed_output_dir / "full_data_training_history.json",
                    {
                        "seed": seed,
                        "class_weighting": full_class_weight_summary,
                        "epochs": full_history,
                    },
                ),
                runtime,
            )

        del optimizer, scheduler, scaler
        on_main(partial(save_adapter, model, tokenizer, final_adapter_dir), runtime)
        restore_adapter(model, final_adapter_dir)
        predictions = score_candidates(
            model,
            prediction_data,
            collator,
            config,
            runtime,
            f"Development prediction seed {seed}",
        )
        probabilities = predictions["probabilities"]
        label_ids = probabilities.argmax(axis=1)
        summary = {
            "seed": seed,
            "epochs": best_epoch,
            "training_rows": len(full_train_data),
            "checkpoint": str(final_adapter_dir),
            "prediction_source": "seed_specific_full_data_adapter",
        }

        def export_seed_predictions(
            seed_output_dir=seed_output_dir,
            label_ids=label_ids,
            summary=summary,
        ) -> None:
            write_submission(
                prediction_frame["id"].tolist(),
                [task.labels[index] for index in label_ids],
                task.labels,
                seed_output_dir,
                task.submission_zip_name,
            )
            _write_json(seed_output_dir / "seed_summary.json", summary)

        on_main(export_seed_predictions, runtime)
        retrained_results.append(
            {
                "seed": seed,
                "probabilities": probabilities,
                "label_ids": label_ids,
                "summary": summary,
            }
        )
        del model, full_loader
        gc.collect()
        torch.cuda.empty_cache()

    stacked_probabilities = np.stack(
        [result["probabilities"] for result in retrained_results], axis=0
    )
    mean_probabilities = stacked_probabilities.mean(axis=0)
    final_ids = mean_probabilities.argmax(axis=1)
    final_labels = [task.labels[index] for index in final_ids]

    def export_final_ensemble() -> None:
        ensemble_data: dict[str, object] = {
            "id": prediction_frame["id"].astype(str).tolist()
        }
        for result in retrained_results:
            ensemble_data[f"seed_{result['seed']}"] = [
                task.labels[index] for index in result["label_ids"]
            ]
        for label_index, label in enumerate(task.labels):
            safe_label = label.lower().replace("-", "_").replace(" ", "_")
            ensemble_data[f"mean_prob_{safe_label}"] = mean_probabilities[
                :, label_index
            ]
            ensemble_data[f"votes_{safe_label}"] = np.sum(
                np.stack(
                    [result["label_ids"] for result in retrained_results], axis=0
                )
                == label_index,
                axis=0,
            )
        ensemble_data["final_label"] = final_labels
        pd.DataFrame(ensemble_data).to_csv(
            output_dir / "ensemble_predictions.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        csv_path, zip_path = write_submission(
            prediction_frame["id"].tolist(),
            final_labels,
            task.labels,
            output_dir,
            task.submission_zip_name,
        )
        ensemble_summary = {
            "method": "equal-weight mean of per-seed candidate probabilities",
            "seeds": list(retrain_seeds),
            "models": len(retrained_results),
            "prediction_file": str(csv_path),
            "audit_file": str(output_dir / "ensemble_predictions.csv"),
            "submission": str(zip_path),
        }
        retrain_summaries = [result["summary"] for result in retrained_results]
        _write_json(
            checkpoint_dir / "metrics.json",
            {
                "oof_selection": best_oof_record,
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
            },
        )
        _write_json(
            output_dir / "run_summary.json",
            {
                "task": task.task_name,
                "method": (
                    "LoRA answer-only causal LM with five-fold OOF epoch "
                    "selection and full-data soft seed ensemble"
                ),
                "model": config.model_name,
                "resolved_model_revision": resolved_revision,
                "world_size": runtime.world_size,
                "compute_dtype": str(compute_dtype(config, runtime)),
                "quantization": None,
                "effective_train_batch_size": config.train_batch_size_per_gpu
                * config.gradient_accumulation_steps
                * runtime.world_size,
                "oof": {
                    "selection_seed": config.seed,
                    "validation_folds": list(folds),
                    "validation_origins": list(
                        task.selection_validation_data_origins
                    ),
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_metric,
                    "best_loss": best_loss,
                    "original_rows": len(original_indices),
                    "folds": fold_summaries,
                    "epochs": oof_history,
                },
                "class_weighting": {
                    "downsampling": None,
                    "full_data": full_class_weight_summary,
                },
                "full_data_retrains": retrain_summaries,
                "ensemble": ensemble_summary,
                "development_rows": len(prediction_data),
                "training_config": asdict(config),
                "parameters": parameter_counts,
                "system_prompt": encoder.system_text,
                "label_order": list(task.labels),
                "completion_token_ids": encoder.completions,
                "completion_token_diagnostics": token_diagnostics,
                "inference": (
                    "mean token log likelihood per allowed code, softmax per model, "
                    "then equal-weight mean probabilities across full-data seeds"
                ),
                "truncated_comments": {
                    "full_train": full_train_data.truncated_count,
                    "development": prediction_data.truncated_count,
                },
                "input_sha256": {
                    name: hashlib.sha256(resolve_data_file(name).read_bytes()).hexdigest()
                    for name in (*task.labeled_filenames(), task.prediction_filename)
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "peft",
                        "accelerate",
                        "scikit-learn",
                    )
                },
            },
        )
        log(
            f"\n=== COMPLETE | {task.task_name} ===\n"
            f"OOF best epoch: {best_epoch} | Macro-F1={best_metric:.5f}\n"
            f"Full-data seeds: {list(retrain_seeds)}\n"
            f"Ensemble: equal-weight soft probability mean\n"
            f"Predictions: {csv_path}\nSubmission: {zip_path}",
            runtime,
        )

    on_main(export_final_ensemble, runtime)


def run_llm_task(task: TaskConfig, config: LLMTrainingConfig) -> None:
    """Select epochs, retrain configured seeds, then predict development."""

    validate_config(config)
    selection_seeds = task.selection_seeds or (config.seed,)
    if len(set(selection_seeds)) != len(selection_seeds):
        raise ValueError("selection_seeds must not contain duplicates")
    if any(seed < 0 for seed in selection_seeds):
        raise ValueError("selection_seeds must contain non-negative integers")
    if task.ensemble_all_seeds:
        if len(selection_seeds) % 2 == 0:
            raise ValueError("ensemble_all_seeds requires an odd number of seeds")
        if set(task.labels) != {"Non-Hate", "Hate"} or len(task.labels) != 2:
            raise ValueError(
                "ensemble_all_seeds currently supports binary Non-Hate/Hate only"
            )
    if task.fixed_full_retrain_epochs is not None:
        if task.fixed_full_retrain_epochs < 1:
            raise ValueError("fixed_full_retrain_epochs must be positive")
        if task.ensemble_all_seeds:
            raise ValueError(
                "Fixed full-data soft ensembling and binary hard voting are exclusive."
            )
        if not task.original_fold_filenames:
            raise ValueError("Fixed full-data training requires predefined fold files.")
        if not task.full_retrain_soft_ensemble:
            raise ValueError("Fixed full-data training requires a soft ensemble.")
    if task.oof_validation_folds:
        if task.ensemble_all_seeds:
            raise ValueError("OOF soft ensembling and binary hard voting are exclusive.")
        if not task.original_fold_filenames:
            raise ValueError("OOF selection requires predefined fold files.")
        if not task.full_retrain_soft_ensemble:
            raise ValueError("OOF selection requires a full-data soft ensemble.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime = initialize_runtime()
    try:
        require_cuda(runtime)
        set_global_seed(config.seed)
        log(
            f"\n{task.task_name} | LoRA {config.model_name} | rank={config.lora_rank} "
            f"| world_size={runtime.world_size} | selection_seeds={selection_seeds}",
            runtime,
        )
        log(
            f"Compute={compute_dtype(config, runtime)}; effective batch="
            f"{config.train_batch_size_per_gpu * config.gradient_accumulation_steps * runtime.world_size}",
            runtime,
        )

        frame, prediction_frame = load_task_frames(task)
        for name, data in (("train", frame), ("development", prediction_frame)):
            if data.empty:
                raise ValueError(f"{name} input must contain at least one row.")
        if task.fixed_full_retrain_epochs is not None:
            base_train_indices = val_indices = None
            split_strategy = "fixed_epoch_full_data"
            split_heading = (
                f"OOF skipped / full-data retrain for "
                f"{task.fixed_full_retrain_epochs} fixed epoch(s)"
            )
        elif task.oof_validation_folds:
            base_train_indices = val_indices = None
            split_strategy = "five_fold_oof"
            split_heading = "five-fold OOF epoch selection"
        elif task.selection_validation_fold is None:
            base_train_indices, val_indices = stratified_holdout_indices(
                frame["label_id"].to_numpy(), config.validation_size, config.seed
            )
            split_strategy = "deterministic_stratified_holdout"
            split_heading = "80% train / 20% validation"
        else:
            base_train_indices, val_indices = predefined_fold_indices(
                frame["fold"].to_numpy(),
                task.selection_validation_fold,
                frame["data_origin"].to_numpy(),
                task.selection_validation_data_origins,
            )
            split_strategy = "predefined_fold_holdout"
            validation_origin_heading = (
                " + ".join(task.selection_validation_data_origins)
                if task.selection_validation_data_origins
                else "all origins"
            )
            split_heading = (
                f"folds 1-{task.selection_validation_fold - 1} train / "
                f"fold {task.selection_validation_fold} "
                f"{validation_origin_heading} validation"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        encoder = ChatEncoder(tokenizer, task, config.max_length)
        token_diagnostics = completion_token_diagnostics(encoder)
        log_completion_token_diagnostics(token_diagnostics, runtime)
        prediction_data = CompletionDataset(prediction_frame["text"].tolist(), encoder)
        collator = CompletionCollator(tokenizer.pad_token_id)

        output_dir = OUTPUT_ROOT / task.output_subdir
        checkpoint_dir = CHECKPOINT_ROOT / task.checkpoint_subdir
        final_adapter_dir = checkpoint_dir / "final_full_data_adapter"
        on_main(lambda: output_dir.mkdir(parents=True, exist_ok=True), runtime)
        on_main(lambda: checkpoint_dir.mkdir(parents=True, exist_ok=True), runtime)

        if task.fixed_full_retrain_epochs is not None:
            _run_fixed_epoch_full_retrain_ensemble(
                task=task,
                config=config,
                runtime=runtime,
                frame=frame,
                prediction_frame=prediction_frame,
                prediction_data=prediction_data,
                collator=collator,
                encoder=encoder,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                retrain_seeds=selection_seeds,
                token_diagnostics=token_diagnostics,
                fixed_epochs=task.fixed_full_retrain_epochs,
            )
            return

        if task.oof_validation_folds:
            _run_oof_full_retrain_ensemble(
                task=task,
                config=config,
                runtime=runtime,
                frame=frame,
                prediction_frame=prediction_frame,
                prediction_data=prediction_data,
                collator=collator,
                encoder=encoder,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                retrain_seeds=selection_seeds,
                token_diagnostics=token_diagnostics,
            )
            return

        val_frame = frame.iloc[val_indices]
        val_data = CompletionDataset(
            val_frame["text"].tolist(), encoder, val_frame["label_id"].tolist()
        )

        seed_results = []
        resolved_revision = None
        parameter_counts = None
        labels = frame[task.label_column].to_numpy()
        for seed_number, selection_seed in enumerate(selection_seeds, start=1):
            set_global_seed(selection_seed)
            train_indices, downsampled_indices, downsampling_summary = (
                cap_training_labels(
                    base_train_indices,
                    labels,
                    task.selection_train_label_caps,
                    selection_seed,
                )
            )
            full_train_indices = np.setdiff1d(
                np.arange(len(frame), dtype=np.int64),
                downsampled_indices,
                assume_unique=True,
            )
            train_frame = frame.iloc[train_indices]
            selection_train_data = CompletionDataset(
                train_frame["text"].tolist(),
                encoder,
                train_frame["label_id"].tolist(),
            )
            multiple_seeds = len(selection_seeds) > 1
            seed_output_dir = (
                output_dir / f"seed_{selection_seed}"
                if multiple_seeds
                else output_dir
            )
            seed_checkpoint_dir = (
                checkpoint_dir / f"seed_{selection_seed}"
                if multiple_seeds
                else checkpoint_dir
            )
            selection_adapter_dir = seed_checkpoint_dir / "best_holdout_adapter"

            def initialize_seed_artifacts(
                seed_output_dir=seed_output_dir,
                seed_checkpoint_dir=seed_checkpoint_dir,
                train_indices=train_indices,
                downsampled_indices=downsampled_indices,
            ):
                seed_output_dir.mkdir(parents=True, exist_ok=True)
                seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                write_split_manifest(
                    frame,
                    train_indices,
                    val_indices,
                    seed_output_dir,
                    downsampled_indices,
                )
                write_downsampled_rows(
                    frame, task, downsampled_indices, seed_output_dir
                )

            on_main(initialize_seed_artifacts, runtime)
            log(
                f"\n=== PHASE 1/2 | SEED {seed_number}/{len(selection_seeds)} "
                f"| seed={selection_seed} | HOLDOUT SELECTION ({split_heading}) ===",
                runtime,
            )
            log(
                f"Selection train={len(selection_train_data)}, validation={len(val_data)}, "
                f"final retrain candidate={len(full_train_indices)}/{len(frame)} rows.",
                runtime,
            )
            for summary in downsampling_summary:
                log(
                    f"Selection label cap: {summary['label']} available="
                    f"{summary['available']}, retained={summary['retained']}, "
                    f"dropped={summary['dropped']}, cap={summary['cap']}, "
                    f"seed={selection_seed}.",
                    runtime,
                )

            selection_loader, selection_sampler = create_train_loader(
                selection_train_data, collator, config, runtime, selection_seed
            )
            selection_model = load_lora_model(config, runtime)
            if parameter_counts is None:
                base = selection_model.get_base_model()
                resolved_revision = getattr(base.config, "_commit_hash", None)
                parameter_counts = {
                    "trainable": sum(
                        parameter.numel()
                        for parameter in selection_model.parameters()
                        if parameter.requires_grad
                    ),
                    "reported_total": selection_model.num_parameters(),
                }
                del base
            selection_model = _wrap_for_ddp(selection_model, runtime)
            optimizer, scheduler, scaler = _training_state(
                selection_model, selection_loader, config.max_epochs, config, runtime
            )
            best_metric, best_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
            selection_history = []
            for epoch in range(1, config.max_epochs + 1):
                if selection_sampler is not None:
                    selection_sampler.set_epoch(epoch)
                train_loss = train_llm_epoch(
                    selection_model,
                    selection_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    runtime,
                    epoch,
                    "Selection train",
                )
                metrics = score_candidates(
                    selection_model,
                    val_data,
                    collator,
                    config,
                    runtime,
                    f"Epoch {epoch} validation",
                )
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{
                        key: metrics[key]
                        for key in ("loss", "macro_f1", "accuracy")
                    },
                    "per_class_f1": metrics["per_class_f1"],
                }
                for key in ("per_class", "confusion_matrix"):
                    if key in metrics:
                        record[key] = metrics[key]
                selection_history.append(record)
                if "confusion_matrix" in metrics:
                    on_main(
                        partial(
                            write_validation_confusion_matrix,
                            metrics["confusion_matrix"],
                            epoch,
                            seed_output_dir,
                        ),
                        runtime,
                    )
                    on_main(
                        partial(
                            write_validation_confusion_history,
                            selection_history,
                            seed_output_dir,
                        ),
                        runtime,
                    )
                log(
                    f"Seed {selection_seed} epoch {epoch}: train_loss={train_loss:.5f}, "
                    f"val_loss={metrics['loss']:.5f}, macro_f1={metrics['macro_f1']:.5f}, "
                    f"accuracy={metrics['accuracy']:.5f}",
                    runtime,
                )
                log(
                    "Validation F1 by class: "
                    + " | ".join(
                        f"{label}={metrics['per_class_f1'][label]:.5f}"
                        for label in task.labels
                    ),
                    runtime,
                )
                for diagnostic_line in format_per_class_diagnostics(metrics):
                    log(diagnostic_line, runtime)
                if is_better_checkpoint(
                    metrics["macro_f1"], metrics["loss"], best_metric, best_loss
                ):
                    best_metric, best_loss, best_epoch, stale = (
                        metrics["macro_f1"],
                        metrics["loss"],
                        epoch,
                        0,
                    )
                    on_main(
                        partial(
                            save_adapter,
                            selection_model,
                            tokenizer,
                            selection_adapter_dir,
                        ),
                        runtime,
                    )
                else:
                    stale += 1
                on_main(
                    lambda seed_output_dir=seed_output_dir,
                    selection_history=selection_history,
                    selection_seed=selection_seed: _write_json(
                        seed_output_dir / "selection_training_history.json",
                        {"seed": selection_seed, "epochs": selection_history},
                    ),
                    runtime,
                )
                if stale >= config.early_stopping_patience:
                    log(
                        f"Seed {selection_seed} early stopping at epoch {epoch}.",
                        runtime,
                    )
                    break

            del optimizer, scheduler, scaler
            restore_adapter(selection_model, selection_adapter_dir)
            best_validation = score_candidates(
                selection_model,
                val_data,
                collator,
                config,
                runtime,
                "Best holdout adapter validation",
            )
            public_summary = {
                "seed": selection_seed,
                "best_epoch": best_epoch,
                "checkpoint": str(selection_adapter_dir),
                **{
                    key: best_validation[key]
                    for key in ("macro_f1", "accuracy", "loss", "per_class_f1")
                },
                "selection_train_rows": len(selection_train_data),
                "full_retrain_rows": len(full_train_indices),
                "downsampled_rows": len(downsampled_indices),
                "downsampling_summary": downsampling_summary,
            }
            for key in ("per_class", "confusion_matrix"):
                if key in best_validation:
                    public_summary[key] = best_validation[key]

            def export_seed_artifacts(
                seed_output_dir=seed_output_dir,
                best_validation=best_validation,
                public_summary=public_summary,
            ):
                write_validation_predictions(
                    val_frame,
                    task,
                    best_validation["probabilities"],
                    seed_output_dir,
                )
                _write_json(seed_output_dir / "seed_summary.json", public_summary)

            on_main(export_seed_artifacts, runtime)
            seed_results.append(
                {
                    "summary": public_summary,
                    "train_indices": train_indices,
                    "downsampled_indices": downsampled_indices,
                    "full_train_indices": full_train_indices,
                    "train_frame": train_frame,
                    "selection_train_truncated": selection_train_data.truncated_count,
                    "selection_history": selection_history,
                    "validation_probabilities": best_validation["probabilities"],
                    "validation_indices": val_indices,
                    "seed_output_dir": seed_output_dir,
                    "seed_checkpoint_dir": seed_checkpoint_dir,
                }
            )
            log(
                f"Seed {selection_seed} completed: best Macro-F1="
                f"{best_validation['macro_f1']:.5f} at epoch {best_epoch}.",
                runtime,
            )
            del selection_model, selection_loader, selection_train_data
            gc.collect()
            torch.cuda.empty_cache()

        if task.ensemble_all_seeds:
            _run_seed_ensemble(
                task=task,
                config=config,
                runtime=runtime,
                frame=frame,
                prediction_frame=prediction_frame,
                val_frame=val_frame,
                val_data=val_data,
                prediction_data=prediction_data,
                collator=collator,
                encoder=encoder,
                tokenizer=tokenizer,
                seed_results=seed_results,
                selection_seeds=selection_seeds,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                split_strategy=split_strategy,
                resolved_revision=resolved_revision,
                parameter_counts=parameter_counts,
                token_diagnostics=token_diagnostics,
            )
            return

        winner = seed_results[0]
        for candidate in seed_results[1:]:
            candidate_summary = candidate["summary"]
            winner_summary = winner["summary"]
            if is_better_checkpoint(
                candidate_summary["macro_f1"],
                candidate_summary["loss"],
                winner_summary["macro_f1"],
                winner_summary["loss"],
            ):
                winner = candidate
        selection_summary = winner["summary"]
        winning_seed = selection_summary["seed"]
        best_epoch = selection_summary["best_epoch"]
        train_indices = winner["train_indices"]
        downsampled_indices = winner["downsampled_indices"]
        full_train_indices = winner["full_train_indices"]
        train_frame = winner["train_frame"]
        full_train_frame = frame.iloc[full_train_indices]
        full_train_data = CompletionDataset(
            full_train_frame["text"].tolist(),
            encoder,
            full_train_frame["label_id"].tolist(),
        )

        def export_winner_artifacts():
            write_split_manifest(
                frame, train_indices, val_indices, output_dir, downsampled_indices
            )
            write_downsampled_rows(frame, task, downsampled_indices, output_dir)
            write_validation_predictions(
                val_frame,
                task,
                winner["validation_probabilities"],
                output_dir,
            )
            _write_json(
                output_dir / "selection_training_history.json",
                {
                    "seed": winning_seed,
                    "epochs": winner["selection_history"],
                },
            )
            for record in winner["selection_history"]:
                if "confusion_matrix" in record:
                    write_validation_confusion_matrix(
                        record["confusion_matrix"], record["epoch"], output_dir
                    )
            write_validation_confusion_history(
                winner["selection_history"], output_dir
            )
            scalar_results = []
            for result in seed_results:
                summary = result["summary"]
                scalar_results.append(
                    {
                        "seed": summary["seed"],
                        "best_epoch": summary["best_epoch"],
                        "macro_f1": summary["macro_f1"],
                        "loss": summary["loss"],
                        "accuracy": summary["accuracy"],
                        "selection_train_rows": summary["selection_train_rows"],
                        "full_retrain_rows": summary["full_retrain_rows"],
                        "downsampled_rows": summary["downsampled_rows"],
                        "checkpoint": summary["checkpoint"],
                        "selected": summary["seed"] == winning_seed,
                    }
                )
            write_seed_selection_results(scalar_results, output_dir)
            macro_scores = [row["macro_f1"] for row in scalar_results]
            _write_json(
                output_dir / "seed_selection_summary.json",
                {
                    "selection_rule": "highest macro_f1; lowest loss breaks exact ties",
                    "validation_is_fixed_across_seeds": True,
                    "split_seed": config.seed,
                    "winning_seed": winning_seed,
                    "macro_f1_mean": float(np.mean(macro_scores)),
                    "macro_f1_std": float(np.std(macro_scores)),
                    "results": [result["summary"] for result in seed_results],
                },
            )

        on_main(export_winner_artifacts, runtime)
        log(
            "Seed comparison: "
            + " | ".join(
                f"seed={result['summary']['seed']} "
                f"Macro-F1={result['summary']['macro_f1']:.5f} "
                f"epoch={result['summary']['best_epoch']}"
                for result in seed_results
            ),
            runtime,
        )
        log(
            f"\nWinning seed={winning_seed}, Macro-F1="
            f"{selection_summary['macro_f1']:.5f}, best_epoch={best_epoch}.",
            runtime,
        )

        # Replay only the winning seed and epoch count on every retained row.
        log(
            f"\n=== PHASE 2/2 | WINNING SEED {winning_seed} | FULL-DATA RETRAIN "
            f"({len(full_train_data)} retained rows / {best_epoch} epoch(s)) ===",
            runtime,
        )
        set_global_seed(winning_seed)
        full_loader, full_sampler = create_train_loader(
            full_train_data, collator, config, runtime, winning_seed
        )
        final_model = _wrap_for_ddp(load_lora_model(config, runtime), runtime)
        optimizer, scheduler, scaler = _training_state(
            final_model, full_loader, best_epoch, config, runtime
        )
        full_history = []
        for epoch in range(1, best_epoch + 1):
            if full_sampler is not None:
                full_sampler.set_epoch(epoch)
            train_loss = train_llm_epoch(
                final_model,
                full_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                runtime,
                epoch,
                "Full-data retrain",
            )
            full_history.append({"epoch": epoch, "train_loss": train_loss})
            log(
                f"Full-data epoch {epoch}/{best_epoch}: train_loss={train_loss:.5f}",
                runtime,
            )
            on_main(
                lambda: _write_json(
                    output_dir / "full_data_training_history.json",
                    {"seed": winning_seed, "epochs": full_history},
                ),
                runtime,
            )

        del optimizer, scheduler, scaler
        on_main(
            partial(save_adapter, final_model, tokenizer, final_adapter_dir), runtime
        )
        restore_adapter(final_model, final_adapter_dir)
        predictions = score_candidates(
            final_model,
            prediction_data,
            collator,
            config,
            runtime,
            "Development prediction from full-data adapter",
        )

        def export_artifacts():
            label_ids = predictions["probabilities"].argmax(axis=1)
            csv_path, zip_path = write_submission(
                prediction_frame["id"].tolist(),
                [task.labels[index] for index in label_ids],
                task.labels,
                output_dir,
                task.submission_zip_name,
            )
            final_summary = {
                "seed": winning_seed,
                "epochs": best_epoch,
                "training_rows": len(full_train_data),
                "loaded_rows": len(frame),
                "downsampled_rows_excluded": len(downsampled_indices),
                "label_distribution": full_train_frame[task.label_column]
                .value_counts()
                .to_dict(),
                "checkpoint": str(final_adapter_dir),
                "prediction_source": "final_full_data_adapter",
            }
            _write_json(
                checkpoint_dir / "metrics.json",
                {
                    "selection": selection_summary,
                    "all_seed_results": [result["summary"] for result in seed_results],
                    "full_data_retrain": final_summary,
                },
            )
            _write_json(
                output_dir / "run_summary.json",
                {
                    "task": task.task_name,
                    "method": "LoRA answer-only causal LM",
                    "model": config.model_name,
                    "resolved_model_revision": resolved_revision,
                    "world_size": runtime.world_size,
                    "compute_dtype": str(compute_dtype(config, runtime)),
                    "quantization": None,
                    "text_input": (
                        "ftfy + Unicode NFKC + HTML entity decoding + whitespace "
                        "normalization; URLs, mentions, and HTML tags preserved"
                    ),
                    "effective_train_batch_size": config.train_batch_size_per_gpu
                    * config.gradient_accumulation_steps
                    * runtime.world_size,
                    "selection_seeds": list(selection_seeds),
                    "winning_seed": winning_seed,
                    "selection_split": {
                        "strategy": split_strategy,
                        "split_seed": config.seed,
                        "winning_downsampling_seed": winning_seed,
                        "configured_validation_size": config.validation_size
                        if task.selection_validation_fold is None
                        else None,
                        "validation_fold": task.selection_validation_fold,
                        "validation_data_origins": list(
                            task.selection_validation_data_origins
                        ),
                        "actual_validation_fraction": len(val_data) / len(frame),
                        "train_rows": len(train_indices),
                        "validation_rows": len(val_data),
                        "selection_excluded_rows": len(frame)
                        - len(train_indices)
                        - len(val_data),
                        "selection_downsampled_rows": len(downsampled_indices),
                        "selection_train_label_caps": {
                            summary["label"]: summary["cap"]
                            for summary in selection_summary["downsampling_summary"]
                        },
                        "downsampling_summary": selection_summary[
                            "downsampling_summary"
                        ],
                        "train_label_distribution": train_frame[task.label_column]
                        .value_counts()
                        .to_dict(),
                        "validation_label_distribution": val_frame[task.label_column]
                        .value_counts()
                        .to_dict(),
                    },
                    "seed_selection": [
                        result["summary"] for result in seed_results
                    ],
                    "best_checkpoint": selection_summary,
                    "full_data_retrain": final_summary,
                    "development_rows": len(prediction_data),
                    "training_config": asdict(config),
                    "parameters": parameter_counts,
                    "system_prompt": encoder.system_text,
                    "label_order": list(task.labels),
                    "completion_token_ids": encoder.completions,
                    "completion_token_diagnostics": token_diagnostics,
                    "completion_token_lengths_equal": len(
                        {int(item["token_count"]) for item in token_diagnostics}
                    )
                    == 1,
                    "inference": (
                        "argmax mean token log P(answer including chat end | prompt)"
                    ),
                    "probabilities": (
                        "softmax over allowed mean token log likelihoods; not calibrated"
                    ),
                    "loss": "mean per-example answer-token NLL; not class cross-entropy",
                    "truncated_comments": {
                        "winning_selection_train": winner[
                            "selection_train_truncated"
                        ],
                        "validation": val_data.truncated_count,
                        "full_train": full_train_data.truncated_count,
                        "development": prediction_data.truncated_count,
                    },
                    "input_sha256": {
                        name: hashlib.sha256(
                            resolve_data_file(name).read_bytes()
                        ).hexdigest()
                        for name in (*task.labeled_filenames(), task.prediction_filename)
                    },
                    "packages": {
                        name: version(name)
                        for name in (
                            "torch",
                            "transformers",
                            "peft",
                            "accelerate",
                            "scikit-learn",
                        )
                    },
                },
            )
            log(
                f"\n=== COMPLETE | {task.task_name} ===\n"
                f"Winning seed: {winning_seed}\n"
                f"Best validation Macro-F1: {selection_summary['macro_f1']:.5f} "
                f"at epoch {best_epoch}\n"
                f"Retrained from scratch on all {len(full_train_data)} retained rows "
                f"for {best_epoch} epoch(s).\n"
                f"Seed comparison: {output_dir / 'seed_selection_results.csv'}\n"
                f"Predictions: {csv_path}\nSubmission: {zip_path}\n"
                f"Winning downsampled rows: {output_dir / 'downsampled_rows.csv'}\n"
                f"Final adapter: {final_adapter_dir}",
                runtime,
            )

        on_main(export_artifacts, runtime)
        del final_model, full_loader
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
