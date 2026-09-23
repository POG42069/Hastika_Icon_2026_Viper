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
    OUTPUT_ROOT,
    LLM_TRAINING,
    LLMTrainingConfig,
    TaskConfig,
)
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
    cap_training_labels,
    create_train_loader,
    is_better_checkpoint,
    log,
    predefined_fold_indices,
    set_global_seed,
    stratified_holdout_indices,
    unwrap_model,
    write_downsampled_rows,
    write_split_manifest,
    write_validation_confusion_history,
    write_validation_confusion_matrix,
    write_validation_predictions,
    write_seed_selection_results,
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
        encoder.task.labels,
        encoder.task.label_codes,
        encoder.completions,
        strict=True,
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
    architecture supports logits_to_keep; preserve the full attention
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
    model, loader, optimizer, scheduler, scaler, config, runtime, epoch, phase="LLM"
):
    """Mean answer-token NLL per example; correctly scale even a partial group."""
    model.train()
    dtype = compute_dtype(config, runtime)
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
                    loss = example_losses.sum() / group_size
                scaler.scale(loss).backward()
            total_loss += float(example_losses.detach().sum())
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


def ensemble_plurality_vote(
    seed_label_ids: np.ndarray,
    seed_mean_token_log_probs: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vote equally across seeds and resolve multiclass ties deterministically.

    Ties in the plurality vote are resolved by summed, tie-aware Borda points,
    then by summed per-seed z-normalized mean token log probabilities. An exact
    tie after all score-based rules falls back to the configured label order.
    """

    label_ids = np.asarray(seed_label_ids, dtype=np.int64)
    scores = np.asarray(seed_mean_token_log_probs, dtype=np.float64)
    if label_ids.ndim != 2 or label_ids.shape[0] < 1:
        raise ValueError("seed_label_ids must have shape (seeds, prediction rows)")
    if len(labels) < 2:
        raise ValueError("Ensemble voting requires at least two labels")
    expected_score_shape = (*label_ids.shape, len(labels))
    if scores.shape != expected_score_shape:
        raise ValueError(
            "seed_mean_token_log_probs must have shape "
            f"{expected_score_shape}, received {scores.shape}"
        )
    if np.any((label_ids < 0) | (label_ids >= len(labels))):
        raise ValueError("seed_label_ids contains an invalid label index")
    if not np.all(np.isfinite(scores)):
        raise ValueError("seed_mean_token_log_probs must contain finite values")

    seed_count, row_count = label_ids.shape
    vote_counts = np.zeros((row_count, len(labels)), dtype=np.int64)
    for label_index in range(len(labels)):
        vote_counts[:, label_index] = np.sum(label_ids == label_index, axis=0)

    # A candidate receives one Borda point for every candidate it outscores.
    # Equal candidate scores therefore receive equal points and do not silently
    # inherit a preference from array order.
    borda_points = np.sum(
        scores[:, :, :, np.newaxis] > scores[:, :, np.newaxis, :], axis=-1
    )
    borda_scores = np.sum(borda_points, axis=0, dtype=np.int64)

    score_means = np.mean(scores, axis=2, keepdims=True)
    score_stds = np.std(scores, axis=2, keepdims=True)
    normalized_scores = np.divide(
        scores - score_means,
        score_stds,
        out=np.zeros_like(scores),
        where=score_stds > 0,
    )
    normalized_score_sums = np.sum(normalized_scores, axis=0)

    final_ids = np.empty(row_count, dtype=np.int64)
    tie_detected = np.zeros(row_count, dtype=bool)
    tie_break_methods = np.empty(row_count, dtype=object)
    for row_index in range(row_count):
        candidates = np.flatnonzero(
            vote_counts[row_index] == vote_counts[row_index].max()
        )
        if len(candidates) == 1:
            final_ids[row_index] = candidates[0]
            tie_break_methods[row_index] = "plurality"
            continue

        tie_detected[row_index] = True
        candidate_borda = borda_scores[row_index, candidates]
        candidates = candidates[candidate_borda == candidate_borda.max()]
        if len(candidates) == 1:
            final_ids[row_index] = candidates[0]
            tie_break_methods[row_index] = "borda"
            continue

        candidate_scores = normalized_score_sums[row_index, candidates]
        best_score = candidate_scores.max()
        candidates = candidates[
            np.isclose(candidate_scores, best_score, rtol=1e-12, atol=1e-12)
        ]
        if len(candidates) == 1:
            final_ids[row_index] = candidates[0]
            tie_break_methods[row_index] = "normalized_score"
            continue

        final_ids[row_index] = candidates[0]
        tie_break_methods[row_index] = "label_order"

    if np.any(vote_counts.sum(axis=1) != seed_count):
        raise RuntimeError("Every prediction row must receive one vote per seed")
    return (
        final_ids,
        vote_counts,
        borda_scores,
        normalized_score_sums,
        tie_detected,
        tie_break_methods,
    )


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
    """Retrain every selected seed, predict, and ensemble their outputs."""

    if len(seed_results) != len(selection_seeds):
        raise RuntimeError("Every configured seed must finish before ensembling")
    if len(selection_seeds) < 2:
        raise ValueError("ensemble_all_seeds requires at least two seeds")
    if len(task.labels) == 2 and len(selection_seeds) % 2 == 0:
        raise ValueError("Binary seed ensembling requires an odd number of seeds")

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
                "mean_token_log_probs": predictions["mean_token_log_probs"],
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
    stacked_scores = np.stack(
        [result["mean_token_log_probs"] for result in retrained_results], axis=0
    )
    (
        final_ids,
        vote_counts,
        borda_scores,
        normalized_score_sums,
        tie_detected,
        tie_break_methods,
    ) = ensemble_plurality_vote(
        stacked_label_ids, stacked_scores, task.labels
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
    for seed_index, (row, seed_result) in enumerate(
        zip(stacked_label_ids, retrained_results, strict=True)
    ):
        seed_prefix = f"seed_{seed_result['seed']}"
        voting_data[seed_prefix] = [
            task.labels[index] for index in row
        ]
        for label_index, label in enumerate(task.labels):
            safe_label = label.lower().replace("-", "_").replace(" ", "_")
            voting_data[
                f"{seed_prefix}_mean_token_log_prob_{safe_label}"
            ] = stacked_scores[seed_index, :, label_index]
            voting_data[f"{seed_prefix}_prob_{safe_label}"] = seed_result[
                "probabilities"
            ][:, label_index]
    for label_index, label in enumerate(task.labels):
        safe_label = label.lower().replace("-", "_").replace(" ", "_")
        voting_data[f"votes_{safe_label}"] = vote_counts[:, label_index]
        voting_data[f"borda_{safe_label}"] = borda_scores[:, label_index]
        voting_data[f"normalized_score_{safe_label}"] = normalized_score_sums[
            :, label_index
        ]
        voting_data[f"mean_prob_{safe_label}"] = mean_probabilities[:, label_index]
    if set(task.labels) == {"Non-Hate", "Hate"} and len(task.labels) == 2:
        voting_data["hate_votes"] = vote_counts[:, task.labels.index("Hate")]
        voting_data["nonhate_votes"] = vote_counts[
            :, task.labels.index("Non-Hate")
        ]
    voting_data["max_votes"] = vote_counts.max(axis=1)
    voting_data["tie_detected"] = tie_detected
    voting_data["tie_break_method"] = tie_break_methods
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
        method_counts = {
            method: int(np.sum(tie_break_methods == method))
            for method in (
                "plurality",
                "borda",
                "normalized_score",
                "label_order",
            )
        }
        ensemble_summary = {
            "method": (
                "equal-weight plurality vote with Borda and normalized-score "
                "tie-breaks"
            ),
            "seeds": list(selection_seeds),
            "tie_break_order": [
                "summed tie-aware Borda points",
                "summed per-seed z-normalized mean token log probabilities",
                "configured label order",
            ],
            "fixed_label_order": list(task.labels),
            "rows_by_resolution_method": method_counts,
            "rows_with_plurality_tie": int(np.sum(tie_detected)),
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
                "method": (
                    "LoRA answer-only causal LM with equal-weight seed ensemble"
                ),
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
                    "prompt), followed by equal-weight plurality voting with "
                    "Borda, normalized-score, and fixed-label-order tie-breaks"
                ),
                "probabilities": (
                    "softmax over mean token log likelihoods; mean probabilities "
                    "are diagnostic only; tie score uses z-normalized raw mean "
                    "token log likelihoods"
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
            "Voting rule: equal-weight plurality; ties use Borda, normalized "
            "candidate score, then fixed label order.\n"
            f"Voting audit: {output_dir / 'voting_predictions.csv'}\n"
            f"Predictions: {csv_path}\nSubmission: {zip_path}",
            runtime,
        )

    on_main(export_ensemble_artifacts, runtime)


def run_llm_task(task: TaskConfig, config: LLMTrainingConfig) -> None:
    """Select epochs, retrain configured seeds, then predict development."""

    validate_config(config)
    selection_seeds = task.selection_seeds or (config.seed,)
    if len(set(selection_seeds)) != len(selection_seeds):
        raise ValueError("selection_seeds must not contain duplicates")
    if any(seed < 0 for seed in selection_seeds):
        raise ValueError("selection_seeds must contain non-negative integers")
    if task.ensemble_all_seeds:
        if len(selection_seeds) < 2:
            raise ValueError("ensemble_all_seeds requires at least two seeds")
        if len(task.labels) == 2 and len(selection_seeds) % 2 == 0:
            raise ValueError("Binary seed ensembling requires an odd number of seeds")
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
        if task.selection_validation_fold is None:
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
        val_frame = frame.iloc[val_indices]
        val_data = CompletionDataset(
            val_frame["text"].tolist(), encoder, val_frame["label_id"].tolist()
        )
        prediction_data = CompletionDataset(prediction_frame["text"].tolist(), encoder)
        collator = CompletionCollator(tokenizer.pad_token_id)

        output_dir = OUTPUT_ROOT / task.output_subdir
        checkpoint_dir = CHECKPOINT_ROOT / task.checkpoint_subdir
        final_adapter_dir = checkpoint_dir / "final_full_data_adapter"
        on_main(lambda: output_dir.mkdir(parents=True, exist_ok=True), runtime)
        on_main(lambda: checkpoint_dir.mkdir(parents=True, exist_ok=True), runtime)

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
                    selection_history=selection_history: _write_json(
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
