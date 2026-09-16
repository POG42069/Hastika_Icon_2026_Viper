"""Shared utilities for LoRA holdout selection and full-data retraining."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from llm_config import TaskConfig
from src.distributed import RuntimeContext


class TrainingConfigLike(Protocol):
    train_batch_size_per_gpu: int
    num_workers: int
    weight_decay: float
    learning_rate: float


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch reproducibly on every DDP worker."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(message: str, runtime: RuntimeContext) -> None:
    """Print once instead of duplicating messages from every DDP worker."""

    if runtime.is_main_process:
        print(message, flush=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying Hugging Face model beneath DDP."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def build_optimizer(model: nn.Module, config: TrainingConfigLike) -> AdamW:
    """Create AdamW for trainable LoRA parameters only."""

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
    if not decay_parameters and not no_decay_parameters:
        raise RuntimeError("The model has no trainable parameters.")
    return AdamW(
        [
            {"params": decay_parameters, "weight_decay": config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def create_train_loader(
    dataset,
    collator,
    config: TrainingConfigLike,
    runtime: RuntimeContext,
    shuffle_seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create a deterministic shuffled loader with one batch per GPU worker."""

    sampler: DistributedSampler | None = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=shuffle_seed,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
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


def stratified_holdout_indices(
    labels: np.ndarray,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one deterministic, stratified train/validation partition."""

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1")
    indices = np.arange(len(labels), dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
    )


def predefined_fold_indices(
    folds: np.ndarray,
    validation_fold: int,
    data_origins: np.ndarray | None = None,
    validation_origins: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Use earlier folds for training and selected origins of one fold for validation."""

    fold_values = np.asarray(folds)
    held_out_fold_mask = fold_values == validation_fold
    validation_mask = held_out_fold_mask.copy()
    if validation_origins:
        if data_origins is None:
            raise ValueError(
                "data_origins is required when validation_origins is configured."
            )
        origin_values = np.asarray(data_origins)
        if len(origin_values) != len(fold_values):
            raise ValueError("folds and data_origins must have the same length.")
        validation_mask &= np.isin(origin_values, validation_origins)
    if not validation_mask.any():
        origin_note = (
            f" with origins {validation_origins}" if validation_origins else ""
        )
        raise ValueError(
            f"No rows were found for validation fold {validation_fold}{origin_note}."
        )
    train_mask = ~held_out_fold_mask
    if not train_mask.any():
        raise ValueError("A predefined split must leave at least one training fold.")
    return (
        np.flatnonzero(train_mask).astype(np.int64),
        np.flatnonzero(validation_mask).astype(np.int64),
    )


def exclude_configured_training_rows(
    frame: pd.DataFrame,
    selection_train_indices: np.ndarray,
    excluded_ids: tuple[str, ...],
    label_column: str,
    expected_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude configured IDs from selection training and full retraining.

    Validation indices are intentionally managed separately by the caller, so
    a configured row in the holdout remains available for model selection but
    is still omitted from the later full-data retraining phase.
    """

    train_indices = np.asarray(selection_train_indices, dtype=np.int64)
    if not excluded_ids:
        return np.sort(train_indices), np.empty(0, dtype=np.int64)
    if len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("excluded_training_ids must be unique")

    ids = frame["id"].astype(str)
    duplicate_ids = ids[ids.duplicated(False)].unique().tolist()
    if duplicate_ids:
        raise ValueError(
            "Cannot apply an ID exclusion when labeled IDs are duplicated: "
            f"{duplicate_ids[:10]}"
        )
    index_by_id = pd.Series(np.arange(len(frame), dtype=np.int64), index=ids)
    missing = sorted(set(excluded_ids) - set(index_by_id.index))
    if missing:
        raise ValueError(
            "Configured training exclusions were not found in the labeled data: "
            f"{missing[:10]}"
        )

    excluded_indices = np.sort(index_by_id.loc[list(excluded_ids)].to_numpy())
    excluded_rows = frame.iloc[excluded_indices]
    if expected_label is not None:
        wrong_label = excluded_rows[label_column].ne(expected_label)
        if wrong_label.any():
            bad_ids = excluded_rows.loc[wrong_label, "id"].astype(str).head(10).tolist()
            raise ValueError(
                f"Configured exclusions must all have label {expected_label!r}: {bad_ids}"
            )
    if "data_origin" in excluded_rows.columns:
        wrong_origin = excluded_rows["data_origin"].ne("original")
        if wrong_origin.any():
            bad_ids = (
                excluded_rows.loc[wrong_origin, "id"].astype(str).head(10).tolist()
            )
            raise ValueError(
                "Configured exclusions must refer only to original rows: "
                f"{bad_ids}"
            )

    retained_train_indices = np.setdiff1d(
        train_indices, excluded_indices, assume_unique=True
    )
    return retained_train_indices, excluded_indices


def is_better_checkpoint(
    current_metric: float,
    current_loss: float,
    best_metric: float,
    best_loss: float,
) -> bool:
    """Prefer higher Macro-F1, then lower loss; exact ties keep the earlier epoch."""

    if current_metric > best_metric:
        return True
    metric_tied = math.isclose(
        current_metric, best_metric, rel_tol=0.0, abs_tol=1e-12
    )
    return metric_tied and current_loss < best_loss


def write_split_manifest(
    train_df: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_dir: Path,
    excluded_indices: np.ndarray | None = None,
) -> Path:
    """Record the exact holdout used only for epoch/checkpoint selection."""

    assignments = np.full(len(train_df), "selection_excluded", dtype=object)
    assignments[train_indices] = "selection_train"
    assignments[validation_indices] = "selection_validation"
    data = {
        "id": train_df["id"].astype(str).tolist(),
        "split": assignments.tolist(),
    }
    if excluded_indices is not None:
        excluded_mask = np.zeros(len(train_df), dtype=bool)
        excluded_mask[np.asarray(excluded_indices, dtype=np.int64)] = True
        data["excluded_from_full_retrain"] = excluded_mask.tolist()
    for column in ("fold", "data_origin", "source_id", "source_group"):
        if column in train_df.columns:
            data[column] = train_df[column].tolist()
    output = pd.DataFrame(data)
    path = output_dir / "split_manifest.csv"
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def write_excluded_training_rows(
    train_df: pd.DataFrame,
    task: TaskConfig,
    excluded_indices: np.ndarray,
    output_dir: Path,
) -> Path:
    """Export the exact original rows omitted without modifying source CSVs."""

    columns = ["id", "Comment", task.label_column]
    for column in ("fold", "data_origin"):
        if column in train_df.columns:
            columns.append(column)
    output = train_df.iloc[np.asarray(excluded_indices, dtype=np.int64)][columns].copy()
    output["exclusion_reason"] = task.training_exclusion_reason or "configured ID filter"
    path = output_dir / "excluded_training_rows.csv"
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def write_validation_predictions(
    validation_df: pd.DataFrame,
    task: TaskConfig,
    probabilities: np.ndarray,
    output_dir: Path,
) -> Path:
    """Write holdout predictions made by the selected adapter."""

    predicted_ids = probabilities.argmax(axis=1)
    output = pd.DataFrame(
        {
            "id": validation_df["id"].astype(str).tolist(),
            "true_label": validation_df[task.label_column].tolist(),
            "predicted_label": [task.labels[index] for index in predicted_ids],
        }
    )
    for label_index, label in enumerate(task.labels):
        safe_name = label.lower().replace("-", "_").replace(" ", "_")
        output[f"prob_{safe_name}"] = probabilities[:, label_index]
    path = output_dir / "validation_predictions.csv"
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path
