"""HASTIKA CSV loading, validation, and light text normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_config import DATA_DIR, TaskConfig
from src.preprocessing import normalize_text


def resolve_data_file(filename: str, configured_dir: Path = DATA_DIR) -> Path:
    """Find an input locally or in an attached Kaggle Dataset."""

    candidates = [
        configured_dir / filename,
        Path(__file__).resolve().parents[1] / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        matches = sorted(kaggle_root.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            rendered = "\n  - ".join(str(path) for path in matches)
            raise FileNotFoundError(
                f"Found multiple copies of {filename}. Set HASTIKA_DATA_DIR to "
                f"the intended directory:\n  - {rendered}"
            )

    raise FileNotFoundError(
        f"Could not find {filename}. Attach the private HASTIKA dataset on "
        "Kaggle or set the HASTIKA_DATA_DIR environment variable."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str, "source_id": str, "source_group": str},
    )


def _validate_columns(
    frame: pd.DataFrame,
    path: Path,
    required: set[str],
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _validate_and_encode_frames(
    train_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    task: TaskConfig,
    train_description: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_train = {"id", "Comment", task.label_column}
    required_prediction = {"id", "Comment"}
    _validate_columns(train_df, Path(train_description), required_train)
    _validate_columns(
        prediction_df,
        Path(task.prediction_filename),
        required_prediction,
    )
    if train_df["id"].isna().any() or prediction_df["id"].isna().any():
        raise ValueError("Missing IDs were found; every row needs an ID.")
    if train_df["id"].duplicated().any():
        duplicates = train_df.loc[train_df["id"].duplicated(False), "id"].unique()
        raise ValueError(
            "Duplicate labeled IDs were found across the Task B fold files: "
            f"{duplicates[:10].tolist()}"
        )
    if prediction_df["id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs were found; each ID must be unique.")

    unknown_labels = sorted(
        set(train_df[task.label_column].dropna()) - set(task.labels)
    )
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {train_description}: {unknown_labels}")
    if train_df[task.label_column].isna().any():
        raise ValueError(f"Missing labels were found in {train_description}.")

    train_df = train_df.copy()
    prediction_df = prediction_df.copy()
    train_df["text"] = train_df["Comment"].map(normalize_text)
    prediction_df["text"] = prediction_df["Comment"].map(normalize_text)
    label_to_id = {label: index for index, label in enumerate(task.labels)}
    train_df["label_id"] = train_df[task.label_column].map(label_to_id).astype(int)
    return train_df, prediction_df


def _load_predefined_folds(task: TaskConfig) -> pd.DataFrame:
    if task.use_augmentation and len(task.original_fold_filenames) != len(
        task.augmentation_fold_filenames
    ):
        raise ValueError(
            "original_fold_filenames and augmentation_fold_filenames must have "
            "the same number of files."
        )
    if not task.original_fold_filenames:
        raise ValueError("At least one predefined fold is required.")
    if task.selection_validation_fold not in range(
        1, len(task.original_fold_filenames) + 1
    ):
        raise ValueError("selection_validation_fold does not identify a configured fold.")

    fold_frames: list[pd.DataFrame] = []
    original_by_fold: dict[int, pd.DataFrame] = {}
    augment_by_fold: dict[int, pd.DataFrame] = {}
    for fold, original_name in enumerate(task.original_fold_filenames, start=1):
        original_path = resolve_data_file(original_name)
        original = _read_csv(original_path)
        _validate_columns(
            original,
            original_path,
            {"id", "Comment", task.label_column},
        )
        original = original.copy()
        original["fold"] = fold
        original["data_origin"] = "original"
        original_by_fold[fold] = original
        fold_frames.append(original)

        if not task.use_augmentation:
            continue

        augment_path = resolve_data_file(task.augmentation_fold_filenames[fold - 1])
        augment = _read_csv(augment_path)
        _validate_columns(
            augment,
            augment_path,
            {
                "id",
                "Comment",
                task.label_column,
                "source_id",
                "source_group",
                "fold",
            },
        )
        if augment["source_id"].isna().any() or augment["source_group"].isna().any():
            raise ValueError(f"{augment_path} contains missing augmentation lineage.")
        parsed_folds = pd.to_numeric(augment["fold"], errors="coerce")
        if parsed_folds.isna().any() or not parsed_folds.eq(fold).all():
            raise ValueError(
                f"{augment_path} must contain only rows whose fold value is {fold}."
            )

        augment = augment.copy()
        augment["fold"] = fold
        augment["data_origin"] = "augmentation"
        augment_by_fold[fold] = augment
        fold_frames.append(augment)

    originals = pd.concat(original_by_fold.values(), ignore_index=True, sort=False)
    if not task.use_augmentation:
        return originals

    original_ids = set(originals["id"])
    for fold, augment in augment_by_fold.items():
        missing_sources = sorted(set(augment["source_id"]) - original_ids)
        if missing_sources:
            raise ValueError(
                f"augment fold {fold} refers to unknown source IDs: "
                f"{missing_sources[:10]}"
            )
        source_folds = originals.set_index("id")["fold"]
        wrong_fold = augment["source_id"].map(source_folds).ne(fold)
        if wrong_fold.any():
            bad_ids = augment.loc[wrong_fold, "id"].head(10).tolist()
            raise ValueError(
                f"augment fold {fold} contains sources from another fold: {bad_ids}"
            )
        source_labels = originals.set_index("id")[task.label_column]
        wrong_label = augment["source_id"].map(source_labels).ne(
            augment[task.label_column]
        )
        if wrong_label.any():
            bad_ids = augment.loc[wrong_label, "id"].head(10).tolist()
            raise ValueError(
                f"augment fold {fold} changed its source label: {bad_ids}"
            )

    all_augments = pd.concat(augment_by_fold.values(), ignore_index=True, sort=False)
    groups_per_fold = all_augments.groupby("source_group")["fold"].nunique()
    crossing_groups = groups_per_fold[groups_per_fold > 1]
    if not crossing_groups.empty:
        raise ValueError(
            "Augmentation source groups cross predefined folds: "
            f"{crossing_groups.index[:10].tolist()}"
        )
    return pd.concat(fold_frames, ignore_index=True, sort=False)


def load_task_frames(task: TaskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and normalize train/development CSVs consistently."""

    prediction_path = resolve_data_file(task.prediction_filename)
    prediction_df = _read_csv(prediction_path)
    if task.original_fold_filenames:
        train_df = _load_predefined_folds(task)
        train_description = (
            "predefined original and augmentation folds"
            if task.use_augmentation
            else "predefined original folds"
        )
    else:
        train_path = resolve_data_file(task.train_filename)
        train_df = _read_csv(train_path)
        train_description = str(train_path)
    return _validate_and_encode_frames(
        train_df,
        prediction_df,
        task,
        train_description,
    )
