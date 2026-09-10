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


def load_task_frames(task: TaskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and normalize train/development CSVs consistently."""

    train_path = resolve_data_file(task.train_filename)
    prediction_path = resolve_data_file(task.prediction_filename)
    train_df = pd.read_csv(train_path, encoding="utf-8-sig", dtype={"id": str})
    prediction_df = pd.read_csv(
        prediction_path, encoding="utf-8-sig", dtype={"id": str}
    )

    required_train = {"id", "Comment", task.label_column}
    required_prediction = {"id", "Comment"}
    missing_train = required_train.difference(train_df.columns)
    missing_prediction = required_prediction.difference(prediction_df.columns)
    if missing_train:
        raise ValueError(f"{train_path} is missing columns: {sorted(missing_train)}")
    if missing_prediction:
        raise ValueError(
            f"{prediction_path} is missing columns: {sorted(missing_prediction)}"
        )
    if train_df["id"].isna().any() or prediction_df["id"].isna().any():
        raise ValueError("Missing IDs were found; every row needs an ID.")
    if train_df["id"].duplicated().any() or prediction_df["id"].duplicated().any():
        raise ValueError("Duplicate IDs were found; each submission ID must be unique.")

    unknown_labels = sorted(
        set(train_df[task.label_column].dropna()) - set(task.labels)
    )
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {train_path}: {unknown_labels}")
    if train_df[task.label_column].isna().any():
        raise ValueError(f"Missing labels were found in {train_path}.")

    train_df = train_df.copy()
    prediction_df = prediction_df.copy()
    train_df["text"] = train_df["Comment"].map(normalize_text)
    prediction_df["text"] = prediction_df["Comment"].map(normalize_text)
    label_to_id = {label: index for index, label in enumerate(task.labels)}
    train_df["label_id"] = train_df[task.label_column].map(label_to_id).astype(int)
    return train_df, prediction_df
