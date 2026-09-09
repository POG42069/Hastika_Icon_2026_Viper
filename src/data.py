"""Data loading, validation, preprocessing, and PyTorch datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from config import DATA_DIR, PreprocessConfig, TaskConfig
from src.preprocessing import normalize_text


def resolve_data_file(filename: str, configured_dir: Path = DATA_DIR) -> Path:
    """Find a data file locally or inside an attached Kaggle Dataset.

    Resolution order:
    1. ``HASTIKA_DATA_DIR`` / configured data directory.
    2. The repository's local ``data`` directory.
    3. A recursive search under ``/kaggle/input``.
    """

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


def load_task_frames(
    task: TaskConfig,
    preprocess: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and preprocess one task's train and prediction files."""

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
    if (
        train_df["id"].duplicated().any()
        or prediction_df["id"].duplicated().any()
    ):
        raise ValueError("Duplicate ids were found; each submission id must be unique.")

    unknown_labels = sorted(
        set(train_df[task.label_column].dropna()) - set(task.labels)
    )
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {train_path}: {unknown_labels}")
    if train_df[task.label_column].isna().any():
        raise ValueError(f"Missing labels were found in {train_path}.")

    train_df = train_df.copy()
    prediction_df = prediction_df.copy()
    train_df["clean_text"] = train_df["Comment"].map(
        lambda value: normalize_text(value, preprocess)
    )
    prediction_df["clean_text"] = prediction_df["Comment"].map(
        lambda value: normalize_text(value, preprocess)
    )
    label_to_id = {label: index for index, label in enumerate(task.labels)}
    train_df["label_id"] = train_df[task.label_column].map(label_to_id).astype(int)
    return train_df, prediction_df


class EncodedTextDataset(Dataset):
    """Tokenize comments lazily so the dataset stays easy to inspect."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: object,
        max_length: int,
        labels: Sequence[int] | None = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.labels is not None and len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

    def __len__(self) -> int:
        """Return the number of comments."""

        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Tokenize one comment and retain its original dataset index."""

        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["sample_index"] = index
        if self.labels is not None:
            encoded["labels"] = int(self.labels[index])
        return encoded


class ClassificationCollator:
    """Dynamically pad token fields while preserving labels and row indices."""

    def __init__(self, tokenizer: object, pad_to_multiple_of: int | None = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        """Convert a list of token dictionaries into one padded tensor batch."""

        sample_indices = [int(feature.pop("sample_index")) for feature in features]
        contains_labels = "labels" in features[0]
        labels = (
            [int(feature.pop("labels")) for feature in features]
            if contains_labels
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["sample_index"] = torch.tensor(sample_indices, dtype=torch.long)
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class DistributedSliceSampler(Sampler[int]):
    """Partition evaluation data without padding or duplicate samples."""

    def __init__(self, dataset_size: int, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, dataset_size, world_size))

    def __iter__(self) -> Iterator[int]:
        """Yield the unique indices assigned to this worker."""

        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of indices assigned to this worker."""

        return len(self.indices)
