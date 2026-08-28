"""Create and validate organizer-compatible HASTIKA submissions."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def write_submission(
    ids: Sequence[str],
    labels: Sequence[str],
    allowed_labels: Sequence[str],
    output_dir: Path,
    zip_name: str,
) -> tuple[Path, Path]:
    """Write ``predictions.csv`` and a ZIP containing only that file.

    The function intentionally validates the exact two-column header, row
    count, identifier order, and label vocabulary before creating the archive.
    """

    string_ids = [str(value) for value in ids]
    string_labels = [str(value) for value in labels]
    if len(string_ids) != len(string_labels):
        raise ValueError("Every input id must have exactly one predicted label.")
    if len(set(string_ids)) != len(string_ids):
        raise ValueError("Submission ids must be unique.")

    invalid_labels = sorted(set(string_labels) - set(allowed_labels))
    if invalid_labels:
        raise ValueError(f"Submission contains invalid labels: {invalid_labels}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    submission_df = pd.DataFrame({"id": string_ids, "label": string_labels})
    submission_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    reloaded = pd.read_csv(predictions_path, dtype={"id": str}, encoding="utf-8-sig")
    if list(reloaded.columns) != ["id", "label"]:
        raise RuntimeError("Submission header must be exactly: id,label")
    if reloaded["id"].tolist() != string_ids:
        raise RuntimeError("Submission ids changed order while writing the CSV.")

    zip_path = output_dir / zip_name
    with zipfile.ZipFile(
        zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(predictions_path, arcname="predictions.csv")
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if archive.namelist() != ["predictions.csv"]:
            raise RuntimeError(
                "Submission ZIP must contain only predictions.csv at its root."
            )
    return predictions_path, zip_path
