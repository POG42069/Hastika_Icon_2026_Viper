"""Fast tests for raw input loading and submission formatting."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm_config import TASK_A
from src.data import load_task_frames
from src.submission import write_submission


class RawDataTests(unittest.TestCase):
    """Verify comments reach tokenization without normalization or cleaning."""

    def test_raw_comments_are_preserved_and_missing_cells_become_empty(self) -> None:
        raw = "HeLLo #Kannada!!! @person https://example.com <br> 😊 CARS"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "id": ["001", "002"],
                    "Comment": [raw, None],
                    "Label": ["Non-Hate", "Hate"],
                }
            ).to_csv(root / TASK_A.train_filename, index=False)
            pd.DataFrame(
                {"id": ["003"], "Comment": ["Soooo GOOD!!!!!"]}
            ).to_csv(root / TASK_A.prediction_filename, index=False)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                train, development = load_task_frames(TASK_A)
        self.assertEqual(train["text"].tolist(), [raw, ""])
        self.assertEqual(development["text"].tolist(), ["Soooo GOOD!!!!!"])


class SubmissionTests(unittest.TestCase):
    """Verify exact CSV and ZIP structure required by the organizer."""

    def test_submission_contains_only_predictions_csv(self) -> None:
        """The ZIP root must contain one id,label CSV in original id order."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path, zip_path = write_submission(
                ids=["001", "958"],
                labels=["Non-Hate", "Hate"],
                allowed_labels=("Non-Hate", "Hate"),
                output_dir=output_dir,
                zip_name="task_a_submission.zip",
            )
            frame = pd.read_csv(csv_path, dtype={"id": str})
            self.assertEqual(frame.columns.tolist(), ["id", "label"])
            self.assertEqual(frame["id"].tolist(), ["001", "958"])
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["predictions.csv"])


if __name__ == "__main__":
    unittest.main()
