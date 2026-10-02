"""Fast tests for light text normalization and submission formatting."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm_config import TASK_A, TASK_B
from src.data import labeled_input_filenames, load_task_frames
from src.submission import write_submission


class TextNormalizationTests(unittest.TestCase):
    """Verify technical cleanup does not discard authored content."""

    def test_urls_mentions_html_and_stylistic_signals_are_preserved(self) -> None:
        raw = (
            "  FranÃ§ais  ＣＡＲＳ &amp; <b>@person "
            "https://example.com</b>\n#Kannada!!! 😊 soooo  "
        )
        normalized = (
            "Français CARS & <b>@person https://example.com</b> "
            "#Kannada!!! 😊 soooo"
        )
        task = replace(
            TASK_A,
            original_fold_filenames=(),
            augmentation_fold_filenames=(),
            use_augmentation=False,
            selection_validation_fold=None,
            selection_validation_data_origins=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "id": ["001", "002"],
                    "Comment": [raw, None],
                    "Label": ["Non-Hate", "Hate"],
                }
            ).to_csv(root / task.train_filename, index=False)
            pd.DataFrame(
                {"id": ["003"], "Comment": ["Soooo GOOD!!!!!"]}
            ).to_csv(root / task.prediction_filename, index=False)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                train, development = load_task_frames(task)
        self.assertEqual(train["text"].tolist(), [normalized, ""])
        self.assertEqual(development["text"].tolist(), ["Soooo GOOD!!!!!"])


class CompactEnglishDatasetTests(unittest.TestCase):
    def test_both_tasks_load_compact_originals_without_augmentation(self) -> None:
        for task in (TASK_A, TASK_B):
            with self.subTest(task=task.task_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pd.DataFrame(
                    {
                        "id": ["001", "002"],
                        "Comment": ["An English comment", "Another comment"],
                        task.label_column: [task.labels[0], task.labels[1]],
                    }
                ).to_csv(root / task.compact_train_filename, index=False)
                pd.DataFrame(
                    {
                        "id": ["003"],
                        "Comment": ["Held-out original"],
                        task.label_column: [task.labels[0]],
                    }
                ).to_csv(root / task.compact_validation_filename, index=False)
                pd.DataFrame(
                    {"id": ["004"], "Comment": ["Predict this"]}
                ).to_csv(root / task.prediction_filename, index=False)
                with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                    train, prediction = load_task_frames(task)
                    selected_files = labeled_input_filenames(task)
                self.assertFalse(task.use_augmentation)
                self.assertEqual(train["id"].tolist(), ["001", "002", "003"])
                self.assertEqual(train["fold"].tolist(), [1, 1, 5])
                self.assertEqual(train["data_origin"].tolist(), ["original"] * 3)
                self.assertEqual(prediction["id"].tolist(), ["004"])
                self.assertEqual(
                    selected_files,
                    (task.compact_train_filename, task.compact_validation_filename),
                )

    def test_one_missing_compact_file_fails_instead_of_mixing_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {"id": ["001"], "Comment": ["English"], "Label": ["Hate"]}
            ).to_csv(root / TASK_A.compact_train_filename, index=False)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                    labeled_input_filenames(TASK_A)

    def test_legacy_original_folds_remain_available_without_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold, filename in enumerate(TASK_A.original_fold_filenames, start=1):
                pd.DataFrame(
                    {
                        "id": [str(fold)],
                        "Comment": [f"Original {fold}"],
                        "Label": ["Hate"],
                    }
                ).to_csv(root / filename, index=False)
            pd.DataFrame(
                {"id": ["006"], "Comment": ["Predict"]}
            ).to_csv(root / TASK_A.prediction_filename, index=False)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                train, _ = load_task_frames(TASK_A)
                selected_files = labeled_input_filenames(TASK_A)
            self.assertEqual(train["fold"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(selected_files, TASK_A.original_fold_filenames)


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
