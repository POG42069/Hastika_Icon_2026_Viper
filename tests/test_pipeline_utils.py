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
from src.data import load_task_frames
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


class OriginalOnlyDataTests(unittest.TestCase):
    """Ensure both branch defaults exclude augment files even when present."""

    def test_both_tasks_load_only_original_folds(self) -> None:
        for task in (TASK_A, TASK_B):
            for augment_present in (False, True):
                with (
                    self.subTest(task=task.task_name, augment_present=augment_present),
                    tempfile.TemporaryDirectory() as temporary_directory,
                ):
                    root = Path(temporary_directory)
                    self.assertFalse(task.use_augmentation)
                    self.assertEqual(
                        task.labeled_filenames(), task.original_fold_filenames
                    )
                    expected_ids = []
                    for fold, original_name in enumerate(
                        task.original_fold_filenames, start=1
                    ):
                        ids = [f"original-{fold}-{i}" for i in range(len(task.labels))]
                        expected_ids.extend(ids)
                        pd.DataFrame(
                            {
                                "id": ids,
                                "Comment": [f"original {value}" for value in ids],
                                task.label_column: list(task.labels),
                            }
                        ).to_csv(root / original_name, index=False)
                    if augment_present:
                        # These invalid CSVs would fail if the loader tried reading them.
                        for name in task.augmentation_fold_filenames:
                            (root / name).write_text("invalid_augment\nignored\n")
                    pd.DataFrame(
                        {"id": ["development-1"], "Comment": ["development comment"]}
                    ).to_csv(root / task.prediction_filename, index=False)
                    with (
                        patch("src.data.resolve_data_file", side_effect=root.joinpath)
                        as resolver,
                        patch("src.data._audit_augmentation_quality") as audit,
                    ):
                        train, development = load_task_frames(task)
                    self.assertEqual(train["id"].tolist(), expected_ids)
                    self.assertEqual(set(train["data_origin"]), {"original"})
                    self.assertEqual(len(development), 1)
                    self.assertEqual(
                        {call.args[0] for call in resolver.call_args_list},
                        set(task.original_fold_filenames) | {task.prediction_filename},
                    )
                    audit.assert_not_called()


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
