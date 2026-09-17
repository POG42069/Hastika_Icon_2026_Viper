"""Tests for the deterministic stratified 80/20 holdout."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from llm_config import TASK_A
from src.training import (
    cap_training_labels,
    is_better_checkpoint,
    predefined_fold_indices,
    stratified_holdout_indices,
    write_downsampled_rows,
    write_seed_selection_results,
    write_split_manifest,
    write_validation_confusion_history,
    write_validation_confusion_matrix,
    write_validation_predictions,
)


class HoldoutSplitTests(unittest.TestCase):
    """Verify official task sizes, stratification and reproducibility."""

    def _assert_valid_split(
        self,
        labels: np.ndarray,
        expected_train: int,
        expected_validation: int,
    ) -> None:
        train_indices, validation_indices = stratified_holdout_indices(
            labels, validation_size=0.20, seed=42
        )
        repeated_train, repeated_validation = stratified_holdout_indices(
            labels, validation_size=0.20, seed=42
        )

        self.assertEqual(len(train_indices), expected_train)
        self.assertEqual(len(validation_indices), expected_validation)
        self.assertTrue(np.array_equal(train_indices, repeated_train))
        self.assertTrue(np.array_equal(validation_indices, repeated_validation))
        self.assertFalse(set(train_indices).intersection(validation_indices))
        self.assertEqual(
            set(train_indices).union(validation_indices), set(range(len(labels)))
        )
        self.assertEqual(set(np.unique(labels[train_indices])), set(np.unique(labels)))
        self.assertEqual(
            set(np.unique(labels[validation_indices])), set(np.unique(labels))
        )

    def test_task_a_split_sizes(self) -> None:
        """6,446 binary labels must split into 5,156 train and 1,290 validation."""

        labels = np.asarray([0] * 3160 + [1] * 3286, dtype=np.int64)
        self._assert_valid_split(labels, expected_train=5156, expected_validation=1290)

    def test_task_b_split_sizes(self) -> None:
        """3,159 multi-class labels must split into 2,527 train and 632 validation."""

        labels = np.concatenate(
            [
                np.full(1362, 0),
                np.full(559, 1),
                np.full(382, 2),
                np.full(186, 3),
                np.full(221, 4),
                np.full(449, 5),
            ]
        ).astype(np.int64)
        self._assert_valid_split(labels, expected_train=2527, expected_validation=632)

    def test_predefined_fold_five_is_the_complete_validation_set(self) -> None:
        folds = np.asarray([1, 1, 2, 3, 4, 5, 5], dtype=np.int64)
        train, validation = predefined_fold_indices(folds, validation_fold=5)
        self.assertEqual(train.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(validation.tolist(), [5, 6])
        self.assertFalse(set(train).intersection(validation))

    def test_predefined_validation_can_use_only_original_rows(self) -> None:
        folds = np.asarray([1, 1, 4, 5, 5], dtype=np.int64)
        origins = np.asarray(
            [
                "original",
                "augmentation",
                "augmentation",
                "original",
                "augmentation",
            ]
        )
        train, validation = predefined_fold_indices(
            folds,
            validation_fold=5,
            data_origins=origins,
            validation_origins=("original",),
        )
        self.assertEqual(train.tolist(), [0, 1, 2])
        self.assertEqual(validation.tolist(), [3])
        self.assertNotIn(4, train)
        self.assertNotIn(4, validation)

    def test_checkpoint_tie_breaking(self) -> None:
        """Macro-F1 wins first, lower loss breaks ties, exact ties keep early epoch."""

        self.assertTrue(is_better_checkpoint(0.6, 0.8, 0.5, 0.4))
        self.assertTrue(is_better_checkpoint(0.6, 0.3, 0.6, 0.4))
        self.assertFalse(is_better_checkpoint(0.6, 0.4, 0.6, 0.4))
        self.assertFalse(is_better_checkpoint(0.5, 0.1, 0.6, 0.4))

    def test_training_label_cap_is_exact_and_reproducible(self) -> None:
        labels = np.asarray(["Gender"] * 10 + ["Violence"] * 4)
        indices = np.arange(len(labels), dtype=np.int64)
        retained, dropped, summary = cap_training_labels(
            indices, labels, (("Gender", 6),), seed=42
        )
        repeated_retained, repeated_dropped, _ = cap_training_labels(
            indices, labels, (("Gender", 6),), seed=42
        )
        other_retained, other_dropped, _ = cap_training_labels(
            indices, labels, (("Gender", 6),), seed=43
        )

        self.assertEqual((labels[retained] == "Gender").sum(), 6)
        self.assertEqual((labels[retained] == "Violence").sum(), 4)
        self.assertEqual(len(dropped), 4)
        self.assertTrue(np.array_equal(retained, repeated_retained))
        self.assertTrue(np.array_equal(dropped, repeated_dropped))
        self.assertFalse(np.array_equal(retained, other_retained))
        self.assertFalse(np.array_equal(dropped, other_dropped))
        self.assertEqual(summary[0]["available"], 10)
        self.assertEqual(summary[0]["retained"], 6)

    def test_holdout_diagnostic_artifacts(self) -> None:
        """Manifest and validation diagnostics must preserve IDs and labels."""

        frame = pd.DataFrame(
            {
                "id": ["001", "002", "003", "004"],
                "Comment": ["a", "b", "c", "d"],
                "Label": ["Non-Hate", "Hate", "Non-Hate", "Hate"],
            }
        )
        train_indices = np.asarray([0, 1], dtype=np.int64)
        validation_indices = np.asarray([2, 3], dtype=np.int64)
        validation_probabilities = np.asarray(
            [[0.8, 0.2], [0.1, 0.9]], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            manifest_path = write_split_manifest(
                frame, train_indices, validation_indices, output_dir
            )
            validation_path = write_validation_predictions(
                frame.iloc[validation_indices],
                TASK_A,
                validation_probabilities,
                output_dir,
            )
            downsampled_path = write_downsampled_rows(
                frame,
                TASK_A,
                np.asarray([1], dtype=np.int64),
                output_dir,
            )
            confusion = {
                "label_order": list(TASK_A.labels),
                "values": [[1, 1], [0, 2]],
            }
            confusion_path = write_validation_confusion_matrix(
                confusion, 1, output_dir
            )
            confusion_history_path = write_validation_confusion_history(
                [{"epoch": 1, "confusion_matrix": confusion}], output_dir
            )
            seed_results_path = write_seed_selection_results(
                [
                    {
                        "seed": 42,
                        "best_epoch": 3,
                        "macro_f1": 0.61,
                        "loss": 0.4,
                        "accuracy": 0.7,
                        "selection_train_rows": 100,
                        "full_retrain_rows": 120,
                        "downsampled_rows": 20,
                        "checkpoint": "seed_42/best_holdout_adapter",
                        "selected": True,
                    }
                ],
                output_dir,
            )

            manifest = pd.read_csv(manifest_path, dtype={"id": str})
            diagnostics = pd.read_csv(validation_path, dtype={"id": str})
            downsampled = pd.read_csv(downsampled_path, dtype={"id": str})
            confusion_table = pd.read_csv(confusion_path)
            confusion_history = pd.read_csv(confusion_history_path)
            seed_results = pd.read_csv(seed_results_path)
            self.assertEqual(manifest["id"].tolist(), frame["id"].tolist())
            self.assertEqual(
                manifest["split"].tolist(),
                [
                    "selection_train",
                    "selection_train",
                    "selection_validation",
                    "selection_validation",
                ],
            )
            self.assertEqual(diagnostics["id"].tolist(), ["003", "004"])
            self.assertEqual(
                diagnostics["predicted_label"].tolist(), ["Non-Hate", "Hate"]
            )
            self.assertEqual(downsampled["id"].tolist(), ["002"])
            self.assertEqual(downsampled["Comment"].tolist(), ["b"])
            self.assertEqual(downsampled["Label"].tolist(), ["Hate"])
            self.assertEqual(
                downsampled["excluded_from"].tolist(),
                ["selection_train_and_final_retrain"],
            )
            self.assertEqual(
                confusion_table.columns.tolist(),
                ["true_label", "Non-Hate", "Hate"],
            )
            self.assertEqual(confusion_table["true_label"].tolist(), list(TASK_A.labels))
            self.assertEqual(
                confusion_table[["Non-Hate", "Hate"]].values.tolist(),
                [[1, 1], [0, 2]],
            )
            self.assertEqual(len(confusion_history), len(TASK_A.labels) ** 2)
            self.assertEqual(confusion_history["count"].sum(), 4)
            self.assertEqual(seed_results["seed"].tolist(), [42])
            self.assertEqual(seed_results["selected"].tolist(), [True])


if __name__ == "__main__":
    unittest.main()
