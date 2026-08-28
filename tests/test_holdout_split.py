"""Tests for the deterministic stratified 80/20 holdout."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import TASK_A
from src.training import (
    is_better_checkpoint,
    stratified_holdout_indices,
    write_split_manifest,
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

    def test_checkpoint_tie_breaking(self) -> None:
        """Macro-F1 wins first, lower loss breaks ties, exact ties keep early epoch."""

        self.assertTrue(is_better_checkpoint(0.6, 0.8, 0.5, 0.4))
        self.assertTrue(is_better_checkpoint(0.6, 0.3, 0.6, 0.4))
        self.assertFalse(is_better_checkpoint(0.6, 0.4, 0.6, 0.4))
        self.assertFalse(is_better_checkpoint(0.5, 0.1, 0.6, 0.4))

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

            manifest = pd.read_csv(manifest_path, dtype={"id": str})
            diagnostics = pd.read_csv(validation_path, dtype={"id": str})
            self.assertEqual(manifest["id"].tolist(), frame["id"].tolist())
            self.assertEqual(
                manifest["split"].tolist(),
                ["train", "train", "validation", "validation"],
            )
            self.assertEqual(diagnostics["id"].tolist(), ["003", "004"])
            self.assertEqual(
                diagnostics["predicted_label"].tolist(), ["Non-Hate", "Hate"]
            )


if __name__ == "__main__":
    unittest.main()
