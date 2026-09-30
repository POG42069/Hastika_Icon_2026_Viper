"""Task B reporting uses external gold labels without changing training state."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from llm_config import LLMTrainingConfig, TASK_A, TASK_B
from src.data import load_epoch_validation_frame
from src.llm_data import ChatEncoder, CompletionCollator
from src.llm_training import (
    TaskBValidationMonitor,
    _classification_metrics,
    format_confusion_matrix,
)
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime


class TaskBEpochValidationTests(unittest.TestCase):
    def gold(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"gold-{i}" for i in range(len(TASK_B.labels))],
                "Comment": [f"comment {i}" for i in range(len(TASK_B.labels))],
                TASK_B.label_column: list(TASK_B.labels),
            }
        )

    def test_external_gold_schema_and_labels_are_validated(self):
        for invalid in ("missing_label", "duplicate_id", "unknown_label", "empty"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                gold = self.gold()
                if invalid == "missing_label":
                    gold = gold.drop(columns=TASK_B.label_column)
                elif invalid == "duplicate_id":
                    gold.loc[1, "id"] = gold.loc[0, "id"]
                elif invalid == "unknown_label":
                    gold.loc[0, TASK_B.label_column] = "Invalid"
                else:
                    gold = gold.iloc[:0]
                gold.to_csv(root / TASK_B.epoch_validation_filename, index=False)
                with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                    with self.assertRaises(ValueError):
                        load_epoch_validation_frame(TASK_B)

    def test_monitor_rejects_validation_ids_in_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.gold().to_csv(root / TASK_B.epoch_validation_filename, index=False)
            encoder = ChatEncoder(TestChatTokenizer(), TASK_B, 512)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                with self.assertRaisesRegex(ValueError, "must not occur in training"):
                    TaskBValidationMonitor(
                        TASK_B, LLMTrainingConfig(), cpu_runtime(),
                        pd.DataFrame({"id": ["gold-0"]}), encoder, CompletionCollator(0),
                    )

    def test_epoch_report_preserves_rng_mode_and_writes_full_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.gold().to_csv(root / TASK_B.epoch_validation_filename, index=False)
            config = replace(LLMTrainingConfig(), max_length=512)
            encoder = ChatEncoder(TestChatTokenizer(), TASK_B, 512)
            with (
                patch("src.data.resolve_data_file", side_effect=root.joinpath),
                patch("src.llm_training.resolve_data_file", side_effect=root.joinpath),
            ):
                monitor = TaskBValidationMonitor(
                    TASK_B, config, cpu_runtime(), pd.DataFrame({"id": ["train-1"]}),
                    encoder, CompletionCollator(0),
                )
            probabilities = np.roll(np.eye(len(TASK_B.labels)), 1, axis=1)
            metrics = _classification_metrics(
                TASK_B, np.arange(len(TASK_B.labels)), probabilities, 0.25
            )
            metrics["probabilities"] = probabilities
            model = torch.nn.Linear(1, 1).train()

            def scorer(*args):
                args[0].eval()
                torch.rand(3)
                return metrics

            torch.manual_seed(52)
            state_before = torch.random.get_rng_state().clone()
            with (
                patch("src.llm_training.score_candidates", side_effect=scorer),
                patch("src.llm_training.log") as logged,
            ):
                monitor.evaluate(model, 52, 3, root / "seed_52")
            self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))
            self.assertTrue(model.training)
            lines = [call.args[0] for call in logged.call_args_list]
            self.assertTrue(any("seed=52 | epoch=3: val_loss=" in line for line in lines))
            self.assertTrue(any("macro_f1=" in line for line in lines))
            self.assertTrue(any("rows=actual labels" in line for line in lines))
            for label in TASK_B.labels:
                self.assertTrue(any(line.startswith(f"  {label}: precision=") for line in lines))
            report_dir = root / "seed_52" / "official_validation"
            saved = pd.read_csv(report_dir / "predictions_epoch_03.csv")
            self.assertEqual(saved["id"].tolist(), self.gold()["id"].tolist())
            self.assertEqual(saved["true_label"].tolist(), list(TASK_B.labels))
            self.assertTrue((report_dir / "metrics_epoch_03.json").is_file())
            self.assertTrue((report_dir / "confusion_matrices" / "validation_epoch_03.csv").is_file())

    def test_confusion_matrix_log_orientation(self):
        probabilities = np.roll(np.eye(len(TASK_B.labels)), 1, axis=1)
        metrics = _classification_metrics(
            TASK_B, np.arange(len(TASK_B.labels)), probabilities, 0.25
        )
        rows = format_confusion_matrix(metrics)
        self.assertIn("rows=actual labels", rows[0])
        gender_row = next(row for row in rows if row.startswith("Gender"))
        self.assertEqual(gender_row.split(), ["Gender", "0", "1", "0", "0", "0", "0"])
        self.assertEqual(len(rows), len(TASK_B.labels) + 2)

    def test_task_a_does_not_enable_external_epoch_monitoring(self):
        self.assertIsNone(TASK_A.epoch_validation_filename)
        self.assertEqual(TASK_B.epoch_validation_filename, "hastika_multiclass_validation.csv")


if __name__ == "__main__":
    unittest.main()
