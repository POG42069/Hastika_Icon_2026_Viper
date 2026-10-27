"""Regression coverage for changed private tests and multiple Kaggle inputs."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from llm_config import TASK_A
from src.llm_training import _completed_retrain_from_catalogs, _save_prediction_scores
from src.resume import prepare_resume_catalogs


class TaskAPredictionResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = replace(TASK_A, input_mode="bilingual")
        self.signature = {"training_sha256": "train-unchanged", "max_length": 512}
        self.old_frame = pd.DataFrame({"id": ["dev-1", "dev-2"]})
        self.frame = pd.DataFrame({"id": ["private-1", "private-2", "private-3"]})
        self.hash = "private-test-hash"

    def result_tree(self, name="first", adapter=True, signature=None):
        project = self.root / name / "Hastika_Icon_2026_Viper"
        output = project / "outputs" / self.task.output_subdir / "seed_42"
        checkpoint = project / "checkpoints" / self.task.checkpoint_subdir / "seed_42"
        output.mkdir(parents=True)
        checkpoint.mkdir(parents=True)
        summary = {"seed": 42, "epochs": 2, "resume_signature": self.signature if signature is None else signature}
        (output / "seed_summary.json").write_text(json.dumps({"full_data_retrain": summary}))
        pd.DataFrame({"id": self.old_frame["id"], "label": ["Hate", "Non-Hate"]}).to_csv(output / "predictions.csv", index=False)
        probabilities = np.asarray([[0.1, 0.9], [0.9, 0.1]])
        _save_prediction_scores(output / "prediction_scores.npz", self.old_frame["id"].tolist(), "old-test-hash",
                                {"probabilities": probabilities, "mean_token_log_probs": np.log(probabilities)})
        final = checkpoint / "final_full_data_adapter"
        if adapter:
            final.mkdir()
            (final / "adapter_config.json").write_text("{}")
            (final / "adapter_model.safetensors").write_bytes(b"fixture")
        return output, final

    def find(self):
        catalogs = prepare_resume_catalogs([self.root], self.root / "scratch", self.task.output_subdir, self.task.checkpoint_subdir)
        return _completed_retrain_from_catalogs(catalogs=catalogs, seed=42, expected_epochs=2,
                prediction_frame=self.frame, prediction_sha256=self.hash,
                task=self.task, expected_signature=self.signature)

    def test_new_ids_and_row_count_reuse_adapter(self):
        _, adapter = self.result_tree()
        result = self.find()
        self.assertEqual(result["mode"], "adapter")
        self.assertEqual(result["adapter"], adapter.resolve())
        self.assertIsNone(result["scores"])

    def test_same_ids_changed_text_hash_does_not_reuse_scores(self):
        self.result_tree()
        self.frame = self.old_frame
        self.assertEqual(self.find()["mode"], "adapter")

    def test_reordered_ids_reuse_adapter(self):
        self.result_tree()
        self.frame = self.old_frame.iloc[::-1]
        self.hash = "old-test-hash"
        self.assertEqual(self.find()["mode"], "adapter")

    def test_deleted_or_corrupt_prediction_files_do_not_invalidate_adapter(self):
        output, _ = self.result_tree()
        (output / "predictions.csv").unlink()
        (output / "prediction_scores.npz").write_bytes(b"invalid cache")
        self.assertEqual(self.find()["mode"], "adapter")

    def test_unchanged_input_reuses_scores_even_without_prediction_csv(self):
        output, _ = self.result_tree()
        (output / "predictions.csv").unlink()
        self.frame = self.old_frame
        self.hash = "old-test-hash"
        self.assertEqual(self.find()["mode"], "scores")

    def test_second_input_supplies_adapter_when_first_has_only_reports(self):
        self.result_tree(adapter=False)
        _, adapter = self.result_tree("second")
        self.assertEqual(self.find()["adapter"], adapter.resolve())

    def test_incompatible_adapter_in_first_input_is_not_selected(self):
        self.result_tree(signature={**self.signature, "max_length": 256})
        _, adapter = self.result_tree("second")
        self.assertEqual(self.find()["adapter"], adapter.resolve())

    def test_training_change_rejects_old_adapter(self):
        self.result_tree()
        self.signature = {**self.signature, "training_sha256": "different-train"}
        self.assertIsNone(self.find())

    def test_missing_weights_fail_instead_of_retraining_completed_seed(self):
        _, adapter = self.result_tree()
        (adapter / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "Refusing to retrain"):
            self.find()

    def test_unknown_hash_cannot_reuse_label_only_cache(self):
        output, _ = self.result_tree(adapter=False)
        (output / "prediction_scores.npz").unlink()
        self.frame = self.old_frame
        with self.assertRaisesRegex(RuntimeError, "Refusing to retrain"):
            self.find()

    def test_verified_label_only_cache_can_still_vote(self):
        output, _ = self.result_tree(adapter=False)
        (output / "prediction_scores.npz").unlink()
        value = json.loads((output / "seed_summary.json").read_text())
        value["full_data_retrain"]["prediction_sha256"] = "old-test-hash"
        (output / "seed_summary.json").write_text(json.dumps(value))
        self.frame = self.old_frame
        self.hash = "old-test-hash"
        self.assertEqual(self.find()["mode"], "labels")


if __name__ == "__main__":
    unittest.main()
