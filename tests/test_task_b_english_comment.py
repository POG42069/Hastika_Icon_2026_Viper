"""English-in-Comment inputs with unchanged raw gold labels matched by ID."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm_config import TASK_B
from src.data import load_epoch_validation_frame, load_task_frames
from src.llm_data import ChatEncoder, CompletionDataset
from tests.test_llm_pipeline import TestChatTokenizer


class EnglishCommentTests(unittest.TestCase):
    def write_inputs(self, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        for fold, filename in enumerate(TASK_B.original_fold_filenames, 1):
            pd.DataFrame({
                "id": [f"original-{fold}-{i}" for i in range(6)],
                "Comment": [f"English training comment {fold}-{i}" for i in range(6)],
                "Hate Category": list(TASK_B.labels),
            }).to_csv(root / filename, index=False)
        development = pd.DataFrame({
            "id": ["001", "002"],
            "Comment": ["English first comment &amp; text", "English second comment"],
        })
        development.to_csv(root / TASK_B.prediction_filename, index=False)
        gold = pd.DataFrame({
            "id": ["002", "001"],
            "Comment": ["RAW Kannada second", "RAW Kannada first"],
            "Hate Category": ["Violence", "Others"],
        })
        gold.to_csv(root / TASK_B.epoch_validation_filename, index=False)
        return development, gold

    def test_new_schema_loads_without_translation_or_augment_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root)
            with (
                patch("src.data.resolve_data_file", side_effect=root.joinpath) as resolver,
                patch("src.data._audit_augmentation_quality") as audit,
            ):
                train, development = load_task_frames(TASK_B)
            self.assertEqual(len(train), 30)
            self.assertEqual(train["fold"].tolist(), [i for i in range(1, 6) for _ in range(6)])
            self.assertEqual(train["data_origin"].unique().tolist(), ["original"])
            self.assertNotIn("text_en", train.columns)
            self.assertEqual(development["id"].tolist(), ["001", "002"])
            self.assertEqual(development["text"].tolist(), [
                "English first comment & text", "English second comment",
            ])
            self.assertEqual(
                {call.args[0] for call in resolver.call_args_list},
                set(TASK_B.original_fold_filenames) | {TASK_B.prediction_filename},
            )
            audit.assert_not_called()

    def test_raw_gold_keeps_labels_order_and_bytes_but_uses_english_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, gold = self.write_inputs(root)
            path = root / TASK_B.epoch_validation_filename
            before = path.read_bytes()
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                _, development = load_task_frames(TASK_B)
                validation = load_epoch_validation_frame(TASK_B, development)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(validation["id"].tolist(), gold["id"].tolist())
            self.assertEqual(validation["Comment"].tolist(), gold["Comment"].tolist())
            self.assertEqual(validation["Hate Category"].tolist(), ["Violence", "Others"])
            self.assertEqual(validation["text"].tolist(), [
                "English second comment", "English first comment & text",
            ])
            tokenizer = TestChatTokenizer()
            encoder = ChatEncoder(tokenizer, TASK_B, 512)
            held_out = CompletionDataset.from_frame(validation, encoder)
            prediction = CompletionDataset.from_frame(development, encoder, labeled=False)
            self.assertEqual(held_out.prompts, list(reversed(prediction.prompts)))
            self.assertNotIn("RAW Kannada", tokenizer.decode(held_out.prompts[0]))
            self.assertEqual(encoder.messages(validation["text"].iloc[0])[1]["content"],
                             "Comment:\nEnglish second comment")
            self.assertIn("hateful English comment", encoder.system_text)
            changed_labels = validation.copy()
            changed_labels["label_id"] = [0, 1]
            self.assertEqual(held_out.prompts,
                             CompletionDataset.from_frame(changed_labels, encoder).prompts)

    def test_validation_rejects_missing_duplicate_or_empty_input(self):
        for failure in ("missing_id", "duplicate_id", "missing_id_value", "empty_comment"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, _ = self.write_inputs(root)
                if failure == "missing_id":
                    inputs.loc[0, "id"] = "999"
                elif failure == "duplicate_id":
                    inputs.loc[0, "id"] = "002"
                elif failure == "missing_id_value":
                    inputs.loc[0, "id"] = None
                else:
                    inputs.loc[0, "Comment"] = "  "
                inputs.to_csv(root / TASK_B.epoch_validation_input_filename, index=False)
                with (
                    patch("src.data.resolve_data_file", side_effect=root.joinpath),
                    self.assertRaises(ValueError),
                ):
                    load_epoch_validation_frame(TASK_B)

    def test_test_prediction_override_does_not_change_validation_text_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root)
            test_input = pd.DataFrame({"id": ["test-1"], "Comment": ["English test"]})
            task = replace(TASK_B, prediction_filename="hastika_multiclass_test.csv")
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                validation = load_epoch_validation_frame(task, test_input)
            self.assertEqual(validation["id"].tolist(), ["002", "001"])
            self.assertEqual(validation["text"].iloc[0], "English second comment")

    def test_long_english_comment_preserves_every_complete_assistant_label(self):
        encoder = ChatEncoder(TestChatTokenizer(), TASK_B, 512)
        prompt, truncated = encoder.encode_prompt("English comment " * 300)
        self.assertTrue(truncated)
        for label in range(len(TASK_B.labels)):
            feature = encoder.feature(prompt, label)
            self.assertLessEqual(len(feature["input_ids"]), 512)
            self.assertEqual(feature["labels"][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(feature["labels"][len(prompt):], encoder.completions[label])


if __name__ == "__main__":
    unittest.main()
