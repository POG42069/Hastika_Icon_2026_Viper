"""Regression coverage for Task B raw code-mixed + English prompts."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm_config import TASK_A, TASK_B
from src.data import load_epoch_validation_frame, load_task_frames
from src.llm_data import ChatEncoder, CompletionDataset
from tests.test_llm_pipeline import TestChatTokenizer

BILINGUAL_TASK_B = TASK_B


class BilingualPromptTests(unittest.TestCase):
    def test_raw_english_and_complete_answer_survive_long_input(self):
        tokenizer = TestChatTokenizer()
        encoder = ChatEncoder(tokenizer, BILINGUAL_TASK_B, 512)
        for raw, english in (
            ("RAW " * 300, "ENGLISH " * 300),
            ("RAW " * 300, "English short"),
            ("Raw short", "ENGLISH " * 300),
        ):
            with self.subTest(raw_length=len(raw), english_length=len(english)):
                prompt, truncated = encoder.encode_prompt(raw, english)
                self.assertTrue(truncated)
                user_text = tokenizer.decode(prompt).split("Comment (raw code-mixed):\n", 1)[1]
                kept_raw, kept_english = user_text.split("\n\nEnglish translation:\n", 1)
                self.assertTrue(kept_raw.strip())
                self.assertTrue(kept_english.split("<end>", 1)[0].strip())
                self.assertTrue(kept_raw.startswith(raw[:3]))
                self.assertTrue(kept_english.startswith(english[:3]))
                for label in range(len(BILINGUAL_TASK_B.labels)):
                    feature = encoder.feature(prompt, label)
                    self.assertLessEqual(len(feature["input_ids"]), 512)
                    self.assertEqual(feature["labels"][:len(prompt)], [-100] * len(prompt))
                    self.assertEqual(feature["labels"][len(prompt):], encoder.completions[label])

    def test_training_and_prediction_use_identical_prompt_without_gold_leakage(self):
        encoder = ChatEncoder(TestChatTokenizer(), BILINGUAL_TASK_B, 512)
        frame = pd.DataFrame({
            "text": ["raw Kannada-English"], "text_en": ["the English text"],
            "label_id": [4],
        })
        training = CompletionDataset.from_frame(frame, encoder)
        prediction = CompletionDataset.from_frame(frame.drop(columns="label_id"), encoder, labeled=False)
        self.assertEqual(training.prompts, prediction.prompts)
        frame["label_id"] = 0
        self.assertEqual(training.prompts, CompletionDataset.from_frame(frame, encoder).prompts)
        with self.assertRaisesRegex(ValueError, "translations are required"):
            CompletionDataset(["raw"], encoder)
        with self.assertRaisesRegex(ValueError, "same length"):
            CompletionDataset(["raw"], encoder, translations=[])

    def test_task_a_prompt_and_configuration_are_unchanged(self):
        self.assertTrue(TASK_A.use_augmentation)
        self.assertIsNone(TASK_A.translation_column)
        encoder = ChatEncoder(TestChatTokenizer(), TASK_A, 512)
        self.assertEqual(encoder.messages("raw")[1]["content"], "Comment:\nraw")
        frame = pd.DataFrame({"text": ["raw"], "label_id": [0], "text_en": ["ignored"]})
        self.assertEqual(
            CompletionDataset.from_frame(frame, encoder).prompts,
            CompletionDataset(["raw"], encoder, [0]).prompts,
        )

    def write_inputs(self, root):
        for fold, name in enumerate(BILINGUAL_TASK_B.original_fold_filenames, 1):
            pd.DataFrame({
                "id": [f"orig-{fold}"], "Comment": [f"raw {fold}"],
                "Comment_en": [f"English {fold}"], "Hate Category": [BILINGUAL_TASK_B.labels[fold-1]],
            }).to_csv(root / name, index=False)
        prediction = pd.DataFrame({
            "id": ["001", "002"], "Comment": ["raw one", "raw two"],
            "Comment_en": ["English one", "English two"],
        })
        prediction.to_csv(root / BILINGUAL_TASK_B.prediction_filename, index=False)
        return prediction

    def test_augment_files_are_neither_resolved_nor_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root)
            with (
                patch("src.data.resolve_data_file", side_effect=root.joinpath) as resolver,
                patch("src.data._audit_augmentation_quality") as audit,
            ):
                train, prediction = load_task_frames(BILINGUAL_TASK_B)
            self.assertEqual(len(train), 5)
            self.assertEqual(train["data_origin"].unique().tolist(), ["original"])
            self.assertEqual(prediction["text_en"].tolist(), ["English one", "English two"])
            self.assertEqual(
                {call.args[0] for call in resolver.call_args_list},
                set(BILINGUAL_TASK_B.original_fold_filenames) | {BILINGUAL_TASK_B.prediction_filename},
            )
            audit.assert_not_called()

    def test_missing_or_blank_translation_fails_for_train_and_prediction(self):
        for filename in (BILINGUAL_TASK_B.original_fold_filenames[0], BILINGUAL_TASK_B.prediction_filename):
            for failure in ("missing_column", "blank"):
                with self.subTest(filename=filename, failure=failure), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.write_inputs(root)
                    frame = pd.read_csv(root / filename, dtype={"id": str})
                    if failure == "missing_column":
                        frame = frame.drop(columns="Comment_en")
                    else:
                        frame.loc[0, "Comment_en"] = "   "
                    frame.to_csv(root / filename, index=False)
                    with (
                        patch("src.data.resolve_data_file", side_effect=root.joinpath),
                        self.assertRaisesRegex(ValueError, "Comment_en|missing English"),
                    ):
                        load_task_frames(BILINGUAL_TASK_B)

    def test_gold_reuses_development_translation_by_id_in_gold_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = self.write_inputs(root)
            gold = prediction.iloc[::-1].drop(columns="Comment_en").copy()
            gold[BILINGUAL_TASK_B.label_column] = ["Violence", "Others"]
            gold.to_csv(root / BILINGUAL_TASK_B.epoch_validation_filename, index=False)
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                validated = load_epoch_validation_frame(BILINGUAL_TASK_B, prediction)
            self.assertEqual(validated["id"].tolist(), ["002", "001"])
            self.assertEqual(validated["text_en"].tolist(), ["English two", "English one"])
            self.assertEqual(validated[BILINGUAL_TASK_B.label_column].tolist(), ["Violence", "Others"])

    def test_gold_misalignment_is_rejected(self):
        for failure in ("wrong_comment", "missing_id", "different_translation"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prediction = self.write_inputs(root)
                gold = prediction.copy()
                gold[BILINGUAL_TASK_B.label_column] = ["Violence", "Others"]
                if failure == "wrong_comment":
                    gold.loc[0, "Comment"] = "unrelated raw text"
                elif failure == "missing_id":
                    gold.loc[0, "id"] = "missing"
                else:
                    gold.loc[0, "Comment_en"] = "unrelated translation"
                gold.to_csv(root / BILINGUAL_TASK_B.epoch_validation_filename, index=False)
                with (
                    patch("src.data.resolve_data_file", side_effect=root.joinpath),
                    self.assertRaisesRegex(ValueError, "differ|missing from translated"),
                ):
                    load_epoch_validation_frame(BILINGUAL_TASK_B, prediction)

    def test_test_prediction_override_keeps_bilingual_development_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = self.write_inputs(root)
            gold = prediction.iloc[::-1].drop(columns="Comment_en").copy()
            gold[BILINGUAL_TASK_B.label_column] = ["Violence", "Others"]
            gold.to_csv(root / BILINGUAL_TASK_B.epoch_validation_filename, index=False)
            task = replace(BILINGUAL_TASK_B, prediction_filename="test.csv")
            test_input = pd.DataFrame({
                "id": ["test-1"], "Comment": ["raw test"], "Comment_en": ["English test"],
            })
            with patch("src.data.resolve_data_file", side_effect=root.joinpath):
                validated = load_epoch_validation_frame(task, test_input)
            self.assertEqual(validated["id"].tolist(), ["002", "001"])
            self.assertEqual(validated["text"].tolist(), ["raw two", "raw one"])
            self.assertEqual(validated["text_en"].tolist(), ["English two", "English one"])

    def test_english_only_dataset_is_rejected_instead_of_used_as_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root)
            path = root / BILINGUAL_TASK_B.original_fold_filenames[0]
            frame = pd.read_csv(path).drop(columns="Comment_en")
            frame.to_csv(path, index=False)
            with (
                patch("src.data.resolve_data_file", side_effect=root.joinpath),
                self.assertRaisesRegex(ValueError, "Comment_en"),
            ):
                load_task_frames(BILINGUAL_TASK_B)


if __name__ == '__main__':
    unittest.main()
