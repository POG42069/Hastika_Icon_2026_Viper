"""Regressions for selecting the labeled paired open set, not the raw test."""
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from llm_config import (
    TASK_A, TASK_B, LLM_TRAINING_B, PROJECT_ROOT, TASK_B_OPEN_VALIDATION_FILENAME,
)
from src.data import (
    MissingDataFileError, load_task_frames, load_epoch_validation_frame,
    resolve_data_file, prompt_inputs,
)
from src.llm_data import ChatEncoder
from src.llm_training import run_llm_task, run_llm_cli
from tests.test_llm_pipeline import TestChatTokenizer
from tests import test_task_b_khangtest as fixtures


class TaskBOpenValidationTests(unittest.TestCase):
    def test_default_prediction_and_gold_share_the_paired_open_set(self):
        for field in ('prediction_filename', 'epoch_validation_filename',
                      'epoch_validation_input_filename'):
            self.assertEqual(getattr(TASK_B, field), TASK_B_OPEN_VALIDATION_FILENAME)
        self.assertTrue(TASK_B.epoch_validation_required)

    def test_bundled_open_set_has_395_unique_complete_labeled_pairs(self):
        path = resolve_data_file(TASK_B_OPEN_VALIDATION_FILENAME)
        self.assertEqual(path, PROJECT_ROOT / 'data' / 'open_validation' / TASK_B_OPEN_VALIDATION_FILENAME)
        frame = pd.read_csv(path, dtype={'id': str}, keep_default_na=False)
        self.assertEqual(len(frame), 395)
        self.assertEqual(set(frame.columns), {'id', 'Comment', 'OriginalComment', 'Hate Category'})
        self.assertFalse(frame['id'].duplicated().any())
        for column in ('id', 'Comment', 'OriginalComment'):
            self.assertFalse(frame[column].str.strip().eq('').any())
        self.assertEqual(set(frame['Hate Category']), set(TASK_B.labels))

    def test_explicit_training_directory_does_not_select_its_raw_legacy_test(self):
        for include in (False, True):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = replace(TASK_B, include_english_translation=include)
                fixtures.TaskBKhangtestTests().write_data(root, task, gold=True)
                pd.DataFrame({'id': ['6820'], 'Comment': ['raw legacy test']}).to_csv(
                    root / 'multiclass_validation_inputs.csv', index=False)
                resolver = partial(resolve_data_file, configured_dir=root)
                with patch.dict(os.environ, {'HASTIKA_DATA_DIR': directory}), patch(
                    'src.data.resolve_data_file', side_effect=resolver
                ):
                    training, prediction = load_task_frames(task)
                    gold = load_epoch_validation_frame(task)
                self.assertEqual(len(training), 30)
                self.assertEqual(len(prediction), 395)
                self.assertEqual(prediction['id'].tolist(), gold['id'].tolist())
                self.assertNotIn('6820', prediction['id'].tolist())
                self.assertFalse(set(training['id']) & set(prediction['id']))
                pair = prompt_inputs(prediction, task)[0]
                body = ChatEncoder(TestChatTokenizer(), task, 700).messages(pair)[1]['content']
                self.assertIn(pair[1], body)
                self.assertEqual('English translation:' in body, include)

    def test_explicit_prediction_override_keeps_the_open_set_for_gold_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.TaskBKhangtestTests().write_data(root, TASK_B, gold=True)
            test_path = root / 'paired_private_test.csv'
            pd.DataFrame({'id': ['private-1'], 'Comment': ['translated test'],
                          'OriginalComment': ['raw test']}).to_csv(test_path, index=False)
            task = replace(TASK_B, prediction_filename=str(test_path))
            with patch('src.data.resolve_data_file', side_effect=partial(resolve_data_file, configured_dir=root)):
                _, prediction = load_task_frames(task)
                gold = load_epoch_validation_frame(task)
            self.assertEqual(prediction['id'].tolist(), ['private-1'])
            self.assertEqual(len(gold), 395)
            self.assertNotIn('private-1', gold['id'].tolist())

    def test_missing_bundled_file_fails_without_falling_back_to_raw_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'id': ['6820'], 'Comment': ['raw test']}).to_csv(
                root / 'multiclass_validation_inputs.csv', index=False)
            with patch('src.data.PROJECT_ROOT', root):
                with self.assertRaisesRegex(MissingDataFileError, 'Bundled Task B open validation is missing'):
                    resolve_data_file(TASK_B_OPEN_VALIDATION_FILENAME, root)

    def test_cli_prediction_override_keeps_the_bundled_gold_reference(self):
        with patch('sys.argv', ['Train_B_LLM.py', '--prediction-file', '/custom/test.csv']), patch(
            'src.llm_training.relaunch_with_torchrun_if_needed'
        ), patch('src.llm_training.run_llm_task') as runner:
            run_llm_cli(TASK_B, LLM_TRAINING_B)
        task = runner.call_args.args[0]
        self.assertEqual(task.prediction_filename, '/custom/test.csv')
        self.assertEqual(task.epoch_validation_filename, TASK_B_OPEN_VALIDATION_FILENAME)
        self.assertEqual(task.epoch_validation_input_filename, TASK_B_OPEN_VALIDATION_FILENAME)

    def test_both_branch_modes_complete_using_one_paired_file_for_prediction_and_gold(self):
        for include in (False, True):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                task = replace(TASK_B, include_english_translation=include, selection_seeds=(42, 52))
                config = replace(LLM_TRAINING_B, max_length=512, max_epochs=1)
                helper = fixtures.TaskBKhangtestTests()
                helper.write_data(root, task, gold=True)
                helper.boundary(stack, root)
                pd.DataFrame({'id': ['6820'], 'Comment': ['raw legacy test']}).to_csv(
                    root / 'multiclass_validation_inputs.csv', index=False)
                run_llm_task(task, config, auto_resume=False)
                output = root / 'outputs' / task.output_subdir
                summary = json.loads((output / 'run_summary.json').read_text())
                self.assertEqual(Path(summary['prediction_dataset']['filename']).name, TASK_B_OPEN_VALIDATION_FILENAME)
                self.assertEqual(summary['prediction_dataset']['rows'], 6)
                self.assertEqual(summary['official_validation']['dataset']['rows'], 6)
                self.assertEqual(summary['official_validation']['dataset']['filename'], TASK_B_OPEN_VALIDATION_FILENAME)
                self.assertIsNotNone(summary['official_validation']['ensemble_metrics'])
                self.assertEqual(summary['prompt_input']['include_english_translation'], include)
                predictions = pd.read_csv(output / 'predictions.csv')
                self.assertEqual(predictions['id'].tolist(), [f'dev-{i}' for i in range(6)])
                self.assertTrue((output / 'official_validation_metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
