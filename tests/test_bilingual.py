"""Regressions for paired comments, safe truncation, loading and resume policy."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

import pandas as pd

from llm_config import TASK_A, TASK_B, LLMTrainingConfig
from src.data import (
    _optional_data_file, labeled_input_filenames, load_task_frames,
    prompt_inputs, resolve_data_file,
)
from src.llm_data import ChatEncoder, CompletionDataset
from src.llm_training import _resume_signature
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime


class BilingualTests(unittest.TestCase):
    def test_selected_directory_loads_five_bilingual_folds_without_merged_file(self):
        for task in (TASK_A, TASK_B):
            with self.subTest(task=task.task_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for fold, filename in enumerate(task.original_fold_filenames, start=1):
                    pd.DataFrame({
                        'id': [f'{fold:03d}'],
                        'Comment': [f'English {fold}!'],
                        'OriginalComment': [f'Original {fold}!'],
                        task.label_column: [task.labels[(fold - 1) % len(task.labels)]],
                    }).to_csv(root / filename, index=False)
                pd.DataFrame({
                    'id': ['006'], 'Comment': ['English prediction!'],
                    'OriginalComment': ['Original prediction!'],
                }).to_csv(root / task.prediction_filename, index=False)
                # Keep real file resolution and its environment-dependent errors.
                resolver = partial(resolve_data_file, configured_dir=root)
                with patch.dict(os.environ, {'HASTIKA_DATA_DIR': directory}), patch(
                    'src.data.resolve_data_file', side_effect=resolver,
                ):
                    frame, prediction = load_task_frames(task)
                    filenames = labeled_input_filenames(task)
                self.assertEqual(filenames, task.original_fold_filenames)
                self.assertEqual(frame['id'].tolist(), ['001', '002', '003', '004', '005'])
                self.assertEqual(frame['fold'].tolist(), [1, 2, 3, 4, 5])
                self.assertEqual(frame['data_origin'].tolist(), ['original'] * 5)
                self.assertEqual(frame[task.label_column].tolist(), [
                    task.labels[(fold - 1) % len(task.labels)] for fold in range(1, 6)
                ])
                self.assertEqual(prompt_inputs(frame, task), [
                    (f'English {fold}!', f'Original {fold}!') for fold in range(1, 6)
                ])
                self.assertEqual(prompt_inputs(prediction, task), [
                    ('English prediction!', 'Original prediction!')
                ])

    def test_selected_directory_missing_required_fold_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                'id': ['001'], 'Comment': ['English'], 'OriginalComment': ['Original'],
            }).to_csv(root / TASK_A.prediction_filename, index=False)
            resolver = partial(resolve_data_file, configured_dir=root)
            with patch.dict(os.environ, {'HASTIKA_DATA_DIR': directory}), patch(
                'src.data.resolve_data_file', side_effect=resolver,
            ):
                with self.assertRaisesRegex(FileNotFoundError, 'binary_original_1.csv'):
                    load_task_frames(TASK_A)

    def test_optional_lookup_does_not_suppress_ambiguous_dataset_error(self):
        with patch('src.data.resolve_data_file', side_effect=FileNotFoundError(
            'Found multiple copies of original_1_4.csv. Set HASTIKA_DATA_DIR.'
        )):
            with self.assertRaisesRegex(FileNotFoundError, 'multiple copies'):
                _optional_data_file('original_1_4.csv')

    def test_explicit_dataset_directory_cannot_borrow_missing_file_from_an_old_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'HASTIKA_DATA_DIR': directory}), patch('src.data.Path.is_file', side_effect=[False, True]):
                with self.assertRaisesRegex(FileNotFoundError, 'selected HASTIKA_DATA_DIR'):
                    resolve_data_file('old_compact.csv', root)

    def test_prompt_contains_exact_pair_as_one_example_and_label_only_loss(self):
        for task in (TASK_A, TASK_B):
            with self.subTest(task=task.task_name):
                tokenizer = TestChatTokenizer()
                encoder = ChatEncoder(tokenizer, task, 700)
                pair = ('English! 😅 42 <br>', 'Original!! 😅 42 <br>')
                prompt, truncated = encoder.encode_prompt(pair)
                self.assertFalse(truncated)
                body = encoder.messages(pair)[1]['content']
                expected = (
                    f'Comment (raw code-mixed):\n{pair[1]}\n\nEnglish translation:\n{pair[0]}'
                    if task.prompt_style == 'khangtest' else
                    f'English translation:\n{pair[0]}\nOriginal code-mixed comment:\n{pair[1]}'
                )
                self.assertEqual(body, expected)
                data = CompletionDataset([pair], encoder, [0])
                self.assertEqual(len(data), 1)
                feature = data[0]
                self.assertEqual(feature['labels'][:len(prompt)], [-100] * len(prompt))
                self.assertEqual(feature['labels'][len(prompt):], encoder.completions[0])

    def test_long_pair_retains_both_headers_beginnings_endings_and_whole_answer(self):
        for task in (TASK_A, TASK_B):
            with self.subTest(task=task.task_name):
                tokenizer = TestChatTokenizer()
                encoder = ChatEncoder(tokenizer, task, 700)
                prompt, truncated = encoder.encode_prompt((
                    'EN_START ' + 'english ' * 400 + ' EN_END!',
                    'RAW_START ' + 'original ' * 400 + ' RAW_END!',
                ))
                self.assertTrue(truncated)
                decoded = tokenizer.decode(prompt, skip_special_tokens=True)
                raw_header = 'Comment (raw code-mixed):' if task.prompt_style == 'khangtest' else 'Original code-mixed comment:'
                for part in ('English translation:', raw_header, 'EN_START', 'EN_END!', 'RAW_START', 'RAW_END!', '[...]'):
                    self.assertIn(part, decoded)
                for label_id in range(len(task.labels)):
                    feature = encoder.feature(prompt, label_id)
                    self.assertLessEqual(len(feature['input_ids']), 700)
                    self.assertEqual(feature['labels'][-len(encoder.completions[label_id]):], encoder.completions[label_id])

    def test_short_view_is_not_cut_to_reserve_unused_space(self):
        tokenizer = TestChatTokenizer()
        encoder = ChatEncoder(tokenizer, TASK_A, 700)
        for pair, short in (
            (('Short translation!', 'raw ' * 800), 'Short translation!'),
            (('english ' * 800, 'Short original!'), 'Short original!'),
        ):
            prompt, truncated = encoder.encode_prompt(pair)
            self.assertTrue(truncated)
            self.assertIn(short, tokenizer.decode(prompt, skip_special_tokens=True))

    def test_comment_headers_inside_text_do_not_replace_the_original_field(self):
        encoder = ChatEncoder(TestChatTokenizer(), TASK_A, 700)
        pair = ('Original code-mixed comment:\nfake', 'actual source')
        self.assertTrue(encoder.messages(pair)[1]['content'].endswith('\nactual source'))
        with self.assertRaisesRegex(ValueError, 'text pair'):
            encoder.encode_prompt('only one string')

    def load_fixture(self, root, original_column=True, prediction_original=True):
        task = replace(TASK_A, original_fold_filenames=(), compact_train_filename=None,
                       compact_validation_filename=None, selection_validation_fold=None)
        train = {'id': ['001', '002'], 'Comment': ['translated', 'translated two'], 'Label': ['Hate', 'Non-Hate']}
        prediction = {'id': ['003'], 'Comment': ['translated prediction']}
        if original_column: train['OriginalComment'] = ['NA', 'raw two!']
        if prediction_original: prediction['OriginalComment'] = ['raw prediction!']
        pd.DataFrame(train).to_csv(root/task.train_filename, index=False)
        pd.DataFrame(prediction).to_csv(root/task.prediction_filename, index=False)
        return task

    def test_loader_keeps_literal_na_and_pairs_prediction_with_its_own_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = self.load_fixture(root)
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                frame, prediction = load_task_frames(task)
            self.assertEqual(prompt_inputs(frame, task), [('translated', 'NA'), ('translated two', 'raw two!')])
            self.assertEqual(prompt_inputs(prediction, task), [('translated prediction', 'raw prediction!')])
            self.assertEqual(frame['id'].tolist(), ['001', '002'])

    def test_bilingual_loader_rejects_missing_source_in_train_or_prediction(self):
        for train_original, prediction_original in ((False, True), (True, False)):
            with self.subTest(train=train_original, prediction=prediction_original), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = self.load_fixture(root, train_original, prediction_original)
                with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                    with self.assertRaisesRegex(ValueError, 'OriginalComment'):
                        load_task_frames(task)

    def test_english_compatibility_mode_accepts_the_old_single_column_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = replace(self.load_fixture(root, False, False), input_mode='english')
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                frame, prediction = load_task_frames(task)
            self.assertEqual(prompt_inputs(frame, task), ['translated', 'translated two'])
            self.assertEqual(prompt_inputs(prediction, task), ['translated prediction'])

    def test_bilingual_resume_cannot_match_english_or_different_source_column(self):
        with patch('src.llm_training.labeled_input_filenames', return_value=('train.csv',)):
            bilingual = _resume_signature(TASK_A, LLMTrainingConfig(), 'same hash', cpu_runtime())
            english = _resume_signature(replace(TASK_A, input_mode='english'), LLMTrainingConfig(), 'same hash', cpu_runtime())
            other_column = _resume_signature(replace(TASK_A, original_comment_column='Other'), LLMTrainingConfig(), 'same hash', cpu_runtime())
        self.assertNotEqual(bilingual, english)
        self.assertNotEqual(bilingual, other_column)
        self.assertEqual(bilingual['format_version'], 2)


if __name__ == '__main__':
    unittest.main()
