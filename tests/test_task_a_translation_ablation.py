"""Task A view ablation, independent open evaluation and majority-vote metrics."""
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from llm_config import (
    TASK_A, LLM_TRAINING, TASK_A_INCLUDE_ENGLISH_TRANSLATION,
    TASK_A_OPEN_VALIDATION_FILENAME, PROJECT_ROOT,
)
from src.data import load_task_frames, resolve_data_file, prompt_inputs
from src.llm_data import ChatEncoder, CompletionDataset
from src.llm_training import (
    _input_hashes, _resume_signature, prediction_label_metrics, run_llm_cli,
    run_llm_task, score_candidates, train_llm_epoch,
)
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime, tiny_model


class TaskATranslationAblationTests(unittest.TestCase):
    def write_data(self, root, task):
        for fold, filename in enumerate(task.original_fold_filenames, 1):
            rows = [{'id': f'train-{fold}-{i}', 'Comment': f'EN training {fold}-{i}',
                     'OriginalComment': f'RAW training {fold}-{i}', 'Label': task.labels[i % 2]}
                    for i in range(4)]
            if fold == 1:
                rows.append({'id': 'open-1', 'Comment': 'EN open 1',
                             'OriginalComment': 'RAW open 1', 'Label': task.labels[0]})
            pd.DataFrame(rows).to_csv(root / filename, index=False)
        pd.DataFrame({'id': ['open-1', 'open-2'], 'Comment': ['EN open 1', 'EN open 2'],
                      'OriginalComment': ['RAW open 1', 'RAW open 2'],
                      'Label': list(task.labels)}).to_csv(root / task.prediction_filename, index=False)
        pd.DataFrame({'id': ['private-old'], 'Comment': ['raw-only old test']}).to_csv(
            root / 'binary_validation_inputs.csv', index=False)

    def boundary(self, stack, root):
        stack.enter_context(patch('src.data.resolve_data_file', side_effect=root.joinpath))
        stack.enter_context(patch('src.llm_training.resolve_data_file', side_effect=root.joinpath))
        stack.enter_context(patch('src.llm_training.initialize_runtime', return_value=cpu_runtime()))
        stack.enter_context(patch('src.llm_training.require_cuda'))
        stack.enter_context(patch('src.llm_training.cleanup_runtime'))
        stack.enter_context(patch('src.llm_training.OUTPUT_ROOT', root / 'outputs'))
        stack.enter_context(patch('src.llm_training.CHECKPOINT_ROOT', root / 'checkpoints'))
        stack.enter_context(patch('src.llm_training.AutoTokenizer.from_pretrained', return_value=TestChatTokenizer()))
        return stack.enter_context(patch('src.llm_training.load_lora_model', side_effect=lambda *_: tiny_model()))

    def test_branch_defaults_and_cli_preserve_binary_training_and_isolate_artifacts(self):
        include = TASK_A_INCLUDE_ENGLISH_TRANSLATION
        self.assertEqual(TASK_A.include_english_translation, include)
        expected = 'khangtest3_bilingual_english_first_256' if include else 'khangtest2_original_only_256'
        self.assertTrue(TASK_A.output_subdir.endswith(expected))
        self.assertEqual(TASK_A.output_subdir, TASK_A.checkpoint_subdir)
        self.assertEqual(TASK_A.prediction_filename, TASK_A_OPEN_VALIDATION_FILENAME)
        self.assertEqual(TASK_A.training_exclusion_filename, TASK_A_OPEN_VALIDATION_FILENAME)
        self.assertTrue(TASK_A.ensemble_all_seeds)
        self.assertTrue(TASK_A.evaluate_prediction_labels)
        self.assertFalse(TASK_A.use_augmentation)
        self.assertEqual(TASK_A.selection_seeds, (42, 52, 62, 72, 82))
        with patch('sys.argv', ['Train_A_LLM.py']), patch(
            'src.llm_training.relaunch_with_torchrun_if_needed'
        ), patch('src.llm_training.run_llm_task') as runner:
            run_llm_cli(TASK_A)
        task, config = runner.call_args.args[:2]
        self.assertEqual(task, TASK_A)
        self.assertEqual(config, LLM_TRAINING)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertEqual(config.early_stopping_patience, 4)

    def test_same_system_and_answers_with_only_the_english_block_toggled(self):
        pair = ('ENGLISH_SENTINEL', 'RAW_SENTINEL')
        raw = ChatEncoder(TestChatTokenizer(), replace(TASK_A, include_english_translation=False), 512)
        paired = ChatEncoder(TestChatTokenizer(), replace(TASK_A, include_english_translation=True), 512)
        self.assertEqual(raw.system_text, paired.system_text)
        self.assertEqual(raw.completions, paired.completions)
        self.assertEqual(raw.messages(pair)[1]['content'], 'Original code-mixed comment:\nRAW_SENTINEL')
        self.assertEqual(paired.messages(pair)[1]['content'],
                         'English translation:\nENGLISH_SENTINEL\nOriginal code-mixed comment:\nRAW_SENTINEL')
        self.assertEqual(raw.encode_prompt(pair), raw.encode_prompt(('changed translation ' * 500, pair[1])))

    def test_long_raw_only_prompt_keeps_head_tail_and_complete_label_without_english_budget(self):
        encoder = ChatEncoder(TestChatTokenizer(), replace(TASK_A, include_english_translation=False), 512)
        raw = 'RAW_START ' + 'raw ' * 500 + ' RAW_END'
        prompt, truncated = encoder.encode_prompt(('ENGLISH_SENTINEL ' * 800, raw))
        self.assertTrue(truncated)
        self.assertEqual((prompt, truncated), encoder.encode_prompt(('', raw)))
        decoded = encoder.tokenizer.decode(prompt)
        for part in ('RAW_START', 'RAW_END', '[...]'):
            self.assertIn(part, decoded)
        self.assertNotIn('ENGLISH_SENTINEL', decoded)
        for label in range(2):
            feature = encoder.feature(prompt, label)
            self.assertLessEqual(len(feature['input_ids']), 512)
            self.assertEqual(feature['labels'][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(feature['labels'][len(prompt):], encoder.completions[label])

    def test_bundled_open_set_has_806_complete_unique_labeled_pairs(self):
        path = resolve_data_file(TASK_A_OPEN_VALIDATION_FILENAME)
        self.assertEqual(path, PROJECT_ROOT / 'data' / 'open_validation' / TASK_A_OPEN_VALIDATION_FILENAME)
        frame = pd.read_csv(path, dtype={'id': str}, keep_default_na=False)
        self.assertEqual(len(frame), 806)
        self.assertEqual(set(frame.columns), {'id', 'OriginalComment', 'Comment', 'Label'})
        self.assertFalse(frame['id'].duplicated().any())
        for column in ('id', 'OriginalComment', 'Comment'):
            self.assertFalse(frame[column].str.strip().eq('').any())
        self.assertEqual(set(frame['Label']), set(TASK_A.labels))

    def test_open_ids_are_removed_before_selection_and_full_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_data(root, TASK_A)
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                training, prediction = load_task_frames(TASK_A)
            self.assertEqual(len(training), 20)
            self.assertEqual(len(prediction), 2)
            self.assertFalse(set(training['id']) & set(prediction['id']))
            self.assertEqual(training.groupby('fold').size().to_dict(), {i: 4 for i in range(1, 6)})
            audit = training.attrs['training_exclusions']
            self.assertEqual((audit['loaded_rows'], audit['excluded_rows'], audit['remaining_rows']), (21, 1, 20))
            self.assertEqual(audit['excluded_records'][0]['id'], 'open-1')
            encoder = ChatEncoder(TestChatTokenizer(), TASK_A, 512)
            inference = CompletionDataset(prompt_inputs(prediction, TASK_A), encoder)
            self.assertIsNone(inference.labels)

    def test_training_fingerprint_tracks_excluded_reference_ids_not_prediction_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_data(root, TASK_A)
            private = root / 'private.csv'
            pd.DataFrame({'id': ['private-1'], 'Comment': ['EN'], 'OriginalComment': ['RAW']}).to_csv(private, index=False)
            with patch('src.data.resolve_data_file', side_effect=root.joinpath), patch(
                'src.llm_training.resolve_data_file', side_effect=root.joinpath
            ):
                first = _input_hashes(TASK_A)
                changed_prediction = _input_hashes(replace(TASK_A, prediction_filename=str(private)))
                self.assertEqual(first[0], changed_prediction[0])
                self.assertNotEqual(first[1], changed_prediction[1])
                reference = pd.read_csv(root / TASK_A.prediction_filename)
                reference.loc[0, 'id'] = 'train-2-0'
                reference.to_csv(root / TASK_A.prediction_filename, index=False)
                self.assertNotEqual(first[0], _input_hashes(TASK_A)[0])

    def test_resume_separates_views_and_rejects_old_task_a_prompt_policy(self):
        with patch('src.llm_training.labeled_input_filenames', return_value=TASK_A.original_fold_filenames):
            raw = _resume_signature(replace(TASK_A, include_english_translation=False), LLM_TRAINING, 'same-data', cpu_runtime())
            paired = _resume_signature(replace(TASK_A, include_english_translation=True), LLM_TRAINING, 'same-data', cpu_runtime())
            old = _resume_signature(replace(TASK_A, prompt_style='paired', training_exclusion_filename=None), LLM_TRAINING, 'same-data', cpu_runtime())
        self.assertNotEqual(raw, paired)
        self.assertNotEqual(paired, old)
        self.assertEqual(raw['system_prompt_sha256'], paired['system_prompt_sha256'])
        self.assertIn('training_exclusion_filename', raw)

    def test_metrics_use_voted_labels_and_unlabeled_overrides_have_no_fabricated_gold(self):
        frame = pd.DataFrame({'id': ['a', 'b'], 'Label': list(TASK_A.labels)})
        metrics = prediction_label_metrics(frame, TASK_A, np.array([0, 1]))
        self.assertEqual(metrics['macro_f1'], 1.0)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['confusion_matrix']['values'], [[1, 0], [0, 1]])
        self.assertIsNone(prediction_label_metrics(frame.drop(columns='Label'), TASK_A, [0, 1]))

    def test_invalid_open_gold_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_data(root, TASK_A)
            frame = pd.read_csv(root / TASK_A.prediction_filename)
            frame.loc[0, 'Label'] = 'invalid'
            frame.to_csv(root / TASK_A.prediction_filename, index=False)
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                with self.assertRaisesRegex(ValueError, 'unknown gold labels'):
                    load_task_frames(TASK_A)

    def test_five_seed_pipeline_uses_each_view_policy_and_reports_majority_vote_f1(self):
        for include in (False, True):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                task = replace(TASK_A, include_english_translation=include)
                config = replace(LLM_TRAINING, max_length=512, max_epochs=1, eval_batch_size_per_gpu=6)
                self.write_data(root, task)
                model_loader = self.boundary(stack, root)
                prediction_calls = []

                def check(data, training=False):
                    for prompt in data.prompts:
                        decoded = data.encoder.tokenizer.decode(prompt)
                        self.assertIn('Original code-mixed comment:', decoded)
                        self.assertEqual('English translation:' in decoded, include)
                        if training:
                            self.assertNotIn('RAW open', decoded)

                def train(*args, **kwargs):
                    check(args[1].dataset, training=True)
                    return train_llm_epoch(*args, **kwargs)

                def score(*args, **kwargs):
                    check(args[1])
                    scores = score_candidates(*args, **kwargs)
                    if args[5] == 'Development prediction from full-data adapter':
                        self.assertIsNone(args[1].labels)
                        # The first three seeds are weakly correct; the final
                        # two confidently disagree. A soft ensemble is wrong.
                        values = [[.51, .49], [.49, .51]] if len(prediction_calls) < 3 else [[.001, .999], [.999, .001]]
                        scores['probabilities'] = np.array(values)
                        scores['mean_token_log_probs'] = np.log(scores['probabilities'])
                        prediction_calls.append(scores['probabilities'])
                    return scores

                with patch('src.llm_training.train_llm_epoch', side_effect=train), patch(
                    'src.llm_training.score_candidates', side_effect=score
                ):
                    run_llm_task(task, config, auto_resume=False)
                self.assertEqual(model_loader.call_count, 10)
                self.assertEqual(len(prediction_calls), 5)
                np.testing.assert_array_equal(np.mean(prediction_calls, axis=0).argmax(axis=1), [1, 0])
                output = root / 'outputs' / task.output_subdir
                final = pd.read_csv(output / 'predictions.csv')
                self.assertEqual(final['label'].tolist(), list(task.labels))
                summary = json.loads((output / 'run_summary.json').read_text())
                self.assertEqual(summary['official_validation']['macro_f1'], 1.0)
                self.assertEqual(summary['official_validation']['rows'], 2)
                self.assertEqual(summary['include_english_translation'], include)
                self.assertEqual(summary['training_exclusions']['excluded_rows'], 1)
                self.assertTrue((output / 'official_validation_metrics.json').is_file())
                self.assertTrue((output / 'training_exclusions.json').is_file())
                # Completed seed caches still produce the same evaluated vote.
                run_llm_task(task, config, auto_resume=False)
                self.assertEqual(model_loader.call_count, 10)
                metrics = json.loads((output / 'official_validation_metrics.json').read_text())
                self.assertEqual(metrics['macro_f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
