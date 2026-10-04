"""Task B weighting, gold alignment, full-data soft ensemble and timeout recovery."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from llm_config import TASK_A, TASK_B, LLM_TRAINING_B
from src.data import load_epoch_validation_frame
from src.llm_data import ChatEncoder
from src.llm_training import run_llm_task, completion_log_probs
from src.task_b_training import (
    TaskBValidationMonitor, WeightedDataset, WeightedCollator,
    compute_class_weights, train_weighted_epoch,
)
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime, tiny_model


class TaskBKhangtestTests(unittest.TestCase):
    def write_data(self, root, task, gold=True):
        for fold, filename in enumerate(task.original_fold_filenames, 1):
            pd.DataFrame({
                'id': [f'train-{fold}-{i}' for i in range(6)],
                'Comment': [f'English {fold}-{i}!' for i in range(6)],
                'OriginalComment': [f'raw {fold}-{i}!' for i in range(6)],
                task.label_column: list(task.labels),
            }).to_csv(root / filename, index=False)
        pd.DataFrame({
            'id': [f'dev-{i}' for i in range(6)],
            'Comment': [f'English dev {i}!' for i in range(6)],
            'OriginalComment': [f'raw dev {i}!' for i in range(6)],
        }).to_csv(root / task.prediction_filename, index=False)
        if gold:
            pd.DataFrame({
                'id': [f'dev-{i}' for i in reversed(range(6))],
                'Comment': [f'raw dev {i}!' for i in reversed(range(6))],
                task.label_column: list(reversed(task.labels)),
            }).to_csv(root / task.epoch_validation_filename, index=False)

    def boundary(self, stack, root):
        stack.enter_context(patch('src.data.resolve_data_file', side_effect=root.joinpath))
        stack.enter_context(patch('src.llm_training.resolve_data_file', side_effect=root.joinpath))
        stack.enter_context(patch('src.task_b_training.resolve_data_file', side_effect=root.joinpath))
        stack.enter_context(patch('src.task_b_training.initialize_runtime', return_value=cpu_runtime()))
        stack.enter_context(patch('src.task_b_training.require_cuda'))
        stack.enter_context(patch('src.task_b_training.cleanup_runtime'))
        stack.enter_context(patch('src.task_b_training.OUTPUT_ROOT', root / 'outputs'))
        stack.enter_context(patch('src.task_b_training.CHECKPOINT_ROOT', root / 'checkpoints'))
        stack.enter_context(patch('src.task_b_training.AutoTokenizer.from_pretrained', return_value=TestChatTokenizer()))
        return stack.enter_context(patch('src.task_b_training.load_lora_model', side_effect=lambda *args: tiny_model()))

    def test_weights_follow_khangtest_and_mean_sample_weight_is_one(self):
        ids = np.array([0] * 100 + [1] * 10 + [2] * 5)
        weights, counts = compute_class_weights(ids, 3, .5, .75, 1.5)
        expected = np.clip((115 / (3 * counts)) ** .5, .75, 1.5)
        expected /= np.dot(counts, expected) / 115
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
        self.assertAlmostEqual(float(weights[ids].mean()), 1., places=6)
        self.assertGreater(weights[2], weights[0])

    def test_weighted_loss_uses_gold_class_and_does_not_weight_candidate_scoring(self):
        task = replace(TASK_B, input_mode='english')
        encoder = ChatEncoder(TestChatTokenizer(), task, 512)
        data = WeightedDataset(['first', 'second'], encoder, [0, 1])
        collator = WeightedCollator(0)
        model = tiny_model()
        weights = torch.tensor([2., .5, 1., 1., 1., 1.])
        batch = collator([data[0], data[1]])
        model.eval()
        scores, counts = completion_log_probs(model, {k:v for k,v in batch.items() if k != 'class_ids'})
        expected = float(((-scores / counts) * weights[batch['class_ids']]).mean().detach())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
        loss = train_weighted_epoch(model, DataLoader(data, batch_size=2, collate_fn=collator), optimizer,
            scheduler, torch.amp.GradScaler('cuda', enabled=False), LLM_TRAINING_B, cpu_runtime(), 1, class_weights=weights)
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertNotIn('class_ids', collator([encoder.feature(data.prompts[0], 0)]))

    def test_gold_alignment_preserves_order_and_uses_both_columns_without_gold_in_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_data(root, TASK_B)
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                frame = load_epoch_validation_frame(TASK_B)
            self.assertEqual(frame['id'].tolist(), [f'dev-{i}' for i in reversed(range(6))])
            self.assertEqual(frame['text'].iloc[0], 'English dev 5!')
            self.assertEqual(frame['original_text'].iloc[0], 'raw dev 5!')
            encoder = ChatEncoder(TestChatTokenizer(), TASK_B, 512)
            body = encoder.messages((frame['text'].iloc[0], frame['original_text'].iloc[0]))[1]['content']
            self.assertEqual(body, 'Comment (raw code-mixed):\nraw dev 5!\n\nEnglish translation:\nEnglish dev 5!')
            self.assertNotIn('Others', body)
            # A test prediction override must not change the development source.
            with patch('src.data.resolve_data_file', side_effect=root.joinpath):
                overridden = load_epoch_validation_frame(replace(TASK_B, prediction_filename='test.csv'))
            pd.testing.assert_frame_equal(frame, overridden)

    def test_invalid_gold_missing_ids_raw_mismatch_and_train_overlap_are_rejected(self):
        for failure in ('unknown_label', 'missing_id', 'raw_mismatch', 'train_overlap'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_data(root, TASK_B)
                gold = pd.read_csv(root / TASK_B.epoch_validation_filename)
                if failure == 'unknown_label': gold.loc[0, TASK_B.label_column] = 'Invalid'
                if failure == 'missing_id': gold.loc[0, 'id'] = 'not-in-input'
                if failure == 'raw_mismatch': gold.loc[0, 'Comment'] = 'wrong raw'
                gold.to_csv(root / TASK_B.epoch_validation_filename, index=False)
                with patch('src.data.resolve_data_file', side_effect=root.joinpath), patch('src.task_b_training.resolve_data_file', side_effect=root.joinpath):
                    with self.assertRaises(ValueError):
                        if failure == 'train_overlap':
                            TaskBValidationMonitor(TASK_B, LLM_TRAINING_B, cpu_runtime(), pd.DataFrame({'id':['dev-5']}),
                                ChatEncoder(TestChatTokenizer(), TASK_B, 512), WeightedCollator(0))
                        else:
                            load_epoch_validation_frame(TASK_B)

    def test_full_data_soft_ensemble_and_completed_seeds_resume_without_loading_model(self):
        for with_gold in (True, False):
            with self.subTest(gold=with_gold), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                task = replace(TASK_B, fixed_full_retrain_epochs=2, selection_seeds=(42, 52))
                self.write_data(root, task, with_gold)
                loader = self.boundary(stack, root)
                run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
                self.assertEqual(loader.call_count, 2)
                output = root / 'outputs' / task.output_subdir
                summary = json.loads((output / 'run_summary.json').read_text())
                self.assertTrue(summary['epoch_selection']['oof_skipped'])
                self.assertIsNone(summary['class_weighting']['downsampling'])
                components = []
                for seed in task.selection_seeds:
                    history = json.loads((output / f'seed_{seed}' / 'full_data_training_history.json').read_text())
                    self.assertEqual([r['epoch'] for r in history['epochs']], [1,2])
                    seed_summary = json.loads((output / f'seed_{seed}' / 'fixed_task_b_summary.json').read_text())
                    self.assertEqual(seed_summary['training_rows'], 30) # All five folds, no Gender cap.
                    with np.load(output / f'seed_{seed}' / 'prediction_scores.npz') as values:
                        components.append(values['probabilities'].copy())
                audit = pd.read_csv(output / 'ensemble_predictions.csv')
                columns = [f'mean_prob_{label.lower().replace("-", "_").replace(" ", "_")}' for label in task.labels]
                np.testing.assert_allclose(audit[columns], np.mean(components, axis=0), atol=1e-7)
                self.assertEqual((output / 'official_validation_metrics.json').exists(), with_gold)
                run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
                self.assertEqual(loader.call_count, 2)
                archive = root / 'previous-run.zip'
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                    for directory in (root / 'outputs', root / 'checkpoints'):
                        for file in directory.rglob('*'):
                            if file.is_file():
                                bundle.write(file, 'previous-run/' + str(file.relative_to(root)))
                with patch('src.task_b_training.OUTPUT_ROOT', root / 'continued' / 'outputs'), patch(
                    'src.task_b_training.CHECKPOINT_ROOT', root / 'continued' / 'checkpoints',
                ):
                    run_llm_task(task, LLM_TRAINING_B, resume_sources=(archive,), auto_resume=False)
                self.assertEqual(loader.call_count, 2) # Resume an attached ZIP in a fresh output tree.
                inputs = pd.read_csv(root / task.prediction_filename)
                inputs.loc[0, 'Comment'] = 'Changed English development input!'
                inputs.to_csv(root / task.prediction_filename, index=False)
                with patch('src.task_b_training.train_weighted_epoch') as training:
                    run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
                training.assert_not_called()
                self.assertEqual(loader.call_count, 4) # Reuse adapters, refresh predictions.

    def test_timeout_recovers_from_completed_epoch_instead_of_retraining_it(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            task = replace(TASK_B, fixed_full_retrain_epochs=2, selection_seeds=(42, 52))
            self.write_data(root, task, gold=False)
            self.boundary(stack, root)
            original = train_weighted_epoch
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(args[7])
                if len(calls) == 2:
                    raise RuntimeError('simulated timeout')
                return original(*args, **kwargs)
            with patch('src.task_b_training.train_weighted_epoch', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated timeout'):
                    run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
            calls.clear()
            def resumed(*args, **kwargs):
                calls.append(args[7])
                return original(*args, **kwargs)
            with patch('src.task_b_training.train_weighted_epoch', side_effect=resumed):
                run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
            self.assertEqual(calls, [2,1,2]) # seed 42 resumes epoch 2, then seed 52 starts fresh.
            resumed_scores = []
            for seed in task.selection_seeds:
                with np.load(root / 'outputs' / task.output_subdir / f'seed_{seed}' / 'prediction_scores.npz') as values:
                    resumed_scores.append(values['probabilities'].copy())
            with tempfile.TemporaryDirectory() as baseline_directory, ExitStack() as baseline_stack:
                baseline = Path(baseline_directory)
                self.write_data(baseline, task, gold=False)
                self.boundary(baseline_stack, baseline)
                run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
                for index, seed in enumerate(task.selection_seeds):
                    with np.load(baseline / 'outputs' / task.output_subdir / f'seed_{seed}' / 'prediction_scores.npz') as values:
                        np.testing.assert_allclose(resumed_scores[index], values['probabilities'], atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
