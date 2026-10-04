"""Task B per-seed selection, fresh retraining, split isolation and phase resume."""
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

from llm_config import TASK_B, LLM_TRAINING_B
from src.llm_training import run_llm_task
from src.task_b_training import score_candidates, train_weighted_epoch
from tests import test_task_b_khangtest as fixtures


class TaskBSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TaskBKhangtestTests()
        self.task = replace(TASK_B, selection_seeds=(42, 52))
        self.config = replace(LLM_TRAINING_B, max_length=512, max_epochs=4, early_stopping_patience=2)

    def test_per_seed_best_epochs_precede_fresh_full_data_training_and_keep_soft_ensemble(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.fixture.write_data(root, self.task, gold=True)
            model_loader = self.fixture.boundary(stack, root)
            calls, phase_models = [], []
            def training(*args, **kwargs):
                calls.append((args[8], args[7], len(args[1].dataset)))
                if args[7] == 1:
                    phase_models.append(args[0])
                self.assertAlmostEqual(float(kwargs['class_weights'].mean()), 1., places=5)
                return train_weighted_epoch(*args, **kwargs)
            def scoring(*args, **kwargs):
                result = score_candidates(*args, **kwargs)
                heading = args[5]
                if heading.startswith('Holdout validation'):
                    seed, epoch = int(heading.split()[3]), int(heading.split()[-1])
                    # Seed 42 peaks at 2, seed 52 at 1. Official gold cannot influence this.
                    result['macro_f1'] = {42: [.2,.8,.3,.2], 52: [.9,.4,.3,.2]}[seed][epoch-1]
                    result['loss'] = float(epoch)
                    self.assertEqual(len(args[1]), 6)
                return result
            with patch('src.task_b_training.train_weighted_epoch', side_effect=training), patch(
                'src.task_b_training.score_candidates', side_effect=scoring):
                run_llm_task(self.task, self.config, auto_resume=False)
            self.assertEqual(model_loader.call_count, 4)
            self.assertEqual(len({id(model) for model in phase_models}), 4)
            self.assertEqual([(name, epoch, rows) for name, epoch, rows in calls],
                [('Holdout train seed 42', i, 24) for i in (1,2,3,4)] +
                [('Holdout train seed 52', i, 24) for i in (1,2,3)] +
                [('Full-data retrain seed 42', i, 30) for i in (1,2)] +
                [('Full-data retrain seed 52', 1, 30)])
            output = root / 'outputs' / self.task.output_subdir
            summary = json.loads((output / 'run_summary.json').read_text())
            self.assertEqual([item['epochs'] for item in summary['full_data_retrains']], [2,1])
            self.assertEqual(summary['epoch_selection']['strategy'], 'per_seed_holdout')
            manifest = pd.read_csv(output / 'split_manifest.csv')
            self.assertTrue((manifest.loc[manifest['split']=='selection_validation','fold']==5).all())
            self.assertTrue((manifest.loc[manifest['split']=='selection_train','fold']!=5).all())
            scores = []
            for seed, best_epoch in ((42,2),(52,1)):
                selected = json.loads((output / f'seed_{seed}' / 'selection_summary.json').read_text())
                self.assertEqual(selected['best_epoch'], best_epoch)
                self.assertEqual(selected['class_weighting']['counts'], dict(zip(self.task.labels,[4]*6)))
                self.assertTrue((root/'checkpoints'/self.task.checkpoint_subdir/f'seed_{seed}'/'best_holdout_adapter'/'adapter_config.json').is_file())
                with np.load(output/f'seed_{seed}'/'prediction_scores.npz') as values:
                    scores.append(values['probabilities'].copy())
            cols = [f'mean_prob_{label.lower().replace("-", "_")}' for label in self.task.labels]
            np.testing.assert_allclose(pd.read_csv(output/'ensemble_predictions.csv')[cols], np.mean(scores,axis=0),atol=1e-7)
            run_llm_task(self.task, self.config, auto_resume=False)
            self.assertEqual(model_loader.call_count, 4) # Both selection and full training are reused.

    def test_selection_and_full_retrain_timeout_recovery_match_uninterrupted_run(self):
        task = self.task
        config = replace(self.config, max_epochs=2)
        for fail_phase in ('Holdout train seed 42','Full-data retrain seed 42'):
            with self.subTest(phase=fail_phase), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                self.fixture.write_data(root, task, gold=False)
                self.fixture.boundary(stack, root)
                calls = []
                def interrupted(*args, **kwargs):
                    calls.append((args[8],args[7]))
                    if args[8] == fail_phase and args[7] == 2:
                        raise RuntimeError('simulated timeout')
                    return train_weighted_epoch(*args, **kwargs)
                # Make epoch 2 better for both seeds, so full retraining has two epochs.
                def scoring(*args, **kwargs):
                    result = score_candidates(*args, **kwargs)
                    if args[5].startswith('Holdout validation'):
                        result['macro_f1'] = float(args[5].split()[-1]) / 10
                    return result
                with patch('src.task_b_training.score_candidates',side_effect=scoring):
                    with patch('src.task_b_training.train_weighted_epoch',side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError,'simulated timeout'):
                            run_llm_task(task,config,auto_resume=False)
                    calls.clear()
                    resume_sources, result_root = (), root
                    if fail_phase.startswith('Full'):
                        archive = root / 'previous-run.zip'
                        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                            for tree in (root/'outputs', root/'checkpoints'):
                                for file in tree.rglob('*'):
                                    if file.is_file():
                                        bundle.write(file, 'previous-run/' + str(file.relative_to(root)))
                        result_root = root / 'continued'
                        stack.enter_context(patch('src.task_b_training.OUTPUT_ROOT', result_root/'outputs'))
                        stack.enter_context(patch('src.task_b_training.CHECKPOINT_ROOT', result_root/'checkpoints'))
                        resume_sources = (archive,)
                    def resumed(*args,**kwargs):
                        calls.append((args[8],args[7]))
                        return train_weighted_epoch(*args,**kwargs)
                    with patch('src.task_b_training.train_weighted_epoch',side_effect=resumed):
                        run_llm_task(task,config,resume_sources=resume_sources,auto_resume=False)
                    self.assertEqual(calls[0],(fail_phase,2))
                    if fail_phase.startswith('Full'):
                        self.assertTrue(all(name.startswith('Full') for name,_ in calls))
                    resumed_scores=[]
                    for seed in task.selection_seeds:
                        with np.load(result_root/'outputs'/task.output_subdir/f'seed_{seed}'/'prediction_scores.npz') as values:
                            resumed_scores.append(values['probabilities'].copy())
                    with tempfile.TemporaryDirectory() as baseline_directory, ExitStack() as baseline_stack:
                        baseline=Path(baseline_directory)
                        self.fixture.write_data(baseline,task,gold=False)
                        self.fixture.boundary(baseline_stack,baseline)
                        run_llm_task(task,config,auto_resume=False)
                        for i,seed in enumerate(task.selection_seeds):
                            with np.load(baseline/'outputs'/task.output_subdir/f'seed_{seed}'/'prediction_scores.npz') as values:
                                np.testing.assert_allclose(resumed_scores[i],values['probabilities'],atol=1e-8,rtol=0)


if __name__ == '__main__':
    unittest.main()
