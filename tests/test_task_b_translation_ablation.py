"""Task B translation ablation uses identical training with isolated prompts."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from llm_config import TASK_A, TASK_B, LLM_TRAINING_B, TASK_B_INCLUDE_ENGLISH_TRANSLATION
from src.llm_data import ChatEncoder
from src.llm_training import _resume_signature, run_llm_cli, run_llm_task
from src.task_b_training import score_candidates, train_weighted_epoch, _finished_seed
from src.resume import ResumeCatalog
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime
from tests import test_task_b_khangtest as fixtures


class TaskBTranslationAblationTests(unittest.TestCase):
    def task(self, include):
        return replace(TASK_B, include_english_translation=include)

    def test_original_only_uses_the_raw_field_and_keeps_the_same_instructions_and_answers(self):
        raw = ChatEncoder(TestChatTokenizer(), self.task(False), 700)
        paired = ChatEncoder(TestChatTokenizer(), self.task(True), 700)
        pair = ('TRANSLATION_SENTINEL', 'RAW_SOURCE_SENTINEL')
        self.assertEqual(raw.messages(pair)[1]['content'],
                         'Comment (raw code-mixed):\nRAW_SOURCE_SENTINEL')
        self.assertEqual(paired.messages(pair)[1]['content'],
                         'English translation:\nTRANSLATION_SENTINEL\n\nComment (raw code-mixed):\nRAW_SOURCE_SENTINEL')
        self.assertEqual(raw.system_text, paired.system_text)
        self.assertEqual(raw.completions, paired.completions)
        self.assertEqual(raw.encode_prompt(pair), raw.encode_prompt(('changed ' * 1000, pair[1])))
        self.assertTrue(TASK_A.include_english_translation)

    def test_original_only_truncation_ignores_english_and_preserves_raw_head_tail_and_label_loss(self):
        encoder = ChatEncoder(TestChatTokenizer(), self.task(False), 512)
        original = 'RAW_START ' + 'original ' * 500 + ' RAW_END'
        prompt, truncated = encoder.encode_prompt(('ENGLISH_SENTINEL ' * 1000, original))
        self.assertTrue(truncated)
        self.assertEqual((prompt, truncated), encoder.encode_prompt(('', original)))
        decoded = encoder.tokenizer.decode(prompt)
        for part in ('Comment (raw code-mixed):', 'RAW_START', 'RAW_END', '[...]'):
            self.assertIn(part, decoded)
        self.assertNotIn('ENGLISH_SENTINEL', decoded)
        self.assertNotIn('English translation:', decoded)
        for label in range(len(TASK_B.labels)):
            feature = encoder.feature(prompt, label)
            self.assertLessEqual(len(feature['input_ids']), 512)
            self.assertEqual(feature['labels'][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(feature['labels'][len(prompt):], encoder.completions[label])

    def test_ignored_english_never_causes_a_short_original_to_be_truncated(self):
        encoder = ChatEncoder(TestChatTokenizer(), self.task(False), 512)
        prompt, truncated = encoder.encode_prompt(('English ' * 1000, 'short raw'))
        self.assertFalse(truncated)
        self.assertIn('short raw', encoder.tokenizer.decode(prompt))

    def test_resume_rejects_the_other_prompt_variant_even_with_the_same_data(self):
        with patch('src.llm_training.labeled_input_filenames', return_value=TASK_B.original_fold_filenames):
            raw = _resume_signature(self.task(False), LLM_TRAINING_B, 'same-data', cpu_runtime())
            paired = _resume_signature(self.task(True), LLM_TRAINING_B, 'same-data', cpu_runtime())
        raw['pipeline'] = paired['pipeline'] = 'holdout-selection-soft-v1'
        self.assertNotEqual(raw, paired)
        self.assertEqual(raw['prompt_policy'], 'original-only-head-tail-v1')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs' / 'seed_42'
            adapter = root / 'checkpoints' / 'seed_42' / 'final_full_data_adapter'
            output.mkdir(parents=True)
            adapter.mkdir(parents=True)
            (output / 'task_b_summary.json').write_text(json.dumps({'resume_signature': raw, 'epochs': 1}))
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'fixture')
            catalogs = [ResumeCatalog('raw run', output.parent, adapter.parent.parent)]
            self.assertIsNotNone(_finished_seed(catalogs, 42, raw, 1))
            self.assertIsNone(_finished_seed(catalogs, 42, paired, 1))

    def test_branch_default_and_cli_keep_the_nhotest_training_configuration(self):
        include = TASK_B_INCLUDE_ENGLISH_TRANSLATION
        self.assertEqual(TASK_B.include_english_translation, include)
        expected = 'khangtest3_bilingual_english_first_256' if include else 'khangtest2_original_only_256'
        self.assertTrue(TASK_B.output_subdir.endswith(expected))
        self.assertEqual(TASK_B.output_subdir, TASK_B.checkpoint_subdir)
        with patch('sys.argv', ['Train_B_LLM.py']), patch(
            'src.llm_training.relaunch_with_torchrun_if_needed'
        ), patch('src.llm_training.run_llm_task') as runner:
            run_llm_cli(TASK_B, LLM_TRAINING_B)
        task, config = runner.call_args.args[:2]
        self.assertEqual(task.include_english_translation, include)
        self.assertEqual(asdict(config), asdict(LLM_TRAINING_B))
        self.assertEqual(task.selection_seeds, (42, 52, 62, 72, 82))
        self.assertFalse(task.use_augmentation)

    def test_each_variant_reaches_training_holdout_official_validation_and_prediction(self):
        for include in (False, True):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                task = replace(self.task(include), selection_seeds=(42, 52))
                config = replace(LLM_TRAINING_B, max_length=512, max_epochs=1)
                helper = fixtures.TaskBKhangtestTests()
                helper.write_data(root, task, gold=True)
                helper.boundary(stack, root)
                phases = set()

                def check(data):
                    for prompt in data.prompts:
                        decoded = data.encoder.tokenizer.decode(prompt)
                        self.assertIn('Comment (raw code-mixed):', decoded)
                        self.assertIn('raw ', decoded)
                        self.assertEqual('English translation:' in decoded, include)
                        if not include:
                            self.assertNotIn('English dev', decoded)

                def train(*args, **kwargs):
                    phases.add(args[8].split(' seed')[0])
                    check(args[1].dataset)
                    return train_weighted_epoch(*args, **kwargs)

                def score(*args, **kwargs):
                    phases.add(args[5].split(' seed')[0])
                    check(args[1])
                    return score_candidates(*args, **kwargs)

                with patch('src.task_b_training.train_weighted_epoch', side_effect=train), patch(
                    'src.task_b_training.score_candidates', side_effect=score
                ):
                    run_llm_task(task, config, auto_resume=False)
                self.assertTrue({'Holdout train', 'Full-data retrain', 'Holdout validation',
                                 'Official validation |', 'Development prediction'}.issubset(phases))
                summary = json.loads((root / 'outputs' / task.output_subdir / 'run_summary.json').read_text())
                self.assertEqual(summary['prompt_input']['include_english_translation'], include)
                self.assertEqual(summary['prompt_input']['english_column'], 'Comment' if include else None)
                self.assertEqual(summary['ensemble']['seeds'], [42, 52])


if __name__ == '__main__':
    unittest.main()
