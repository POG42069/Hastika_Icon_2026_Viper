"""Order experiments preserve instructions and cannot resume the opposite order."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from llm_config import TASK_B, LLM_TRAINING_B, with_prompt_order
from src.llm_data import ChatEncoder
from src.llm_training import _resume_signature, run_llm_cli, run_llm_task
from src.resume import ResumeCatalog
from src.task_b_training import _finished_seed
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime
from tests import test_task_b_khangtest as fixtures


class PromptOrderTests(unittest.TestCase):
    def test_only_view_order_changes_and_labels_and_instructions_stay_identical(self):
        tokenizer = TestChatTokenizer()
        before = ChatEncoder(tokenizer, TASK_B, 700)
        after = ChatEncoder(tokenizer, with_prompt_order(TASK_B, 'english-first'), 700)
        pair = ('English! 😅 42', 'Original!! 😅 42')
        self.assertEqual(before.system_text, after.system_text)
        self.assertEqual(before.completions, after.completions)
        self.assertEqual(after.messages(pair)[1]['content'],
                         'English translation:\nEnglish! 😅 42\n\nComment (raw code-mixed):\nOriginal!! 😅 42')
        self.assertEqual(before.messages(pair)[1]['content'],
                         'Comment (raw code-mixed):\nOriginal!! 😅 42\n\nEnglish translation:\nEnglish! 😅 42')
        for encoder in (before, after):
            prompt, cut = encoder.encode_prompt(pair)
            self.assertFalse(cut)
            feature = encoder.feature(prompt, 4)
            self.assertEqual(feature['labels'][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(feature['labels'][len(prompt):], encoder.completions[4])

    def test_long_comments_keep_both_views_and_complete_answer_in_either_order(self):
        for order in ('english-first', 'original-first'):
            encoder = ChatEncoder(TestChatTokenizer(), with_prompt_order(TASK_B, order), 700)
            prompt, cut = encoder.encode_prompt(('EN_START ' + 'english ' * 300 + ' EN_END',
                                                  'RAW_START ' + 'original ' * 300 + ' RAW_END'))
            self.assertTrue(cut)
            body = encoder.tokenizer.decode(prompt)
            for fragment in ('EN_START', 'EN_END', 'RAW_START', 'RAW_END', '[...]'):
                self.assertIn(fragment, body)
            self.assertLessEqual(len(prompt) + encoder.response_reserve, 700)

    def test_default_order_keeps_resume_compatibility_and_reversed_order_rejects_old_seed(self):
        with patch('src.llm_training.labeled_input_filenames', return_value=TASK_B.original_fold_filenames):
            original = _resume_signature(TASK_B, LLM_TRAINING_B, 'same-data', cpu_runtime())
            explicit_original = _resume_signature(with_prompt_order(TASK_B, 'original-first'), LLM_TRAINING_B, 'same-data', cpu_runtime())
            reversed_order = _resume_signature(with_prompt_order(TASK_B, 'english-first'), LLM_TRAINING_B, 'same-data', cpu_runtime())
        self.assertEqual(original, explicit_original)
        self.assertNotEqual(original, reversed_order)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs' / 'seed_42'
            adapter = root / 'checkpoints' / 'seed_42' / 'final_full_data_adapter'
            output.mkdir(parents=True)
            adapter.mkdir(parents=True)
            (output / 'fixed_task_b_summary.json').write_text(json.dumps({'resume_signature': original}))
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'fixture')
            catalogs = [ResumeCatalog('prior run', output.parent, adapter.parent.parent)]
            self.assertIsNotNone(_finished_seed(catalogs, 42, explicit_original))
            self.assertIsNone(_finished_seed(catalogs, 42, reversed_order))

    def test_cli_selects_order_and_isolates_outputs_without_changing_training_config(self):
        with patch('sys.argv', ['Train_B_LLM.py', '--prompt-order', 'english-first']), patch(
            'src.llm_training.relaunch_with_torchrun_if_needed'
        ), patch('src.llm_training.run_llm_task') as runner:
            run_llm_cli(TASK_B, LLM_TRAINING_B)
        task, config = runner.call_args.args[:2]
        self.assertEqual(task.prompt_order, 'english-first')
        self.assertEqual(task.prompt_style, TASK_B.prompt_style)
        self.assertTrue(task.output_subdir.endswith('_english_first'))
        self.assertTrue(task.checkpoint_subdir.endswith('_english_first'))
        self.assertEqual(asdict(config), asdict(LLM_TRAINING_B))
        self.assertEqual(with_prompt_order(TASK_B, 'original-first').output_subdir, TASK_B.output_subdir)

    def test_reversed_full_pipeline_records_order_and_uses_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            helper = fixtures.TaskBKhangtestTests()
            task = with_prompt_order(replace(TASK_B, fixed_full_retrain_epochs=1, selection_seeds=(42,52)), 'english-first')
            helper.write_data(root, task, gold=False)
            helper.boundary(stack, root)
            run_llm_task(task, LLM_TRAINING_B, auto_resume=False)
            summary = json.loads((root / 'outputs' / task.output_subdir / 'run_summary.json').read_text())
            self.assertEqual(summary['prompt_input']['prompt_order'], 'english-first')
            self.assertEqual(summary['resume_signature']['prompt_order'], 'english-first')
            self.assertFalse((root / 'outputs' / TASK_B.output_subdir).exists())


if __name__ == '__main__':
    unittest.main()
