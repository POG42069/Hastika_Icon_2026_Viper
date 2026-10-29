"""Sarvam preset isolation and completion-loss compatibility with Llama models."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from llm_config import LLM_TRAINING, LLM_TRAINING_B, TASK_A, TASK_B
from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
from src.llm_training import _resume_signature, completion_log_probs
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime


class UTF8OffsetTokenizer(TestChatTokenizer):
    """Byte-fallback fixture with offsets, including multi-token emoji."""
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        return [value + 10 for value in text.encode('utf-8')]

    def __call__(self, text, **kwargs):
        offsets = [span for i, char in enumerate(text)
                   for span in [(i, i + 1)] * len(char.encode('utf-8'))]
        return {'input_ids': self.encode(text), 'offset_mapping': offsets}

    def decode(self, ids, skip_special_tokens=False):
        parts, buffer = [], bytearray()
        special = {0: '<pad>', 2: '<end>', 3: '<system>', 4: '<user>', 5: '<assistant>'}
        for value in ids:
            if value in special:
                parts.append(buffer.decode('utf-8', errors='replace'))
                buffer.clear()
                if not skip_special_tokens:
                    parts.append(special[value])
            else:
                buffer.append(value - 10)
        parts.append(buffer.decode('utf-8', errors='replace'))
        return ''.join(parts)


class SarvamTests(unittest.TestCase):
    def test_byte_fallback_truncation_preserves_unicode_and_complete_answers(self):
        for task in (TASK_A, TASK_B):
            encoder = ChatEncoder(UTF8OffsetTokenizer(), task, 700)
            pair = ('EN_START ' + '🎫🙏' * 100 + ' EN_END',
                    'RAW_START ' + '🙏🎫' * 100 + ' RAW_END')
            prompt, cut = encoder.encode_prompt(pair)
            self.assertTrue(cut)
            rendered = encoder.tokenizer.decode(prompt)
            self.assertNotIn('\ufffd', rendered)
            for marker in ['EN_START', 'EN_END', 'RAW_START', 'RAW_END', '[...]']:
                self.assertIn(marker, rendered)
            for label in range(len(task.labels)):
                feature = encoder.feature(prompt, label)
                self.assertLessEqual(len(feature['input_ids']), 700)
                self.assertEqual(feature['input_ids'][len(prompt):], encoder.completions[label])

    def test_preset_changes_backbone_and_isolates_qwen_adapters(self):
        for task, config, learning_rate, patience in [
            (TASK_A, LLM_TRAINING, 2e-4, 4),
            (TASK_B, LLM_TRAINING_B, 1e-4, 2),
        ]:
            self.assertEqual(config.model_name, 'sarvamai/sarvam-1')
            self.assertEqual(config.model_revision, 'e9607337286ddf496d4a2562b194e489dcf3feea')
            self.assertIn('sarvam_1_lora_khangtest4', task.output_subdir)
            self.assertEqual(task.output_subdir, task.checkpoint_subdir)
            self.assertTrue(task.include_english_translation)
            self.assertEqual(task.fixed_prompt_order, 'english-first')
            self.assertFalse(task.use_augmentation)
            self.assertEqual(config.max_length, 256)
            self.assertEqual(config.learning_rate, learning_rate)
            self.assertEqual(config.early_stopping_patience, patience)
            self.assertEqual(task.selection_seeds, (42, 52, 62, 72, 82))
            with patch('src.llm_training.labeled_input_filenames', return_value=task.original_fold_filenames):
                sarvam = _resume_signature(task, config, 'same-data', cpu_runtime())
                qwen = _resume_signature(task, replace(
                    config, model_name='Qwen/Qwen3-4B-Instruct-2507',
                    model_revision='cdbee75f17c01a7cc42f958dc650907174af0554'),
                    'same-data', cpu_runtime())
            self.assertNotEqual(sarvam, qwen)
            self.assertEqual(sarvam['truncation_serialization'], 'unicode-safe-offsets-v1')

    def test_llama_completion_loss_matches_native_loss_and_trains_only_lora(self):
        torch.manual_seed(42)
        model = get_peft_model(
            LlamaForCausalLM(LlamaConfig(
                vocab_size=256, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=1024,
                pad_token_id=0, eos_token_id=2, use_cache=False,
                attention_dropout=0.0,
            )),
            LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                       lora_dropout=0.0, target_modules='all-linear', bias='none'),
        ).eval()
        for task in (TASK_A, TASK_B):
            encoder = ChatEncoder(TestChatTokenizer(), task, 700)
            data = CompletionDataset(
                [('English one', 'Raw one'), ('A longer English comment', 'Raw two')],
                encoder, [0, 1],
            )
            batch = CompletionCollator(0)([data[0], data[1]])
            scores, counts = completion_log_probs(model, batch)
            native_loss = model(**batch).loss
            self.assertTrue(torch.allclose(-scores.sum() / counts.sum(), native_loss, atol=1e-6))
            self.assertTrue(torch.all(batch['labels'][batch['attention_mask'] == 0] == -100))
            model.zero_grad(set_to_none=True)
            (-scores / counts).mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for name, p in model.named_parameters() if 'lora_' in name))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))


if __name__ == '__main__':
    unittest.main()
