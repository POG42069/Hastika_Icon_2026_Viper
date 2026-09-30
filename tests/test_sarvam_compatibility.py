"""Offline checks for Sarvam's native template and Llama answer-only loss.

The fixture is the public chat_template from sarvamai/sarvam-1's
tokenizer_config.json. Tests use a byte tokenizer and a tiny random Llama;
they never download pretrained weights or read competition data.
"""

from pathlib import Path
import unittest

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from llm_config import TASK_B
from src.llm_data import ChatEncoder, CompletionCollator
from src.llm_training import completion_log_probs


def native_template_tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab.setdefault(character, len(vocab))
    backend = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    template = Path(__file__).with_name("fixtures").joinpath(
        "sarvam_chat_template.jinja"
    ).read_text(encoding="utf-8")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        chat_template=template,
        padding_side="left",
    )


class SarvamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tokenizer = native_template_tokenizer()
        self.encoder = ChatEncoder(self.tokenizer, TASK_B, 512)

    def test_native_template_matches_full_chat_for_every_label(self):
        text = "A short code-mixed comment."
        prompt, truncated = self.encoder.encode_prompt(text)
        self.assertFalse(truncated)
        rendered = self.tokenizer.decode(prompt)
        self.assertIn("<<SYS>>", rendered)
        self.assertIn("[INST]", rendered)
        self.assertIn("[/INST]", rendered)
        for label_id, label in enumerate(TASK_B.labels):
            with self.subTest(label=label):
                feature = self.encoder.feature(prompt, label_id)
                expected = self.tokenizer.apply_chat_template(
                    self.encoder.messages(text)
                    + [{"role": "assistant", "content": label}],
                    tokenize=True,
                )
                self.assertEqual(feature["input_ids"], expected)
                self.assertEqual(feature["labels"][:len(prompt)], [-100] * len(prompt))
                self.assertIn(self.tokenizer.eos_token_id, self.encoder.completions[label_id])
                # Sarvam's template includes a newline AFTER EOS as well.
                self.assertIn(label, self.tokenizer.decode(self.encoder.completions[label_id]))

    def test_truncation_preserves_all_native_answers(self):
        prompt, truncated = self.encoder.encode_prompt("long comment " * 500)
        self.assertTrue(truncated)
        for label_id in range(len(TASK_B.labels)):
            feature = self.encoder.feature(prompt, label_id)
            self.assertLessEqual(len(feature["input_ids"]), 512)
            self.assertEqual(feature["labels"][len(prompt):], self.encoder.completions[label_id])

    def test_lora_llama_suffix_loss_matches_full_loss_with_eos_padding(self):
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        model = get_peft_model(
            LlamaForCausalLM(config),
            LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                       target_modules="all-linear", lora_dropout=0.0),
        ).eval()
        features = []
        for text, label_id in (("short", 0), ("a longer comment", 3)):
            prompt, _ = self.encoder.encode_prompt(text)
            features.append(self.encoder.feature(prompt, label_id))
        batch = CompletionCollator(self.tokenizer.pad_token_id)(features)
        with torch.no_grad():
            scores, counts = completion_log_probs(model, batch)
            reference = model(**batch).loss
        self.assertTrue(torch.allclose(-scores.sum() / counts.sum(), reference, atol=1e-6))
        self.assertEqual(counts.tolist(), [len(self.encoder.completions[i]) for i in (0, 3)])
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        self.assertTrue(all("lora_" in name for name, p in model.named_parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
