"""Real CUDA/LoRA smoke test with tiny random Qwen2 weights (no download).

Run manually on a CUDA machine after installing requirements.txt. This exercises
the production model loader, gradient checkpointing, FP16 backward, legal
label scoring and adapter save/reload without claiming pretrained-model quality.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_config import LLMTrainingConfig, TASK_A, TASK_B
from src.distributed import RuntimeContext
from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
from src.llm_training import (
    load_lora_model,
    restore_adapter,
    save_adapter,
    score_candidates,
    train_llm_epoch,
)
from tests.test_llm_pipeline import TestChatTokenizer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This smoke test needs a CUDA-enabled PyTorch build and GPU."
        )
    torch.set_num_threads(1)
    torch.manual_seed(42)
    runtime = RuntimeContext(torch.device("cuda", 0), 0, 0, 1, False)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        backbone = root / "base"
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                pad_token_id=0,
                eos_token_id=2,
            )
        ).save_pretrained(backbone)
        config = replace(
            LLMTrainingConfig(),
            model_name=str(backbone),
            max_length=512,
            lora_rank=4,
            gradient_accumulation_steps=2,
            compute_dtype="float16",
        )
        for task in (TASK_A, TASK_B):
            model = load_lora_model(config, runtime)
            assert not getattr(model, "is_loaded_in_4bit", False)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            tokenizer = TestChatTokenizer()
            encoder = ChatEncoder(tokenizer, task, 512)
            collator = CompletionCollator(0)
            data = CompletionDataset([("one", "raw one"), ("two", "raw two"), ("three", "raw three")], encoder, [0, 1, 0])
            loader = DataLoader(data, batch_size=1, collate_fn=collator)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            train_llm_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                torch.amp.GradScaler("cuda", init_scale=config.initial_loss_scale),
                config,
                runtime,
                1,
            )
            changed = [
                name
                for name, p in model.named_parameters()
                if not torch.equal(before[name], p)
            ]
            assert changed and all("lora_" in name for name in changed), changed
            expected = score_candidates(
                model, data, collator, config, runtime, "CUDA eval"
            )
            assert np.isfinite(expected["loss"])
            adapter = root / task.checkpoint_subdir
            save_adapter(model, tokenizer, adapter)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        parameter.zero_()
            restore_adapter(model, adapter)
            actual = score_candidates(
                model, data, collator, config, runtime, "CUDA reload"
            )
            np.testing.assert_allclose(
                actual["probabilities"], expected["probabilities"], atol=1e-6
            )
            del model, optimizer, scheduler, before
            torch.cuda.empty_cache()
    print("LLM_CUDA_LORA_SMOKE_OK")


if __name__ == "__main__":
    main()
