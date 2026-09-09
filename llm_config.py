"""QLoRA settings; the original BERT configuration and entry points stay intact."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from config import TASK_A, TASK_B, TRAINING, PreprocessConfig, TaskConfig


@dataclass(frozen=True)
class LLMTrainingConfig:
    """Defaults target Kaggle T4 x2 (FP16 compute, one 4-bit model per GPU)."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "HASTIKA_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"
        )
    )
    model_revision: str = "main"
    validation_size: float = TRAINING.validation_size
    seed: int = TRAINING.seed
    # Match the BERT text preprocessing by default for a controlled comparison.
    text_mode: str = "paper"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    max_length: int = 256  # Full chat, including instructions, answer and EOS.
    max_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    train_batch_size_per_gpu: int = 1
    # Number of candidate completions per forward, NOT number of comments.
    eval_batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    gradient_checkpointing: bool = True
    compute_dtype: str = "auto"  # BF16 if supported; FP16 on T4.
    initial_loss_scale: float = 1024.0  # Conservative FP16 start; GradScaler adapts.
    num_workers: int = 0
    use_all_available_gpus: bool = True
    max_gpus: int = 2


def llm_task(task: TaskConfig) -> TaskConfig:
    """Reuse label order/input files; isolate all LLM artifacts from BERT."""

    return replace(
        task,
        checkpoint_subdir=f"{task.checkpoint_subdir}_llm",
        output_subdir=f"{task.output_subdir}_llm",
    )


LLM_TRAINING = LLMTrainingConfig()
LLM_TRAINING_B = replace(
    LLM_TRAINING,
    train_batch_size_per_gpu=4,
    gradient_accumulation_steps=2,
)
TASK_A_LLM = llm_task(TASK_A)
TASK_B_LLM = llm_task(TASK_B)
