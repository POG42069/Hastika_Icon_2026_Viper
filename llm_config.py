"""Configuration for the lightly normalized LoRA HASTIKA pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("HASTIKA_DATA_DIR", str(PROJECT_ROOT / "data")))
CHECKPOINT_ROOT = Path(
    os.getenv("HASTIKA_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))
)
OUTPUT_ROOT = Path(os.getenv("HASTIKA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific input names, labels and artifact directories."""

    task_name: str
    train_filename: str
    prediction_filename: str
    label_column: str
    labels: tuple[str, ...]
    checkpoint_subdir: str
    output_subdir: str
    submission_zip_name: str
    original_fold_filenames: tuple[str, ...] = ()
    augmentation_fold_filenames: tuple[str, ...] = ()
    use_augmentation: bool = True
    selection_validation_fold: int | None = None
    selection_validation_data_origins: tuple[str, ...] = ()
    selection_train_label_caps: tuple[tuple[str, int], ...] = ()
    selection_seeds: tuple[int, ...] = ()
    direct_full_train_epochs: int | None = None
    report_per_class_metrics: bool = False

    def labeled_filenames(self) -> tuple[str, ...]:
        """Return every labeled CSV used by this task."""

        if self.original_fold_filenames:
            augmentation_files = (
                self.augmentation_fold_filenames if self.use_augmentation else ()
            )
            return self.original_fold_filenames + augmentation_files
        return (self.train_filename,)


@dataclass(frozen=True)
class LLMTrainingConfig:
    """LoRA defaults targeting Kaggle T4 x2 with FP16 compute."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "HASTIKA_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"
        )
    )
    model_revision: str = "main"
    validation_size: float = 0.20
    seed: int = 42
    max_length: int = 256  # Full chat, including instructions, answer and EOS.
    max_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    train_batch_size_per_gpu: int = 4
    # Number of candidate completions per forward, NOT number of comments.
    eval_batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    gradient_checkpointing: bool = True
    compute_dtype: str = "auto"  # BF16 if supported; FP16 on T4.
    initial_loss_scale: float = 1024.0  # Conservative FP16 start; GradScaler adapts.
    progress_log_percent: int = 10  # Stable Kaggle log line every N percent.
    num_workers: int = 0
    use_all_available_gpus: bool = True
    max_gpus: int = 2


TASK_A = TaskConfig(
    task_name="Task A - Binary Hate Speech Detection",
    train_filename="binary_train.csv",
    prediction_filename="binary_validation_inputs.csv",
    label_column="Label",
    labels=("Non-Hate", "Hate"),
    checkpoint_subdir="task_a_lora",
    output_subdir="task_a_lora",
    submission_zip_name="task_a_submission.zip",
)

TASK_B = TaskConfig(
    task_name="Task B - Fine-Grained Hate Speech Classification",
    train_filename="multiclass_train.csv",
    prediction_filename="multiclass_validation_inputs.csv",
    label_column="Hate Category",
    labels=(
        "Gender",
        "Political",
        "Religion",
        "Geo-political",
        "Violence",
        "Others",
    ),
    checkpoint_subdir="task_b_lora",
    output_subdir="task_b_lora",
    submission_zip_name="task_b_submission.zip",
    original_fold_filenames=tuple(f"original_{fold}.csv" for fold in range(1, 6)),
    augmentation_fold_filenames=tuple(f"augment{fold}.csv" for fold in range(1, 6)),
    use_augmentation=True,
    selection_validation_fold=5,
    selection_validation_data_origins=("original",),
    selection_train_label_caps=(("Gender", 870),),
    selection_seeds=(52,),
    direct_full_train_epochs=3,
    report_per_class_metrics=True,
)


def llm_task(task: TaskConfig) -> TaskConfig:
    """Compatibility helper retained for callers that already use it."""

    return task


LLM_TRAINING = LLMTrainingConfig()
# Task B keeps the same T4 x2 hyperparameters as the restored 03c7ca8 pipeline.
LLM_TRAINING_B = replace(LLM_TRAINING)
TASK_A_LLM = TASK_A
TASK_B_LLM = TASK_B
