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
    # Answers emitted by the model, in the same order as the official labels.
    label_codes: tuple[str, ...]
    checkpoint_subdir: str
    output_subdir: str
    submission_zip_name: str
    original_fold_filenames: tuple[str, ...] = ()
    augmentation_fold_filenames: tuple[str, ...] = ()
    compact_train_filename: str | None = None
    compact_validation_filename: str | None = None
    use_augmentation: bool = True
    input_mode: str = "english"
    original_comment_column: str = "OriginalComment"
    prompt_style: str = "paired"
    fixed_full_retrain_epochs: int | None = None
    full_retrain_soft_ensemble: bool = False
    class_weight_power: float | None = None
    class_weight_min: float = 0.75
    class_weight_max: float = 1.50
    epoch_validation_filename: str | None = None
    epoch_validation_input_filename: str | None = None
    epoch_validation_required: bool = False
    selection_validation_fold: int | None = None
    selection_validation_data_origins: tuple[str, ...] = ()
    selection_train_label_caps: tuple[tuple[str, int], ...] = ()
    selection_seeds: tuple[int, ...] = ()
    ensemble_all_seeds: bool = False
    report_per_class_metrics: bool = False
    enable_augmentation_audit: bool = True
    augmentation_audit_action: str = "report"
    augmentation_near_duplicate_threshold: float = 0.92
    augmentation_min_source_similarity: float = 0.20

    def labeled_filenames(self) -> tuple[str, ...]:
        """Return the preferred labeled CSV names for this task."""

        if (
            not self.use_augmentation
            and self.fixed_full_retrain_epochs is None
            and self.compact_train_filename
            and self.compact_validation_filename
        ):
            return (self.compact_train_filename, self.compact_validation_filename)
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
            "HASTIKA_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
        )
    )
    # Exact Hugging Face revision for the non-thinking Qwen3 Instruct release.
    model_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    validation_size: float = 0.20
    seed: int = 42
    max_length: int = 512  # Full chat, including instructions, answer and EOS.
    max_epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    train_batch_size_per_gpu: int = 4
    # Number of candidate completions per forward, NOT number of comments.
    eval_batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 4
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
    label_codes=("Non-Hate", "Hate"),
    checkpoint_subdir="task_a_qwen3_4b_2507_lora_bilingual",
    output_subdir="task_a_qwen3_4b_2507_lora_bilingual",
    submission_zip_name="task_a_submission.zip",
    original_fold_filenames=tuple(
        f"binary_original_{fold}.csv" for fold in range(1, 6)
    ),
    augmentation_fold_filenames=tuple(
        f"binary_augment{fold}.csv" for fold in range(1, 6)
    ),
    compact_train_filename="binary_original_1_4.csv",
    compact_validation_filename="binary_original_5.csv",
    use_augmentation=False,
    input_mode="bilingual",
    selection_validation_fold=5,
    selection_validation_data_origins=("original",),
    selection_seeds=(42, 52, 62, 72, 82),
    ensemble_all_seeds=True,
    report_per_class_metrics=True,
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
    # Task B continues to train and score the official category names directly.
    label_codes=(
        "Gender",
        "Political",
        "Religion",
        "Geo-political",
        "Violence",
        "Others",
    ),
    checkpoint_subdir="task_b_qwen3_4b_2507_lora_bilingual_fixed_soft",
    output_subdir="task_b_qwen3_4b_2507_lora_bilingual_fixed_soft",
    submission_zip_name="task_b_submission.zip",
    original_fold_filenames=tuple(f"original_{fold}.csv" for fold in range(1, 6)),
    augmentation_fold_filenames=tuple(f"augment{fold}.csv" for fold in range(1, 6)),
    compact_train_filename="original_1_4.csv",
    compact_validation_filename="original_5.csv",
    use_augmentation=False,
    input_mode="bilingual",
    prompt_style="khangtest",
    selection_validation_fold=None,
    selection_validation_data_origins=("original",),
    selection_seeds=(42, 52, 62, 72, 82),
    ensemble_all_seeds=False,
    fixed_full_retrain_epochs=5,
    full_retrain_soft_ensemble=True,
    class_weight_power=0.50,
    class_weight_min=0.75,
    class_weight_max=1.50,
    report_per_class_metrics=True,
    epoch_validation_filename="hastika_multiclass_validation.csv",
    epoch_validation_input_filename="multiclass_validation_inputs.csv",
    enable_augmentation_audit=True,
    augmentation_audit_action="report",
)


def llm_task(task: TaskConfig) -> TaskConfig:
    """Compatibility helper retained for callers that already use it."""

    return task


LLM_TRAINING = LLMTrainingConfig()
# Match Khangtest Task B optimization, retaining the pinned Qwen3 and 512 tokens.
LLM_TRAINING_B = replace(LLM_TRAINING, learning_rate=1e-4, early_stopping_patience=2)
TASK_A_LLM = TASK_A
TASK_B_LLM = TASK_B
