"""Central configuration for both HASTIKA training pipelines.

Edit this file when changing epochs, learning rate, per-GPU batch size,
preprocessing options, paths, or model settings.  The two entry points do not
contain hidden training hyperparameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The repository root is used for local paths and Kaggle output files.
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PreprocessConfig:
    """Options for conservative social-media text normalization."""

    fix_unicode: bool = True
    decode_html_entities: bool = True
    replace_urls: bool = True
    replace_mentions: bool = True
    keep_hashtag_text: bool = True
    normalize_repeated_characters: bool = True
    max_repeated_characters: int = 2
    lowercase: bool = False
    remove_stopwords: bool = False
    lemmatize: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters shared by Task A and Task B."""

    model_name: str = "google/muril-base-cased"
    num_folds: int = 5
    max_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_length: int = 128

    # This value is the batch size handled by EACH GPU.  On Kaggle T4 x2,
    # the effective batch is 8 x 2 = 16 before gradient accumulation.
    train_batch_size_per_gpu: int = 8
    eval_batch_size_per_gpu: int = 16
    gradient_accumulation_steps: int = 1

    early_stopping_patience: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    use_amp: bool = True
    pad_to_multiple_of: int = 8

    # Running ``python Train_A.py`` or ``python Train_B.py`` automatically
    # relaunches through torchrun when two GPUs are visible.
    use_all_available_gpus: bool = True
    max_gpus: int = 2

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific file names, labels and output locations."""

    task_name: str
    train_filename: str
    test_filename: str
    label_column: str
    labels: tuple[str, ...]
    use_class_weights: bool
    checkpoint_subdir: str
    output_subdir: str
    submission_zip_name: str


# ``HASTIKA_DATA_DIR`` is useful when the Kaggle Dataset slug differs from the
# default.  The loader also searches every /kaggle/input child automatically.
DATA_DIR = Path(os.getenv("HASTIKA_DATA_DIR", str(PROJECT_ROOT / "data")))
CHECKPOINT_ROOT = Path(
    os.getenv("HASTIKA_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))
)
OUTPUT_ROOT = Path(os.getenv("HASTIKA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

TRAINING = TrainingConfig()

TASK_A = TaskConfig(
    task_name="Task A - Binary Hate Speech Detection",
    train_filename="binary_train.csv",
    test_filename="binary_validation_inputs.csv",
    label_column="Label",
    labels=("Non-Hate", "Hate"),
    use_class_weights=False,
    checkpoint_subdir="task_a",
    output_subdir="task_a",
    submission_zip_name="task_a_submission.zip",
)

TASK_B = TaskConfig(
    task_name="Task B - Fine-Grained Hate Speech Classification",
    train_filename="multiclass_train.csv",
    test_filename="multiclass_validation_inputs.csv",
    label_column="Hate Category",
    labels=(
        "Gender",
        "Political",
        "Religion",
        "Geo-political",
        "Violence",
        "Others",
    ),
    use_class_weights=True,
    checkpoint_subdir="task_b",
    output_subdir="task_b",
    submission_zip_name="task_b_submission.zip",
)
