"""Train and ensemble Task A with one command.

Usage:
    python Train_A.py

The script performs stratified 5-fold fine-tuning, selects the best checkpoint
from each fold by validation Macro-F1, ensembles probabilities, and writes a
CodaBench-ready ``predictions.csv`` plus ``task_a_submission.zip``.
"""

from config import TASK_A, TRAINING
from src.distributed import relaunch_with_torchrun_if_needed
from src.training import run_cross_validated_task


def main() -> None:
    """Run the complete Task A pipeline."""

    relaunch_with_torchrun_if_needed(TRAINING)
    run_cross_validated_task(TASK_A, TRAINING)


if __name__ == "__main__":
    main()
