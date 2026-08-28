"""Train and predict Task A with one command.

Usage:
    python Train_A.py

The script creates one stratified 80/20 holdout, fine-tunes BERT on the 80%,
selects one checkpoint by validation Macro-F1, and writes a CodaBench-ready
``predictions.csv`` plus ``task_a_submission.zip``.
"""

from config import TASK_A, TRAINING
from src.distributed import relaunch_with_torchrun_if_needed
from src.training import run_holdout_task


def main() -> None:
    """Run the complete Task A pipeline."""

    relaunch_with_torchrun_if_needed(TRAINING)
    run_holdout_task(TASK_A, TRAINING)


if __name__ == "__main__":
    main()
