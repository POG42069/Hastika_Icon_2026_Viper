"""Train and predict Task B with one command.

Usage:
    python Train_B.py

The script creates one stratified 80/20 holdout, fine-tunes BERT with standard
cross-entropy, selects one checkpoint by validation Macro-F1, and writes a
CodaBench-ready submission for the six target classes.
"""

from config import TASK_B, TRAINING
from src.distributed import relaunch_with_torchrun_if_needed
from src.training import run_holdout_task


def main() -> None:
    """Run the complete Task B pipeline."""

    relaunch_with_torchrun_if_needed(TRAINING)
    run_holdout_task(TASK_B, TRAINING)


if __name__ == "__main__":
    main()
