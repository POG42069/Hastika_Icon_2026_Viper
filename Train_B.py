"""Train and ensemble Task B with one command.

Usage:
    python Train_B.py

The script performs stratified 5-fold fine-tuning with fold-specific class
weights, selects checkpoints by validation Macro-F1, ensembles probabilities,
and writes a CodaBench-ready submission for the six target classes.
"""

from config import TASK_B, TRAINING
from src.distributed import relaunch_with_torchrun_if_needed
from src.training import run_cross_validated_task


def main() -> None:
    """Run the complete Task B pipeline."""

    relaunch_with_torchrun_if_needed(TRAINING)
    run_cross_validated_task(TASK_B, TRAINING)


if __name__ == "__main__":
    main()
