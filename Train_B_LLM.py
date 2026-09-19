"""Reproduce the recorded Task B 0.66 selection/retrain pipeline."""

from llm_config import LLM_TRAINING_B, TASK_B_LLM
from src.llm_training import run_llm_cli


def main() -> None:
    run_llm_cli(TASK_B_LLM, LLM_TRAINING_B)


if __name__ == "__main__":
    main()
