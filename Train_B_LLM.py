"""Select the best epoch per Task B seed, full-data retrain, and soft ensemble."""

from llm_config import LLM_TRAINING_B, TASK_B_LLM
from src.llm_training import run_llm_cli


def main() -> None:
    run_llm_cli(TASK_B_LLM, LLM_TRAINING_B)


if __name__ == "__main__":
    main()
