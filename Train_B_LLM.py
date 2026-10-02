"""Train Qwen3-4B-Instruct-2507 on English Task B originals, five epochs per seed."""

from llm_config import LLM_TRAINING_B, TASK_B_LLM
from src.llm_training import run_llm_cli


def main() -> None:
    run_llm_cli(TASK_B_LLM, LLM_TRAINING_B)


if __name__ == "__main__":
    main()
