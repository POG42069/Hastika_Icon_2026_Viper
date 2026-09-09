"""Fine-tune Qwen with QLoRA and export a Task A CodaBench submission."""

from llm_config import TASK_A_LLM
from src.llm_training import run_llm_cli


def main() -> None:
    run_llm_cli(TASK_A_LLM)


if __name__ == "__main__":
    main()
