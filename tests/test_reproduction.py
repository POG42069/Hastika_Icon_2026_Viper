"""Tests for the read-only Task B 0.66 reproduction preflight."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from llm_config import TASK_B, LLMTrainingConfig
from src.distributed import RuntimeContext
from src.reproduction import (
    verify_reproduction_manifest,
    verify_reproduction_model,
    verify_reproduction_tokenization,
)


class ReproductionManifestTests(unittest.TestCase):
    def test_matching_manifest_passes_and_input_change_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            input_path.write_bytes(b"stable-input\n")
            manifest_path = root / "manifest.json"
            manifest = {
                "model": {"name": "model", "resolved_revision": "revision"},
                "runtime": {
                    "world_size": 2,
                    "compute_dtype": "torch.float16",
                    "packages": {"example-package": "1.2.3"},
                },
                "task_config": {"selection_seeds": [42, 52]},
                "training_config": {
                    "max_length": 256,
                    "use_all_available_gpus": True,
                },
                "system_prompt": "prompt",
                "completion_token_ids": [[1, 2]],
                "parameters": {"trainable": 10, "reported_total": 100},
                "input_sha256": {
                    "input.csv": hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            task = replace(
                TASK_B,
                selection_seeds=(42, 52),
                reproduction_manifest=str(manifest_path),
            )
            config = replace(
                LLMTrainingConfig(),
                model_name="model",
                model_revision="revision",
            )
            runtime = RuntimeContext(torch.device("cpu"), 0, 0, 2, True)

            with (
                patch("src.reproduction.version", return_value="1.2.3"),
                patch(
                    "src.reproduction.resolve_data_file", return_value=input_path
                ),
            ):
                result = verify_reproduction_manifest(
                    task, config, runtime, "torch.float16"
                )
                self.assertEqual(result["model"]["resolved_revision"], "revision")
                verify_reproduction_tokenization(result, "prompt", [[1, 2]])
                verify_reproduction_model(
                    result,
                    "revision",
                    {"trainable": 10, "reported_total": 100},
                )

                input_path.write_bytes(b"changed-input\n")
                with self.assertRaisesRegex(RuntimeError, "input input.csv"):
                    verify_reproduction_manifest(
                        task, config, runtime, "torch.float16"
                    )


if __name__ == "__main__":
    unittest.main()
