"""Read-only preflight checks for recorded competition runs."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.data import resolve_data_file


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def verify_reproduction_manifest(task, config, runtime, actual_dtype: str) -> dict:
    """Fail before training when the recorded Task B setup does not match."""

    if task.reproduction_manifest is None:
        return {}
    path = Path(task.reproduction_manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    model = manifest["model"]
    if config.model_name != model["name"]:
        mismatches.append(
            f"model_name: expected {model['name']!r}, got {config.model_name!r}"
        )
    if config.model_revision != model["resolved_revision"]:
        mismatches.append(
            "model_revision: expected the recorded resolved revision "
            f"{model['resolved_revision']!r}, got {config.model_revision!r}"
        )

    for field, expected in manifest["task_config"].items():
        actual = _json_value(getattr(task, field))
        if actual != expected:
            mismatches.append(f"task.{field}: expected {expected!r}, got {actual!r}")
    for field, expected in manifest["training_config"].items():
        actual = _json_value(getattr(config, field))
        if actual != expected:
            mismatches.append(f"config.{field}: expected {expected!r}, got {actual!r}")

    runtime_manifest = manifest["runtime"]
    if runtime.world_size != runtime_manifest["world_size"]:
        mismatches.append(
            f"world_size: expected {runtime_manifest['world_size']}, "
            f"got {runtime.world_size}"
        )
    if actual_dtype != runtime_manifest["compute_dtype"]:
        mismatches.append(
            f"compute_dtype: expected {runtime_manifest['compute_dtype']!r}, "
            f"got {actual_dtype!r}"
        )
    for package, expected in runtime_manifest["packages"].items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(
                f"package {package}: expected {expected!r}, got {actual!r}"
            )

    for filename, expected in manifest["input_sha256"].items():
        input_path = resolve_data_file(filename)
        actual = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(
                f"input {filename}: expected SHA-256 {expected}, got {actual}"
            )

    if mismatches:
        rendered = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Task B does not match the recorded 0.66 run:\n  - " + rendered
        )
    return manifest


def verify_reproduction_tokenization(
    manifest: dict, actual_system_prompt: str, actual_completions: list[list[int]]
) -> None:
    """Verify prompt text and the exact recorded Qwen completion token IDs."""

    if not manifest:
        return
    mismatches = []
    if actual_system_prompt != manifest["system_prompt"]:
        mismatches.append("the Task B system prompt changed")
    if actual_completions != manifest["completion_token_ids"]:
        mismatches.append(
            "completion token IDs changed; check tokenizer revision and packages"
        )
    if mismatches:
        raise RuntimeError(
            "Task B does not match the recorded 0.66 run: " + "; ".join(mismatches)
        )


def verify_reproduction_model(
    manifest: dict, resolved_revision: str | None, parameter_counts: dict[str, int]
) -> None:
    """Verify the loaded backbone revision and LoRA parameter topology."""

    if not manifest:
        return
    mismatches = []
    expected_revision = manifest["model"]["resolved_revision"]
    if resolved_revision != expected_revision:
        mismatches.append(
            f"resolved model revision expected {expected_revision!r}, "
            f"got {resolved_revision!r}"
        )
    if parameter_counts != manifest["parameters"]:
        mismatches.append(
            f"parameter counts expected {manifest['parameters']!r}, "
            f"got {parameter_counts!r}"
        )
    if mismatches:
        raise RuntimeError(
            "Task B does not match the recorded 0.66 run: " + "; ".join(mismatches)
        )
