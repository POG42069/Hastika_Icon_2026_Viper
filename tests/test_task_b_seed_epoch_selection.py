"""Per-seed Task B holdout selection followed by independent full-data retrains."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from llm_config import TASK_B, LLMTrainingConfig
from src.llm_training import (
    compute_class_weights,
    run_llm_task,
    score_candidates,
    train_llm_epoch,
)
from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime, tiny_model


class TaskBSeedEpochSelectionTests(unittest.TestCase):
    def test_every_seed_uses_its_own_best_holdout_epoch_and_full_data_weights(self):
        torch.set_num_threads(1)
        seeds = TASK_B.selection_seeds
        self.assertEqual(seeds, (42, 52, 62, 72, 82))
        self.assertEqual(TASK_B.selection_validation_fold, 5)
        self.assertIsNone(TASK_B.fixed_full_retrain_epochs)
        self.assertFalse(TASK_B.oof_validation_folds)
        self.assertFalse(TASK_B.use_augmentation)
        # Include F1 ties: lower loss selects epoch 4 for seed 72, while an
        # exact F1/loss tie keeps the earlier epoch for seeds 62 and 82.
        f1_by_seed = {
            42: [0.6, 0.5, 0.5],
            52: [0.2, 0.4, 0.7, 0.6, 0.6],
            62: [0.4, 0.6, 0.6, 0.5],
            72: [0.1, 0.3, 0.7, 0.7, 0.6],
            82: [0.7, 0.7, 0.7],
        }
        loss_by_seed = {
            42: [0.4, 0.4, 0.4],
            52: [0.5, 0.5, 0.3, 0.3, 0.3],
            62: [0.5, 0.2, 0.2, 0.2],
            72: [0.5, 0.5, 0.4, 0.3, 0.3],
            82: [0.3, 0.3, 0.3],
        }
        expected_epochs = {42: 1, 52: 3, 62: 2, 72: 4, 82: 1}
        tokenizer = TestChatTokenizer()
        config = replace(
            LLMTrainingConfig(),
            max_length=512,
            max_epochs=5,
            train_batch_size_per_gpu=8,
            gradient_accumulation_steps=1,
            eval_batch_size_per_gpu=12,
            gradient_checkpointing=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for fold, name in enumerate(TASK_B.original_fold_filenames, start=1):
                labels = (
                    [TASK_B.labels[0]] * 3 + list(TASK_B.labels[1:])
                    if fold < 5
                    else [TASK_B.labels[0]]
                    + [TASK_B.labels[1]] * 3
                    + list(TASK_B.labels[2:])
                )
                frame = pd.DataFrame(
                    {
                        "id": [f"original-{fold}-{i}" for i in range(len(labels))],
                        "Comment": [
                            f"original {fold} comment {i}" for i in range(len(labels))
                        ],
                        TASK_B.label_column: labels,
                    }
                )
                frame.to_csv(root / name, index=False)
                frames.append(frame)
            # Invalid augment schemas must not be read by the original-only run.
            for name in TASK_B.augmentation_fold_filenames:
                (root / name).write_text("invalid_augment_schema\n", encoding="utf-8")
            development = pd.DataFrame(
                {
                    "id": ["0100", "0101", "0102"],
                    "Comment": ["one", "two", "three"],
                }
            )
            development.to_csv(root / TASK_B.prediction_filename, index=False)
            gold = pd.DataFrame(
                {
                    "id": [f"gold-{i}" for i in range(len(TASK_B.labels))],
                    "Comment": [f"gold comment {i}" for i in range(len(TASK_B.labels))],
                    TASK_B.label_column: list(TASK_B.labels),
                }
            )
            gold.to_csv(root / TASK_B.epoch_validation_filename, index=False)
            all_original = pd.concat(frames, ignore_index=True)
            selection_labels = (
                pd.concat(frames[:4])[TASK_B.label_column]
                .map({label: i for i, label in enumerate(TASK_B.labels)})
                .to_numpy()
            )
            full_labels = (
                all_original[TASK_B.label_column]
                .map({label: i for i, label in enumerate(TASK_B.labels)})
                .to_numpy()
            )
            selection_weights, _ = compute_class_weights(
                selection_labels, 6, 0.5, 0.75, 1.5
            )
            full_weights, _ = compute_class_weights(full_labels, 6, 0.5, 0.75, 1.5)
            self.assertFalse(torch.equal(selection_weights, full_weights))

            loaded_models = []
            model_info = {}
            initial_weights = {}
            trained_weights = {}
            train_calls = []
            restored_seeds = []

            def adapter_state(model):
                return {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                    if value.requires_grad
                }

            def model_factory(*_args, **_kwargs):
                index = len(loaded_models)
                phase = "selection" if index < len(seeds) else "full"
                seed = seeds[index % len(seeds)]
                model = tiny_model()
                loaded_models.append(model)
                model_info[id(model)] = (phase, seed)
                initial_weights[phase, seed] = adapter_state(model)
                return model

            def tracked_train(*args, **kwargs):
                model, loader = args[:2]
                phase, seed = model_info[id(model)]
                epoch = args[7]
                self.assertEqual(
                    len(loader.dataset), 32 if phase == "selection" else 40
                )
                np.testing.assert_allclose(
                    kwargs["class_weights"].numpy(),
                    (
                        selection_weights if phase == "selection" else full_weights
                    ).numpy(),
                )
                train_calls.append((phase, seed, epoch))
                loss = train_llm_epoch(*args, **kwargs)
                trained_weights[phase, seed, epoch] = adapter_state(model)
                return loss

            def controlled_score(*args):
                metrics = score_candidates(*args)
                phase, seed = model_info[id(args[0])]
                description = args[-1]
                if description.startswith("Epoch ") and description.endswith(
                    " validation"
                ):
                    epoch = int(description.split()[1])
                    metrics["macro_f1"] = f1_by_seed[seed][epoch - 1]
                    metrics["loss"] = loss_by_seed[seed][epoch - 1]
                elif description == "Best holdout adapter validation":
                    expected = trained_weights["selection", seed, expected_epochs[seed]]
                    for name, actual in adapter_state(args[0]).items():
                        self.assertTrue(torch.equal(actual, expected[name]))
                    restored_seeds.append(seed)
                    metrics["macro_f1"] = f1_by_seed[seed][expected_epochs[seed] - 1]
                    metrics["loss"] = loss_by_seed[seed][expected_epochs[seed] - 1]
                elif description.startswith("Official validation | seed="):
                    # A different optimum on official gold must not change the
                    # holdout-derived epoch counts or stop full-data training.
                    epoch = int(description.split("epoch=")[-1])
                    metrics["macro_f1"] = 1.0 if epoch == 1 else 0.0
                elif description.startswith("Development prediction seed "):
                    self.assertEqual(phase, "full")
                    expected = trained_weights["full", seed, expected_epochs[seed]]
                    for name, actual in adapter_state(args[0]).items():
                        self.assertTrue(torch.equal(actual, expected[name]))
                return metrics

            with (
                patch("src.llm_training.require_cuda"),
                patch(
                    "src.llm_training.initialize_runtime", return_value=cpu_runtime()
                ),
                patch("src.llm_training.load_lora_model", side_effect=model_factory),
                patch(
                    "src.llm_training.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch("src.data.resolve_data_file", side_effect=root.joinpath),
                patch("src.llm_training.resolve_data_file", side_effect=root.joinpath),
                patch("src.llm_training.OUTPUT_ROOT", root / "outputs"),
                patch("src.llm_training.CHECKPOINT_ROOT", root / "checkpoints"),
                patch(
                    "src.llm_training.score_candidates", side_effect=controlled_score
                ),
                patch("src.llm_training.train_llm_epoch", side_effect=tracked_train),
                patch("src.llm_training.log") as logged,
            ):
                run_llm_task(TASK_B, config)

            self.assertEqual(len(loaded_models), 2 * len(seeds))
            self.assertEqual(restored_seeds, list(seeds))
            for seed in seeds:
                for name, value in initial_weights["selection", seed].items():
                    self.assertTrue(
                        torch.equal(value, initial_weights["full", seed][name])
                    )
            expected_train_calls = [
                ("selection", seed, epoch)
                for seed in seeds
                for epoch in range(1, len(f1_by_seed[seed]) + 1)
            ] + [
                ("full", seed, epoch)
                for seed in seeds
                for epoch in range(1, expected_epochs[seed] + 1)
            ]
            self.assertEqual(train_calls, expected_train_calls)
            output = root / "outputs" / TASK_B.output_subdir
            summary = json.loads((output / "run_summary.json").read_text())
            selection = summary["epoch_selection"]
            self.assertEqual(selection["strategy"], "per_seed_holdout")
            self.assertEqual(selection["validation_fold"], 5)
            self.assertEqual(
                selection["best_epoch_by_seed"],
                {str(seed): epoch for seed, epoch in expected_epochs.items()},
            )
            self.assertTrue(selection["all_seeds_retrained"])
            results = pd.read_csv(output / "seed_selection_results.csv")
            self.assertEqual(
                results["best_epoch"].tolist(), list(expected_epochs.values())
            )
            self.assertTrue(results["selected"].all())
            self.assertFalse((output / "oof_predictions.csv").exists())
            self.assertFalse((output / "oof_selection_summary.json").exists())
            for seed in seeds:
                seed_output = output / f"seed_{seed}"
                manifest = pd.read_csv(seed_output / "split_manifest.csv")
                val_ids = manifest.loc[
                    manifest["split"] == "selection_validation", "id"
                ]
                self.assertEqual(val_ids.tolist(), frames[4]["id"].tolist())
                validation = pd.read_csv(seed_output / "validation_predictions.csv")
                self.assertEqual(validation["id"].tolist(), frames[4]["id"].tolist())
                history = json.loads(
                    (seed_output / "full_data_training_history.json").read_text()
                )
                self.assertEqual(len(history["epochs"]), expected_epochs[seed])
                self.assertTrue(
                    all("official_validation" in row for row in history["epochs"])
                )
                seed_summary = json.loads(
                    (seed_output / "seed_summary.json").read_text()
                )
                self.assertEqual(seed_summary["epochs"], expected_epochs[seed])
                self.assertEqual(
                    seed_summary["holdout_selection"]["best_epoch"],
                    expected_epochs[seed],
                )
                checkpoint = (
                    root / "checkpoints" / TASK_B.checkpoint_subdir / f"seed_{seed}"
                )
                self.assertTrue((checkpoint / "best_holdout_adapter").is_dir())
                self.assertTrue((checkpoint / "final_full_data_adapter").is_dir())
            for record in summary["full_data_retrains"]:
                self.assertEqual(record["training_rows"], 40)
                self.assertEqual(record["epochs"], expected_epochs[record["seed"]])
            final = pd.read_csv(output / "predictions.csv", dtype={"id": str})
            ensemble = pd.read_csv(
                output / "ensemble_predictions.csv", dtype={"id": str}
            )
            self.assertEqual(final.columns.tolist(), ["id", "label"])
            self.assertEqual(final["id"].tolist(), development["id"].tolist())
            mean_columns = [
                f"mean_prob_{label.lower().replace('-', '_')}"
                for label in TASK_B.labels
            ]
            expected_labels = [
                TASK_B.labels[i]
                for i in ensemble[mean_columns].to_numpy().argmax(axis=1)
            ]
            self.assertEqual(final["label"].tolist(), expected_labels)
            self.assertEqual(summary["ensemble"]["models"], 5)
            with zipfile.ZipFile(output / TASK_B.submission_zip_name) as archive:
                self.assertEqual(archive.namelist(), ["predictions.csv"])
                self.assertEqual(
                    archive.read("predictions.csv"),
                    (output / "predictions.csv").read_bytes(),
                )
            self.assertTrue((output / "official_validation_metrics.json").is_file())
            lines = [call.args[0] for call in logged.call_args_list]
            self.assertTrue(
                any(
                    "FINAL SOFT ENSEMBLE" in line and "macro_f1=" in line
                    for line in lines
                )
            )
            self.assertEqual(
                sum(line.startswith("Confusion matrix |") for line in lines),
                len(train_calls) + 1,
            )


if __name__ == "__main__":
    unittest.main()
