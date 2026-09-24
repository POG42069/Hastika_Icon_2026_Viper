"""Offline tests with a real, tiny random Qwen2 and trainable PEFT adapters.

No pretrained weights, competition data or GPU kernels are needed.
The integration tests replace only the CUDA/model loading boundary; token loss,
training, best-checkpoint reload, diagnostics and submission run normally.
"""

from __future__ import annotations

import copy
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
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_config import TASK_A, TASK_B, LLMTrainingConfig, llm_task
from src.distributed import RuntimeContext
from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
from src.llm_training import (
    binary_majority_vote,
    completion_log_probs,
    completion_token_diagnostics,
    compute_class_weights,
    compute_dtype,
    format_per_class_diagnostics,
    require_cuda,
    restore_adapter,
    run_llm_task,
    save_adapter,
    score_candidates,
    train_llm_epoch,
)
from src.training import cap_training_labels, stratified_holdout_indices


class TestChatTokenizer:
    """Character tokenizer deliberately makes every label multi-token."""

    chat_template = "test-only-template"
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<end>"
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 10 for character in text]

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for value in ids:
            if value == self.eos_token_id:
                if not skip_special_tokens:
                    pieces.append(self.eos_token)
            elif value >= 10:
                pieces.append(chr(value - 10))
            elif not skip_special_tokens:
                pieces.append(f"<{value}>")
        return "".join(pieces)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        role_ids = {"system": 3, "user": 4, "assistant": 5}
        ids = []
        for message in messages:
            ids += [role_ids[message["role"]]] + self.encode(message["content"]) + [2]
        if add_generation_prompt:
            ids += [5]
        return ids

    def save_pretrained(self, path):
        Path(path, "test_tokenizer.json").write_text("{}", encoding="utf-8")


def tiny_model():
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        pad_token_id=0,
        eos_token_id=2,
        use_cache=False,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config)
    return get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
        ),
    )


def cpu_runtime():
    return RuntimeContext(torch.device("cpu"), 0, 0, 1, False)


class LLMTests(unittest.TestCase):
    def test_t4_uses_fp16_even_when_pytorch_supports_emulated_bf16(self):
        runtime = RuntimeContext(torch.device("cuda", 0), 0, 0, 1, False)
        config = LLMTrainingConfig()
        with (
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            self.assertEqual(compute_dtype(config, runtime), torch.float16)
            with self.assertRaisesRegex(ValueError, "select auto or float16"):
                compute_dtype(replace(config, compute_dtype="bfloat16"), runtime)

    def test_native_bf16_gpu_auto_and_explicit_fp16(self):
        runtime = RuntimeContext(torch.device("cuda", 0), 0, 0, 1, False)
        config = LLMTrainingConfig()
        with (
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            self.assertEqual(compute_dtype(config, runtime), torch.bfloat16)
            self.assertEqual(
                compute_dtype(replace(config, compute_dtype="float16"), runtime),
                torch.float16,
            )

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = TestChatTokenizer()
        self.encoder = ChatEncoder(self.tokenizer, TASK_A, 512)
        self.collator = CompletionCollator(0)
        self.config = replace(
            LLMTrainingConfig(),
            num_workers=0,
            max_length=512,
            gradient_checkpointing=False,
            lora_dropout=0.0,
        )

    def test_answer_mask_and_truncation_preserve_full_labels(self):
        self.assertEqual(TASK_A.label_codes, TASK_A.labels)
        self.assertIn("Allowed labels: Non-Hate; Hate", self.encoder.system_text)
        self.assertNotIn("Code mapping", self.encoder.system_text)
        for task in (TASK_A, TASK_B):
            encoder = ChatEncoder(self.tokenizer, task, 512)
            self.assertEqual(task.label_codes, task.labels)
            self.assertIn("Allowed labels: " + "; ".join(task.labels), encoder.system_text)
            self.assertNotIn("Code mapping", encoder.system_text)
            prompt, truncated = encoder.encode_prompt("long comment " * 300)
            self.assertTrue(truncated)
            for label_id, code in enumerate(task.label_codes):
                feature = encoder.feature(prompt, label_id)
                self.assertLessEqual(len(feature["input_ids"]), 512)
                self.assertEqual(feature["labels"][: len(prompt)], [-100] * len(prompt))
                self.assertEqual(
                    feature["labels"][len(prompt) :], self.tokenizer.encode(code) + [2]
                )
                self.assertEqual(feature["input_ids"][: len(prompt)], prompt)

    def test_encoding_matches_full_training_chat(self):
        prompt, _ = self.encoder.encode_prompt("hello")
        for label_id, code in enumerate(TASK_A.label_codes):
            expected = self.tokenizer.apply_chat_template(
                self.encoder.messages("hello")
                + [{"role": "assistant", "content": code}]
            )
            self.assertEqual(
                self.encoder.feature(prompt, label_id)["input_ids"], expected
            )

    def test_shifted_answer_loss_matches_full_qwen_loss(self):
        model = tiny_model().eval()
        dataset = CompletionDataset(
            ["hello", "a much longer comment"], self.encoder, [0, 1]
        )
        batch = self.collator([dataset[0], dataset[1]])
        scores, counts = completion_log_probs(model, batch)
        reference = model(**batch).loss
        self.assertTrue(
            torch.allclose(-scores.sum() / counts.sum(), reference, atol=1e-6)
        )
        self.assertEqual(counts.tolist(), [len(x) for x in self.encoder.completions])
        padding = batch["attention_mask"] == 0
        self.assertTrue(torch.all(batch["labels"][padding] == -100))
        prompt_position = len(batch["labels"][0]) - len(self.encoder.completions[0]) - 1
        self.assertEqual(batch["attention_mask"][0, prompt_position].item(), 1)
        self.assertEqual(batch["labels"][0, prompt_position].item(), -100)

    def test_causal_mask_blocks_future_answer(self):
        model = tiny_model().eval()
        prompt, _ = self.encoder.encode_prompt("hello")
        feature = self.encoder.feature(prompt, 1)
        batch = self.collator([feature])
        altered = {
            key: value.clone() for key, value in batch.items() if key != "labels"
        }
        original = model(**altered).logits
        altered["input_ids"][0, len(prompt)] = 200
        changed = model(**altered).logits
        self.assertTrue(
            torch.allclose(
                original[:, len(prompt) - 1], changed[:, len(prompt) - 1], atol=1e-6
            )
        )

    def test_only_adapter_weights_change_and_roundtrip(self):
        model = tiny_model()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        dataset = CompletionDataset(["aa", "bbb", "cccc"], self.encoder, [0, 1, 0])
        loader = DataLoader(dataset, batch_size=1, collate_fn=self.collator)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        config = replace(self.config, gradient_accumulation_steps=2)
        train_llm_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            torch.amp.GradScaler("cuda", enabled=False),
            config,
            cpu_runtime(),
            1,
        )
        self.assertEqual(scheduler.last_epoch, 2)
        changed = [
            name
            for name, p in model.named_parameters()
            if not torch.equal(before[name], p)
        ]
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in name for name in changed))
        model.eval()
        expected = score_candidates(
            model, dataset, self.collator, config, cpu_runtime(), "test"
        )
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(model, self.tokenizer, Path(directory))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        parameter.zero_()
            restore_adapter(model, Path(directory))
            actual = score_candidates(
                model, dataset, self.collator, config, cpu_runtime(), "test"
            )
            np.testing.assert_allclose(
                actual["probabilities"], expected["probabilities"], atol=1e-7
            )

    def test_partial_accumulation_matches_single_large_batch(self):
        first = tiny_model()
        second = copy.deepcopy(first)
        dataset = CompletionDataset(["a", "b", "long"], self.encoder, [0, 1, 0])
        # One group of 3 examples although accumulation is configured as 8.
        for model, batch_size, accumulation in ((first, 1, 8), (second, 3, 1)):
            loader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=self.collator
            )
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.01
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            train_llm_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                torch.amp.GradScaler("cuda", enabled=False),
                replace(self.config, gradient_accumulation_steps=accumulation),
                cpu_runtime(),
                1,
            )
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(torch.allclose(a, b, atol=1e-7))

    def test_mild_class_weights_preserve_scale_and_weight_rare_classes_more(self):
        labels = np.asarray([0] * 12 + [1] * 4 + [2] * 3, dtype=np.int64)
        weights, counts = compute_class_weights(labels, 3, 0.5, 0.75, 1.5)
        self.assertEqual(counts.tolist(), [12, 4, 3])
        self.assertAlmostEqual(
            float(np.dot(counts, weights.numpy()) / len(labels)), 1.0, places=6
        )
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_class_weighted_partial_accumulation_matches_large_batch(self):
        first = tiny_model()
        second = copy.deepcopy(first)
        dataset = CompletionDataset(["a", "b", "long"], self.encoder, [0, 1, 0])
        class_weights = torch.tensor([0.8, 1.4])
        for model, batch_size, accumulation in ((first, 1, 8), (second, 3, 1)):
            loader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=self.collator
            )
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.01
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            train_llm_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                torch.amp.GradScaler("cuda", enabled=False),
                replace(self.config, gradient_accumulation_steps=accumulation),
                cpu_runtime(),
                1,
                class_weights=class_weights,
            )
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(torch.allclose(a, b, atol=1e-7))

    def test_candidate_scores_independent_of_batching_and_gold_labels(self):
        model = tiny_model().eval()
        diagnostic_task = replace(TASK_B, report_per_class_metrics=True)
        encoder = ChatEncoder(self.tokenizer, diagnostic_task, 512)
        data = CompletionDataset(["a", "some other text", "xyz"], encoder, [0, 1, 5])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        scores = score_candidates(
            model, data, self.collator, self.config, cpu_runtime(), "test"
        )
        data.labels = [5, 5, 5]
        other = score_candidates(
            model,
            data,
            self.collator,
            replace(self.config, eval_batch_size_per_gpu=1),
            cpu_runtime(),
            "test",
        )
        np.testing.assert_allclose(
            scores["probabilities"], other["probabilities"], atol=1e-6
        )
        np.testing.assert_allclose(scores["probabilities"].sum(axis=1), np.ones(3))
        self.assertEqual(list(scores["per_class"]), list(TASK_B.labels))
        self.assertEqual(
            [scores["per_class"][label]["support"] for label in TASK_B.labels],
            [1, 1, 0, 0, 0, 1],
        )
        matrix = np.asarray(scores["confusion_matrix"]["values"])
        self.assertEqual(matrix.shape, (len(TASK_B.labels), len(TASK_B.labels)))
        self.assertEqual(matrix.sum(axis=1).tolist(), [1, 1, 0, 0, 0, 1])
        diagnostic_lines = format_per_class_diagnostics(scores)
        self.assertEqual(len(diagnostic_lines), len(TASK_B.labels) * 3)
        self.assertTrue(
            any(
                "False positives into Others (actual label)" in line
                for line in diagnostic_lines
            )
        )
        self.assertTrue(
            any(
                "False negatives from Violence (predicted label)" in line
                for line in diagnostic_lines
            )
        )
        synthetic_matrix = np.zeros(
            (len(TASK_B.labels), len(TASK_B.labels)), dtype=int
        )
        synthetic_matrix[0, 5] = 2  # Actual Gender predicted as Others.
        synthetic_matrix[4, 5] = 3  # Actual Violence predicted as Others.
        synthetic_metrics = {
            "per_class": {
                label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for label in TASK_B.labels
            },
            "confusion_matrix": {
                "label_order": list(TASK_B.labels),
                "values": synthetic_matrix.tolist(),
            },
        }
        synthetic_lines = format_per_class_diagnostics(synthetic_metrics)
        self.assertIn(
            "Gender=2, Political=0, Religion=0, Geo-political=0, Violence=3",
            synthetic_lines[16],
        )
        self.assertIn("Others=3", synthetic_lines[14])
        for class_metrics in scores["per_class"].values():
            self.assertEqual(
                set(class_metrics), {"precision", "recall", "f1", "support"}
            )
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter))

    def test_completion_token_counts_are_reported_and_constant_per_label(self):
        encoder = ChatEncoder(self.tokenizer, TASK_B, 512)
        data = CompletionDataset(["short", "a different comment"], encoder)
        result = score_candidates(
            tiny_model().eval(),
            data,
            self.collator,
            self.config,
            cpu_runtime(),
            "token count test",
        )
        expected = np.asarray(
            [len(token_ids) for token_ids in encoder.completions], dtype=np.float64
        )
        self.assertEqual(
            result["completion_token_counts"].shape,
            (2, len(TASK_B.labels)),
        )
        np.testing.assert_array_equal(
            result["completion_token_counts"],
            np.broadcast_to(expected, result["completion_token_counts"].shape),
        )
        diagnostics = completion_token_diagnostics(encoder)
        self.assertEqual(
            [item["token_count"] for item in diagnostics], expected.tolist()
        )
        self.assertTrue(all(item["token_pieces"] is None for item in diagnostics))

    def test_task_a_direct_labels_use_length_normalized_candidate_scores(self):
        encoder = ChatEncoder(self.tokenizer, TASK_A, 512)
        self.assertNotEqual(len(encoder.completions[0]), len(encoder.completions[1]))
        data = CompletionDataset(["a direct-label example"], encoder)
        result = score_candidates(
            tiny_model().eval(),
            data,
            self.collator,
            self.config,
            cpu_runtime(),
            "direct-label test",
        )
        np.testing.assert_allclose(
            result["mean_token_log_probs"],
            result["sequence_log_probs"] / result["completion_token_counts"],
        )
        expected = torch.softmax(
            torch.from_numpy(result["mean_token_log_probs"]), dim=1
        ).numpy()
        np.testing.assert_allclose(result["probabilities"], expected)

    def test_binary_majority_vote_uses_all_five_seeds(self):
        seed_predictions = np.asarray(
            [
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ]
        )
        final_ids, hate_votes, nonhate_votes = binary_majority_vote(
            seed_predictions, TASK_A.labels
        )
        np.testing.assert_array_equal(hate_votes, [3, 1, 3, 3])
        np.testing.assert_array_equal(nonhate_votes, [2, 4, 2, 2])
        np.testing.assert_array_equal(final_ids, [1, 0, 1, 1])

    def test_binary_majority_vote_rejects_even_seed_count(self):
        with self.assertRaisesRegex(ValueError, "odd number of seeds"):
            binary_majority_vote(np.asarray([[0, 1], [1, 0]]), TASK_A.labels)

    def test_cpu_production_run_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            require_cuda(cpu_runtime())

    def test_task_a_selects_then_retrains_every_seed_before_voting(self):
        for original_task in (TASK_A,):
            with (
                self.subTest(task=original_task.task_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                test_seeds = (
                    (42, 52, 62) if original_task.ensemble_all_seeds else (42,)
                )
                task = replace(llm_task(original_task), selection_seeds=test_seeds)
                if task.original_fold_filenames:
                    task = replace(
                        task, selection_train_label_caps=(("Gender", 4),)
                    )
                    original_frames = []
                    augment_frames = []
                    for fold, (original_name, augment_name) in enumerate(
                        zip(
                            task.original_fold_filenames,
                            task.augmentation_fold_filenames,
                            strict=True,
                        ),
                        start=1,
                    ):
                        original_labels = list(task.labels) * 2
                        original = pd.DataFrame(
                            {
                                "id": [
                                    f"original-{fold}-{i}"
                                    for i in range(len(original_labels))
                                ],
                                "Comment": [
                                    f"original comment {fold}-{i}"
                                    for i in range(len(original_labels))
                                ],
                                task.label_column: original_labels,
                            }
                        )
                        augment = pd.DataFrame(
                            {
                                "id": [
                                    f"augment-{fold}-{i}"
                                    for i in range(len(task.labels))
                                ],
                                "Comment": [
                                    f"paraphrase {fold}-{i}"
                                    for i in range(len(task.labels))
                                ],
                                task.label_column: list(task.labels),
                                "source_id": original["id"].iloc[
                                    : len(task.labels)
                                ],
                                "source_group": [
                                    f"group-{fold}-{i}"
                                    for i in range(len(task.labels))
                                ],
                                "fold": fold,
                            }
                        )
                        original.to_csv(root / original_name, index=False)
                        if task.use_augmentation:
                            augment.to_csv(root / augment_name, index=False)
                        original["fold"] = fold
                        original["data_origin"] = "original"
                        augment["fold"] = fold
                        augment["data_origin"] = "augmentation"
                        original_frames.append(original)
                        if task.use_augmentation:
                            augment_frames.append(augment)
                    frame = pd.concat(
                        original_frames + augment_frames,
                        ignore_index=True,
                        sort=False,
                    )
                    raw_train = np.flatnonzero(
                        frame["fold"].to_numpy() != task.selection_validation_fold
                    )
                    val = np.flatnonzero(
                        (frame["fold"].to_numpy() == task.selection_validation_fold)
                        & (frame["data_origin"].to_numpy() == "original")
                    )
                    train, downsampled, _ = cap_training_labels(
                        raw_train,
                        frame[task.label_column].to_numpy(),
                        task.selection_train_label_caps,
                        seed=42,
                    )
                    full_train = np.setdiff1d(
                        np.arange(len(frame)), downsampled, assume_unique=True
                    )
                else:
                    # Sufficient examples for the deterministic stratified holdout.
                    labels = list(task.labels) * 10
                    frame = pd.DataFrame(
                        {
                            "id": [f"{i:04d}" for i in range(len(labels))],
                            "Comment": [f"comment {i}" for i in range(len(labels))],
                            task.label_column: labels,
                        }
                    )
                    frame.to_csv(root / task.train_filename, index=False)
                    train, val = stratified_holdout_indices(
                        np.asarray(list(range(len(task.labels))) * 10), 0.2, 42
                    )
                    downsampled = np.empty(0, dtype=np.int64)
                    full_train = np.arange(len(frame), dtype=np.int64)
                prediction_frame = pd.DataFrame(
                    {"id": ["0100", "0101", "0102"], "Comment": ["one", "two", "three"]}
                )
                prediction_frame.to_csv(root / task.prediction_filename, index=False)
                config = replace(
                    self.config,
                    max_epochs=2,
                    train_batch_size_per_gpu=4,
                    gradient_accumulation_steps=2,
                    eval_batch_size_per_gpu=6,
                )
                real_score = score_candidates
                calls = []
                loaded_models = []
                train_sizes = []

                def model_factory(
                    *_args, loaded_models=loaded_models, **_kwargs
                ):
                    model = tiny_model()
                    loaded_models.append(model)
                    return model

                def tracked_train(*args, train_sizes=train_sizes, **kwargs):
                    train_sizes.append(len(args[1].dataset))
                    return train_llm_epoch(*args, **kwargs)

                def controlled_score(*args, real_score=real_score, calls=calls):
                    result = real_score(*args)
                    calls.append(args[-1])
                    # Force epoch 1 to win; full-data replay must use one epoch.
                    if args[-1] == "Epoch 1 validation":
                        result["macro_f1"] = 1.0
                    elif args[-1] == "Epoch 2 validation":
                        result["macro_f1"] = 0.0
                    return result

                with (
                    patch("src.llm_training.require_cuda"),
                    patch(
                        "src.llm_training.initialize_runtime",
                        return_value=cpu_runtime(),
                    ),
                    patch(
                        "src.llm_training.load_lora_model",
                        side_effect=model_factory,
                    ),
                    patch(
                        "src.llm_training.AutoTokenizer.from_pretrained",
                        return_value=self.tokenizer,
                    ),
                    patch("src.data.resolve_data_file", side_effect=root.joinpath),
                    patch(
                        "src.llm_training.resolve_data_file", side_effect=root.joinpath
                    ),
                    patch("src.llm_training.OUTPUT_ROOT", root / "outputs"),
                    patch("src.llm_training.CHECKPOINT_ROOT", root / "checkpoints"),
                    patch(
                        "src.llm_training.score_candidates",
                        side_effect=controlled_score,
                    ),
                    patch(
                        "src.llm_training.train_llm_epoch",
                        side_effect=tracked_train,
                    ),
                ):
                    run_llm_task(task, config)
                output = root / "outputs" / task.output_subdir
                result = pd.read_csv(output / "predictions.csv", dtype={"id": str})
                self.assertEqual(result.columns.tolist(), ["id", "label"])
                self.assertEqual(result["id"].tolist(), ["0100", "0101", "0102"])
                self.assertTrue(set(result["label"]).issubset(task.labels))
                with zipfile.ZipFile(output / task.submission_zip_name) as archive:
                    self.assertEqual(archive.namelist(), ["predictions.csv"])
                    self.assertEqual(
                        archive.read("predictions.csv"),
                        (output / "predictions.csv").read_bytes(),
                    )
                summary = json.loads((output / "run_summary.json").read_text())
                self.assertEqual(summary["quantization"], None)
                if task.ensemble_all_seeds:
                    self.assertEqual(
                        summary["best_validation_checkpoint_for_reporting"][
                            "best_epoch"
                        ],
                        1,
                    )
                    self.assertEqual(
                        len(summary["full_data_retrains"]), len(test_seeds)
                    )
                    for retrain_summary in summary["full_data_retrains"]:
                        self.assertEqual(retrain_summary["epochs"], 1)
                        self.assertEqual(
                            retrain_summary["training_rows"], len(full_train)
                        )
                        self.assertEqual(
                            retrain_summary["downsampled_rows_excluded"],
                            len(downsampled),
                        )
                    voting = pd.read_csv(
                        output / "voting_predictions.csv", dtype={"id": str}
                    )
                    self.assertEqual(voting["id"].tolist(), result["id"].tolist())
                    self.assertEqual(
                        voting["final_label"].tolist(), result["label"].tolist()
                    )
                    self.assertEqual(
                        (voting["hate_votes"] + voting["nonhate_votes"]).tolist(),
                        [len(test_seeds)] * len(voting),
                    )
                else:
                    self.assertEqual(summary["best_checkpoint"]["best_epoch"], 1)
                    self.assertEqual(summary["full_data_retrain"]["epochs"], 1)
                    self.assertEqual(
                        summary["full_data_retrain"]["training_rows"],
                        len(full_train),
                    )
                    self.assertEqual(
                        summary["full_data_retrain"]["downsampled_rows_excluded"],
                        len(downsampled),
                    )
                expected_model_count = (
                    len(test_seeds) * 2 if task.ensemble_all_seeds else 2
                )
                self.assertEqual(len(loaded_models), expected_model_count)
                expected_train_sizes = [len(train), len(train)] * len(test_seeds)
                expected_train_sizes += [len(full_train)] * (
                    len(test_seeds) if task.ensemble_all_seeds else 1
                )
                self.assertEqual(train_sizes, expected_train_sizes)
                manifest = pd.read_csv(output / "split_manifest.csv", dtype={"id": str})
                self.assertEqual(
                    manifest.loc[val, "split"].tolist(),
                    ["selection_validation"] * len(val),
                )
                self.assertEqual(
                    manifest.loc[train, "split"].tolist(),
                    ["selection_train"] * len(train),
                )
                self.assertEqual(
                    manifest.loc[downsampled, "split"].tolist(),
                    ["selection_downsampled"] * len(downsampled),
                )
                if task.original_fold_filenames and task.use_augmentation:
                    excluded = np.flatnonzero(
                        (frame["fold"].to_numpy() == task.selection_validation_fold)
                        & (frame["data_origin"].to_numpy() == "augmentation")
                    )
                    self.assertEqual(
                        manifest.loc[excluded, "split"].tolist(),
                        ["selection_excluded"] * len(excluded),
                    )
                self.assertIn("Best holdout adapter validation", calls)
                self.assertEqual(
                    calls.count("Development prediction from full-data adapter"),
                    len(test_seeds) if task.ensemble_all_seeds else 1,
                )
                encoder = ChatEncoder(self.tokenizer, task, config.max_length)
                data = CompletionDataset(prediction_frame["Comment"].tolist(), encoder)
                if task.ensemble_all_seeds:
                    seed_label_ids = []
                    for index, seed in enumerate(test_seeds):
                        final_model = loaded_models[len(test_seeds) + index]
                        restore_adapter(
                            final_model,
                            root
                            / "checkpoints"
                            / task.checkpoint_subdir
                            / f"seed_{seed}"
                            / "final_full_data_adapter",
                        )
                        scores = real_score(
                            final_model,
                            data,
                            self.collator,
                            config,
                            cpu_runtime(),
                            "verify",
                        )
                        seed_label_ids.append(
                            scores["probabilities"].argmax(axis=1)
                        )
                    expected_ids, _, _ = binary_majority_vote(
                        np.stack(seed_label_ids), task.labels
                    )
                    expected = [task.labels[index] for index in expected_ids]
                else:
                    final_model = loaded_models[1]
                    restore_adapter(
                        final_model,
                        root
                        / "checkpoints"
                        / task.checkpoint_subdir
                        / "final_full_data_adapter",
                    )
                    scores = real_score(
                        final_model,
                        data,
                        self.collator,
                        config,
                        cpu_runtime(),
                        "verify",
                    )
                    expected = [
                        task.labels[index]
                        for index in scores["probabilities"].argmax(axis=1)
                    ]
                self.assertEqual(result["label"].tolist(), expected)

    def test_task_b_skips_oof_and_soft_ensembles_fixed_epoch_full_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retrain_seeds = (42, 52)
            task = replace(
                TASK_B,
                selection_seeds=retrain_seeds,
                enable_augmentation_audit=False,
            )
            rows_per_fold = len(task.labels) * 2
            for fold, (original_name, augment_name) in enumerate(
                zip(
                    task.original_fold_filenames,
                    task.augmentation_fold_filenames,
                    strict=True,
                ),
                start=1,
            ):
                original_labels = list(task.labels) * 2
                original = pd.DataFrame(
                    {
                        "id": [f"original-{fold}-{i}" for i in range(rows_per_fold)],
                        "Comment": [
                            f"original comment {fold}-{i}" for i in range(rows_per_fold)
                        ],
                        task.label_column: original_labels,
                    }
                )
                augment = pd.DataFrame(
                    {
                        "id": [f"augment-{fold}-{i}" for i in range(len(task.labels))],
                        "Comment": [
                            f"paraphrase {fold}-{i}" for i in range(len(task.labels))
                        ],
                        task.label_column: list(task.labels),
                        "source_id": original["id"].iloc[: len(task.labels)],
                        "source_group": [
                            f"group-{fold}-{i}" for i in range(len(task.labels))
                        ],
                        "fold": fold,
                    }
                )
                original.to_csv(root / original_name, index=False)
                augment.to_csv(root / augment_name, index=False)

            prediction_frame = pd.DataFrame(
                {
                    "id": ["0100", "0101", "0102"],
                    "Comment": ["one", "two", "three"],
                }
            )
            prediction_frame.to_csv(root / task.prediction_filename, index=False)
            config = replace(
                self.config,
                max_epochs=2,
                train_batch_size_per_gpu=6,
                gradient_accumulation_steps=1,
                eval_batch_size_per_gpu=6,
            )
            real_score = score_candidates
            loaded_models = []
            train_calls = []

            def model_factory(*_args, **_kwargs):
                model = tiny_model()
                loaded_models.append(model)
                return model

            def tracked_train(*args, **kwargs):
                train_calls.append(
                    {
                        "rows": len(args[1].dataset),
                        "class_weights": kwargs.get("class_weights"),
                    }
                )
                return train_llm_epoch(*args, **kwargs)

            with (
                patch("src.llm_training.require_cuda"),
                patch(
                    "src.llm_training.initialize_runtime", return_value=cpu_runtime()
                ),
                patch(
                    "src.llm_training.load_lora_model", side_effect=model_factory
                ),
                patch(
                    "src.llm_training.AutoTokenizer.from_pretrained",
                    return_value=self.tokenizer,
                ),
                patch("src.data.resolve_data_file", side_effect=root.joinpath),
                patch(
                    "src.llm_training.resolve_data_file", side_effect=root.joinpath
                ),
                patch("src.llm_training.OUTPUT_ROOT", root / "outputs"),
                patch("src.llm_training.CHECKPOINT_ROOT", root / "checkpoints"),
                patch(
                    "src.llm_training.score_candidates", side_effect=real_score
                ),
                patch(
                    "src.llm_training.train_llm_epoch", side_effect=tracked_train
                ),
            ):
                run_llm_task(task, config)

            output = root / "outputs" / task.output_subdir
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["epoch_selection"]["fixed_epoch"], 5)
            self.assertTrue(summary["epoch_selection"]["oof_skipped"])
            self.assertEqual(
                summary["epoch_selection"]["strategy"], "fixed_without_oof"
            )
            self.assertEqual(summary["class_weighting"]["downsampling"], None)
            self.assertEqual(len(summary["full_data_retrains"]), len(retrain_seeds))
            self.assertEqual(summary["ensemble"]["method"], (
                "equal-weight mean of per-seed candidate probabilities"
            ))

            self.assertFalse((output / "oof_predictions.csv").exists())
            self.assertFalse((output / "oof_selection_summary.json").exists())

            result = pd.read_csv(output / "predictions.csv", dtype={"id": str})
            ensemble = pd.read_csv(
                output / "ensemble_predictions.csv", dtype={"id": str}
            )
            self.assertEqual(result["id"].tolist(), prediction_frame["id"].tolist())
            self.assertEqual(result["label"].tolist(), ensemble["final_label"].tolist())
            probability_columns = [
                f"mean_prob_{label.lower().replace('-', '_').replace(' ', '_')}"
                for label in task.labels
            ]
            expected_ids = ensemble[probability_columns].to_numpy().argmax(axis=1)
            self.assertEqual(
                result["label"].tolist(),
                [task.labels[index] for index in expected_ids],
            )
            with zipfile.ZipFile(output / task.submission_zip_name) as archive:
                self.assertEqual(archive.namelist(), ["predictions.csv"])

            self.assertEqual(len(loaded_models), len(retrain_seeds))
            self.assertEqual(
                [call["rows"] for call in train_calls],
                [rows_per_fold * 5 + len(task.labels) * 5]
                * (len(retrain_seeds) * task.fixed_full_retrain_epochs),
            )
            self.assertTrue(
                all(call["class_weights"] is not None for call in train_calls)
            )
            self.assertFalse((output / "downsampled_rows.csv").exists())
            for seed in retrain_seeds:
                self.assertTrue(
                    (
                        root
                        / "checkpoints"
                        / task.checkpoint_subdir
                        / f"seed_{seed}"
                        / "final_full_data_adapter"
                    ).is_dir()
                )


if __name__ == "__main__":
    unittest.main()
