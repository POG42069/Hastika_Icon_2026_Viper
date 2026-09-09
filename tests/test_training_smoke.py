"""CPU smoke test for the shared training and prediction engine."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from config import TRAINING
from src.data import ClassificationCollator, EncodedTextDataset
from src.distributed import RuntimeContext
from src.training import (
    build_optimizer,
    create_eval_loader,
    create_train_loader,
    evaluate_model,
    load_model_state,
    predict_distributed,
    save_model_state,
    train_one_epoch,
)


class DummyTokenizer:
    """Small tokenizer implementing only the methods used by the pipeline."""

    def __call__(
        self,
        text: str,
        truncation: bool,
        max_length: int,
        padding: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding
        token_ids = [ord(character) % 31 + 1 for character in text][:max_length]
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def pad(
        self,
        features: list[dict[str, object]],
        padding: bool,
        pad_to_multiple_of: int | None,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        longest = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            longest = (
                (longest + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
        input_ids = []
        attention_masks = []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            padding_size = longest - len(ids)
            input_ids.append(ids + [0] * padding_size)
            attention_masks.append(mask + [0] * padding_size)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class DummyClassifier(nn.Module):
    """Tiny sequence classifier with a Hugging Face-like return object."""

    def __init__(self, num_labels: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8, padding_idx=0)
        self.classifier = nn.Linear(8, num_labels)
        self.config = SimpleNamespace(num_labels=num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(logits=self.classifier(pooled))


class TrainingSmokeTests(unittest.TestCase):
    """Exercise one CPU epoch plus validation and prediction."""

    def test_training_evaluation_and_prediction(self) -> None:
        """The core loop should preserve every row and return probabilities."""

        config = replace(
            TRAINING,
            train_batch_size_per_gpu=2,
            eval_batch_size_per_gpu=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            use_amp=False,
            pad_to_multiple_of=4,
        )
        runtime = RuntimeContext(
            device=torch.device("cpu"),
            rank=0,
            local_rank=0,
            world_size=1,
            distributed=False,
        )
        tokenizer = DummyTokenizer()
        collator = ClassificationCollator(tokenizer, config.pad_to_multiple_of)
        labeled_dataset = EncodedTextDataset(
            ["aa", "bbb", "cccc", "ddddd"], tokenizer, max_length=8, labels=[0, 1, 0, 1]
        )
        prediction_dataset = EncodedTextDataset(
            ["aa", "bbb", "cccc", "ddddd"], tokenizer, max_length=8
        )
        train_loader, _ = create_train_loader(
            labeled_dataset, collator, config, runtime, shuffle_seed=42
        )
        eval_loader = create_eval_loader(labeled_dataset, collator, config, runtime)
        prediction_loader = create_eval_loader(
            prediction_dataset, collator, config, runtime
        )

        model = DummyClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config,
            runtime,
            epoch_number=1,
        )
        weights_before_validation = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        metrics = evaluate_model(
            model,
            eval_loader,
            criterion,
            num_labels=2,
            runtime=runtime,
            description="smoke validation",
        )
        for name, parameter in model.named_parameters():
            self.assertTrue(
                torch.equal(weights_before_validation[name], parameter.detach()),
                msg=f"Validation unexpectedly updated parameter: {name}",
            )
        indices, probabilities, labels = predict_distributed(
            model, prediction_loader, runtime, description="smoke prediction"
        )

        self.assertGreater(train_loss, 0.0)
        self.assertEqual(metrics["probabilities"].shape, (4, 2))
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertIsNone(labels)
        self.assertTrue(
            torch.allclose(torch.tensor(probabilities.sum(axis=1)), torch.ones(4))
        )

        expected_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best_model.pt"
            save_model_state(model, checkpoint_path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            load_model_state(model, checkpoint_path, runtime.device)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(expected_state[name], parameter.detach()))


if __name__ == "__main__":
    unittest.main()
