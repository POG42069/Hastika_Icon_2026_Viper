"""Two-process CPU DDP smoke test executed manually through torchrun."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from config import TRAINING
from src.data import ClassificationCollator, EncodedTextDataset
from src.distributed import cleanup_runtime, initialize_runtime
from src.training import (
    build_optimizer,
    create_eval_loader,
    create_train_loader,
    evaluate_model,
    predict_distributed,
    train_one_epoch,
)
from tests.test_training_smoke import DummyClassifier, DummyTokenizer


def main() -> None:
    """Exercise training, evaluation, and gathering with two DDP workers."""

    runtime = initialize_runtime()
    try:
        if runtime.world_size != 2:
            raise RuntimeError("Run this smoke test with exactly two torchrun workers.")
        config = replace(
            TRAINING,
            train_batch_size_per_gpu=2,
            eval_batch_size_per_gpu=2,
            num_workers=0,
            use_amp=False,
            pad_to_multiple_of=4,
        )
        tokenizer = DummyTokenizer()
        collator = ClassificationCollator(tokenizer, config.pad_to_multiple_of)
        texts = ["aa", "bbb", "cccc", "ddddd", "ee", "fff", "gggg", "hhhhh"]
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        labeled_dataset = EncodedTextDataset(
            texts, tokenizer, max_length=8, labels=labels
        )
        prediction_dataset = EncodedTextDataset(texts, tokenizer, max_length=8)
        train_loader, sampler = create_train_loader(
            labeled_dataset, collator, config, runtime, fold_seed=42
        )
        eval_loader = create_eval_loader(labeled_dataset, collator, config, runtime)
        prediction_loader = create_eval_loader(
            prediction_dataset, collator, config, runtime
        )

        model = DummyClassifier().to(runtime.device)
        model = nn.parallel.DistributedDataParallel(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        if sampler is not None:
            sampler.set_epoch(1)
        train_one_epoch(
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
        metrics = evaluate_model(
            model,
            eval_loader,
            criterion,
            num_labels=2,
            runtime=runtime,
            description="DDP smoke validation",
        )
        indices, probabilities, _ = predict_distributed(
            model, prediction_loader, runtime, description="DDP smoke prediction"
        )
        if indices.tolist() != list(range(8)) or probabilities.shape != (8, 2):
            raise RuntimeError("DDP gather did not preserve every prediction row.")
        if runtime.is_main_process:
            print(f"DDP_SMOKE_OK macro_f1={metrics['macro_f1']:.5f}", flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
