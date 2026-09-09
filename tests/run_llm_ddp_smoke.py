"""Two CPU/Gloo workers: adapter updates, uneven shards and an empty eval shard."""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(rank: int, port: int) -> None:
    # This explicitly tests CPU/Gloo even on a CUDA development machine.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        USE_LIBUV="0",
    )
    import numpy as np
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from config import TASK_B
    from llm_config import LLMTrainingConfig
    from src.distributed import RuntimeContext, cleanup_runtime, gather_python_objects
    from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
    from src.llm_training import score_candidates, train_llm_epoch
    from src.training import create_train_loader
    from tests.test_llm_pipeline import TestChatTokenizer, cpu_runtime, tiny_model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.distributed.init_process_group("gloo", init_method="env://")
    runtime = RuntimeContext(torch.device("cpu"), rank, rank, 2, True)
    try:
        config = replace(
            LLMTrainingConfig(),
            max_length=512,
            gradient_accumulation_steps=2,
            eval_batch_size_per_gpu=4,
            num_workers=0,
        )
        encoder = ChatEncoder(TestChatTokenizer(), TASK_B, 512)
        collator = CompletionCollator(0)
        dataset = CompletionDataset(
            ["one", "two", "three", "four", "five"], encoder, [0, 1, 2, 3, 4]
        )
        model = DistributedDataParallel(
            tiny_model(), broadcast_buffers=False, find_unused_parameters=False
        )
        loader, sampler = create_train_loader(dataset, collator, config, runtime, 42)
        sampler.set_epoch(1)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        train_llm_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            torch.amp.GradScaler("cuda", enabled=False),
            config,
            runtime,
            1,
        )
        for count in (5, 1):
            evaluation = CompletionDataset(["eval"] * count, encoder)
            distributed = score_candidates(
                model, evaluation, collator, config, runtime, "DDP eval"
            )
            single = score_candidates(
                model.module, evaluation, collator, config, cpu_runtime(), "single eval"
            )
            np.testing.assert_allclose(
                distributed["probabilities"], single["probabilities"], atol=1e-6
            )
        weights = torch.cat(
            [p.detach().flatten() for p in model.parameters() if p.requires_grad]
        )
        gathered = gather_python_objects(weights.tolist(), runtime)
        np.testing.assert_allclose(gathered[0], gathered[1], atol=1e-7)
        if rank == 0:
            print("LLM_DDP_SMOKE_OK", flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
