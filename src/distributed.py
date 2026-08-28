"""Utilities for transparent single-GPU and two-GPU execution.

The public entry points remain simple (``python Train_A.py`` and
``python Train_B.py``).  When two GPUs are visible, the parent process
relaunches the same script with ``torchrun`` and DistributedDataParallel (DDP).
Each worker receives the configured per-GPU batch size.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from config import TrainingConfig


@dataclass(frozen=True)
class RuntimeContext:
    """Distributed state used by the training loop."""

    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main_process(self) -> bool:
        """Return True only for the process allowed to write artifacts."""

        return self.rank == 0


def relaunch_with_torchrun_if_needed(config: TrainingConfig) -> None:
    """Relaunch the current script with one DDP process per visible GPU.

    This function returns immediately in a torchrun worker, on CPU, or when
    only one GPU is available.  In a normal Kaggle T4 x2 session, the original
    process waits for two workers and exits with their return code.
    """

    already_distributed = "LOCAL_RANK" in os.environ
    visible_gpu_count = torch.cuda.device_count()
    requested_gpu_count = min(visible_gpu_count, config.max_gpus)

    if (
        already_distributed
        or not config.use_all_available_gpus
        or requested_gpu_count <= 1
    ):
        return

    script_path = str(Path(sys.argv[0]).resolve())
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={requested_gpu_count}",
        script_path,
        *sys.argv[1:],
    ]
    print(
        f"Detected {visible_gpu_count} GPU(s). Starting DDP with "
        f"{requested_gpu_count} process(es): {' '.join(command)}",
        flush=True,
    )
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


def initialize_runtime() -> RuntimeContext:
    """Initialize NCCL DDP when launched by torchrun, otherwise run locally."""

    distributed = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            # Gloo keeps manual CPU torchrun smoke tests possible. The normal
            # Kaggle T4 x2 path always uses NCCL.
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return RuntimeContext(
            device=device,
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            distributed=True,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return RuntimeContext(
        device=device,
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )


def synchronize(runtime: RuntimeContext) -> None:
    """Wait until every worker reaches the same point."""

    if runtime.distributed:
        dist.barrier()


def gather_python_objects(value: object, runtime: RuntimeContext) -> list[object]:
    """Gather one picklable object from every worker onto every worker."""

    if not runtime.distributed:
        return [value]
    gathered: list[object] = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def cleanup_runtime(runtime: RuntimeContext) -> None:
    """Close the DDP process group cleanly."""

    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
