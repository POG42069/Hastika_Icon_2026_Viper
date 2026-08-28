"""Launch the DDP smoke worker without depending on the torchrun CLI."""

from __future__ import annotations

import os
import socket

import torch.multiprocessing as mp


def find_free_port() -> int:
    """Reserve an ephemeral localhost port for the two test workers."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def worker(rank: int, world_size: int, port: int) -> None:
    """Populate torchrun-compatible variables and execute one DDP worker."""

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["USE_LIBUV"] = "0"
    from tests.ddp_smoke_worker import main

    main()


def main() -> None:
    """Spawn two CPU workers and wait for both to finish successfully."""

    world_size = 2
    mp.spawn(worker, args=(world_size, find_free_port()), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
