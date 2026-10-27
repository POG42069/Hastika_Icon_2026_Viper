"""Exercise both Task B phases and completed-run resume on two CPU/Gloo ranks."""
from __future__ import annotations
import os
from pathlib import Path
import socket
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(rank, port, directory):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE='2', USE_LIBUV='0')
    from contextlib import ExitStack
    from dataclasses import replace
    import json
    from unittest.mock import patch
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from llm_config import TASK_B, LLM_TRAINING_B
    from src.distributed import RuntimeContext, cleanup_runtime
    from src.llm_training import run_llm_task
    from tests.test_task_b_khangtest import TaskBKhangtestTests

    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='env://')
    runtime = RuntimeContext(torch.device('cpu'), rank, rank, 2, True)
    try:
        root = Path(directory)
        task = replace(TASK_B, selection_seeds=(42,52))
        config = replace(LLM_TRAINING_B, max_epochs=2, num_workers=0)
        fixture = TaskBKhangtestTests()
        if rank == 0:
            fixture.write_data(root, task, gold=True)
        torch.distributed.barrier()
        with ExitStack() as stack:
            loader = fixture.boundary(stack, root)
            stack.enter_context(patch('src.task_b_training.initialize_runtime', return_value=runtime))
            stack.enter_context(patch('src.task_b_training._wrap_for_ddp', side_effect=lambda model, ctx:
                                     DDP(model, broadcast_buffers=False, find_unused_parameters=False)))
            run_llm_task(task, config, auto_resume=False)
            assert loader.call_count == 4, loader.call_count
            output = root/'outputs'/task.output_subdir
            summary = json.loads((output/'run_summary.json').read_text())
            assert summary['world_size'] == 2
            assert [item['epochs'] for item in summary['full_data_retrains']] == [
                item['best_epoch'] for item in summary['epoch_selection']['seeds']]
            assert all(item['selection_training_rows']==24 and item['validation_rows']==6
                       for item in summary['epoch_selection']['seeds'])
            assert (output/'task_b_submission.zip').is_file()
            run_llm_task(task, config, auto_resume=False)
            assert loader.call_count == 4
        if rank == 0:
            print('TASK_B_SELECTION_DDP_SMOKE_OK', flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(port,directory), nprocs=2, join=True)
