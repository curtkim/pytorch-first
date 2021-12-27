#!/usr/bin/env python
import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_reduce(rank, world_size):
    """ All-Reduce example."""
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"rank {rank}, world_size {world_size}, pid {os.getpid()}  has data ", tensor[0])
    logging.info(f"rank {rank}, world_size {world_size}, pid {os.getpid()}")


def init_process(rank, size, fn, backend='gloo'):
    print('init_process', os.getpid())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"debug_{rank}.log")
        ]
    )
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    print('start', os.getpid())
    size = 2
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('end', os.getpid())

'''
start 247162
init_process 247209
init_process 247210
rank 0, world_size 2, pid 247209  has data  tensor(2.)  
rank 1, world_size 2, pid 247210  has data  tensor(2.)
end 247162

---
debug_0.log
---
2021-12-27 10:38:30,152 [INFO] Added key: store_based_barrier_key:1 to store for rank: 0
2021-12-27 10:38:30,152 [INFO] Rank 0: Completed store-based barrier for 2 nodes.
2021-12-27 10:38:30,164 [INFO] Added key: store_based_barrier_key:2 to store for rank: 0
2021-12-27 10:38:30,164 [INFO] Rank 0: Completed store-based barrier for 2 nodes.
2021-12-27 10:38:30,193 [INFO] rank 0, world_size 2, pid 247742

---
debug_1.log
---
2021-12-27 10:38:30,152 [INFO] Added key: store_based_barrier_key:1 to store for rank: 1
2021-12-27 10:38:30,152 [INFO] Rank 1: Completed store-based barrier for 2 nodes.
2021-12-27 10:38:30,164 [INFO] Added key: store_based_barrier_key:2 to store for rank: 1
2021-12-27 10:38:30,164 [INFO] Rank 1: Completed store-based barrier for 2 nodes.
2021-12-27 10:38:30,193 [INFO] rank 1, world_size 2, pid 247743
'''