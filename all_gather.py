# run the script:
# torchrun --nproc_per_node=8 --nnode=1 all_gather.py 

import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.getenv('RANK', '0')) # RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    print(rank, world_size, local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    t = torch.rand(1).to(local_rank) # RuntimeError: Tensors must be CUDA and dense
    gather_t = [torch.ones_like(t) for _ in range(world_size)]
    dist.all_gather(gather_t, t)

    # dist.barrier()

    if rank == 0:
        print(rank, t, gather_t)

if __name__ == '__main__':
    main()
