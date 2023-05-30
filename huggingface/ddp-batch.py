import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import pipeline
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# def pad_to_desired_length(list_of_tensors, desired_length):

#     # Pad tensors with zeros to the maximum length
#     # Print the shape of each tensor
#     for tensor in list_of_tensors:
#         print("Tensor shape:", tensor.shape)
        
#     padded_tensors = torch.stack([
#         torch.nn.functional.pad(tensor, (0, desired_length - tensor.shape[1]), value=0)
#         for tensor in list_of_tensors
#     ], dim=0)

#     return padded_tensors.squeeze()


def pad_to_desired_length(list_of_tensors, desired_length):
    # Pad tensors with zeros to the desired length
    # Print the shape of each tensor
    for tensor in list_of_tensors:
        print("Tensor shape:", tensor.shape)
    
    padded_tensors = []
    for tensor in list_of_tensors:
        if len(tensor.shape) == 1:
            # It's a 1D tensor
            pad_size = desired_length - tensor.shape[0]
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
        elif len(tensor.shape) == 2:
            # It's a 2D tensor
            pad_size = desired_length - tensor.shape[1]
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
        else:
            raise ValueError("This function can only handle 1D and 2D tensors.")
        
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=0)

def create_evenly_distributed_chunks(inputs_list, n_gpus):
    total_data_size = len(inputs_list)
    # Calculate the size of each chunk
    per_gpu_size = math.ceil(total_data_size / n_gpus)

    # calculate new total size to ensure equal-sized chunks
    new_total_size = per_gpu_size * n_gpus
    if new_total_size > total_data_size:
        # duplicate last element to fill in remaining space
        inputs_list += [inputs_list[-1]] * (new_total_size - total_data_size)

    chunks = np.array_split(np.array(inputs_list), n_gpus)
    return chunks

def main():

    # Initialize the distributed training environment
    torch.distributed.init_process_group(backend='nccl')

    # Determine the number of available GPUs and the current GPU rank
    rank = int(os.getenv('RANK', '0')) # when using torchrun, environment variables like RANK and WORLD_SIZE are automatically set,
    n_gpus = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = f"cuda:{local_rank}"
    print(rank, n_gpus, local_rank)
    torch.cuda.set_device(local_rank) # RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)
    
    
    
    # read data
    test_file = '../new_test_fixed.tsv'
    df = pd.read_csv(test_file, sep="\t")
    print(f"Initial data size: {df.shape}")
    # rm empty cells
    df['input'].replace('', np.nan, inplace=True)
    df.dropna(subset=['input'], inplace=True)
    docs_to_remove = ["2021_Apr_SeattleTBD-0eaa0c.pdf", "2021_Jan_KingCoLodging-7f1cff.pdf", "lincoln-fee8ff.html", "sales-and-use-tax-rates-effective-july-1-2020-9ce3f4.html", "vermilion-e5c8f1.html"]
    df = df[~df['document_url'].isin(docs_to_remove)]
    
#     df = df.sample(16, random_state=1)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    print(f"Final data size after removing empty cells: {df.shape}")
    input_data = list(df['input'])


    # tokenizer and model
    print("Loading Model")
    model_name = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained('../origin-model/')
    model = model.to(device)
    model.eval()
    print("Loading Model Done!")
    # Wrap model in DistributedDataParallel for multi-GPU training
    model = DistributedDataParallel(model, device_ids=[local_rank])


    chunks = create_evenly_distributed_chunks(input_data, n_gpus)

    # data on this gpu
    input_chunk = chunks[rank].tolist()
    print(f"Rank {rank} are dealing with data size {len(input_chunk)}")
        
    start_time = time.time()
    local_outputs = []
    batch_size = 3
    for i in range(0, len(input_chunk), batch_size):
        print(local_rank, i)
        batch = input_chunk[i:i+batch_size]
        encoding = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=4096).to(device)
        
        with torch.no_grad():
            generated_ids = model.module.generate(**encoding, max_length=4096)
        
        local_outputs.extend(generated_ids)
    

    if local_rank == 0:
        print("Final local outputs: ", local_outputs)
        
    local_outputs = pad_to_desired_length(local_outputs, desired_length=4096)
    print(f"Rank {rank} output shape: {local_outputs.shape}")

    # Synchronize all GPUs
    dist.barrier()


    # All-gather the decoded results from all GPUs
    gathered_outputs_list = [torch.zeros_like(local_outputs) for _ in range(n_gpus)]
    dist.all_gather(gathered_outputs_list, local_outputs) # collective operations; every process needs to call dist.all_gather, not just the one with rank 0. Each process contributes its local tensor and collects the tensors from all other processes.

    if rank == 0:
        print(f"Gathered output list: {gathered_outputs_list}")

        # convert output to list 
        print(f"cat started rank {rank}")
        final_outputs = torch.cat(gathered_outputs_list, dim=0).tolist()
        print(f"cat finished rank {rank} \n")

        # Decode gathered outputs to text
        decoded_outputs = []
        for output in final_outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            decoded_outputs.append(decoded_output)


        total_time = time.time() - start_time

        print("Gathered Outputs:")
        print(decoded_outputs)
        print(f"Total time: {total_time} seconds; {total_time / 60} mins")

        df['prediction'] = decoded_outputs
        df.to_csv(f"hf-parallel-batch-8-inference.tsv", sep='\t', index=False)

    torch.distributed.destroy_process_group()



if __name__ == '__main__':
    main()
