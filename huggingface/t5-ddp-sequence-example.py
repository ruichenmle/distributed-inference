import os, time, math
import numpy as np
import torch
import torch.distributed as dist
from transformers import pipeline
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def pad_to_desired_length(list_of_tensors, desired_length):

    # Pad tensors with zeros to the maximum length
    padded_tensors = torch.stack([
        torch.nn.functional.pad(tensor, (0, desired_length - tensor.shape[1]), value=0)
        for tensor in list_of_tensors
    ], dim=0)

    return padded_tensors.squeeze()

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


    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    # Wrap model in DistributedDataParallel for multi-GPU training
    model = DistributedDataParallel(model, device_ids=[local_rank])


    input_data = ["question: Which is capital city of India? context: New Delhi is India's capital", 
                "translate English to French: New Delhi is India's capital",
                "translate English to German: The house is wonderful.",
                "translate English to German: I love you.",
                "translate English to German: I like the cat.",
                "translate English to German: How old are you?",
                "translate English to German: Very good.",
                "translate English to German: How far is it?",
                "translate English to German: I like the dog.",
                "translate English to German: What is it?",
                "translate English to German: Very bad.",
                "translate English to German: How long is it?", 
                "translate English to German: I don't like the cat.",
                "translate English to German: How is the movie?",
                "translate English to German: Let's go.",
                "translate English to German: Can we go now?"]  

    chunks = create_evenly_distributed_chunks(input_data, n_gpus)


    # data on this gpu
    input_chunk = chunks[rank].tolist()
    if rank == 0:
        print(f"Rank {rank} are dealing with data size {len(input_chunk)}")

    tokenized_chunk = tokenizer(input_chunk, return_tensors="pt", padding='longest', truncation=True, max_length=4096)

    input_ids_chunk = tokenized_chunk['input_ids'].to(device)
    attention_mask_chunk = tokenized_chunk['attention_mask'].to(device)


    start_time = time.time()

    local_outputs = []
    for i in range(len(input_chunk)):
        input_id = input_ids_chunk[i].unsqueeze(0)
        mask = attention_mask_chunk[i].unsqueeze(0)
        outputs = model.module.generate(input_ids=input_id, 
                                attention_mask=mask,
                                max_length=4096) # only current gpu
        
        local_outputs.append(outputs)
        
        del input_id
        del mask
        del outputs
        

    del tokenized_chunk
    del input_ids_chunk
    del attention_mask_chunk
    del model
        
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

    #     df['prediction'] = decoded_outputs
    #     df.to_csv(f"rui-hf-parallel-inference.tsv", sep='\t', index=False)

    torch.distributed.destroy_process_group()



if __name__ == '__main__':
    main()
