import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import pipeline
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


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

    # Determine the number of available GPUs and the current GPU rank
    rank = int(os.getenv('RANK', '0')) # when using torchrun, environment variables like RANK and WORLD_SIZE are automatically set,
    n_gpus = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = f"cuda:{local_rank}"
    print(rank, n_gpus, local_rank)
    
    
    
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
    model = T5ForConditionalGeneration.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    exit(0)
    # ValueError: `.to` is not supported for `4-bit` or `8-bit` models. 
    # Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    # model = model.to(device)
    # model.eval()
    print("Loading Model Done!")
    # data on this gpu

    start_time = time.time()
    local_outputs = []
    batch_size = 1
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
        encoding = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=4096).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**encoding, max_length=4096)
        
        local_outputs.extend(generated_ids)
    

    local_outputs = pad_to_desired_length(local_outputs, desired_length=4096)
    print(f"Rank {rank} output shape: {local_outputs.shape}")

    # Decode gathered outputs to text
    decoded_outputs = []
    for output in local_outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        decoded_outputs.append(decoded_output)


    total_time = time.time() - start_time

    print("Gathered Outputs:")
    print(f"Total time: {total_time} seconds; {total_time / 60} mins")
    print(f"Total time: {total_time / len(input_data)} seconds/sample")

    df['prediction'] = decoded_outputs
    df.to_csv(f"hf-parallel-batch-{batch_size}-inference.tsv", sep='\t', index=False)




if __name__ == '__main__':
    main()
