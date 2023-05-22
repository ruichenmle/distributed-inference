import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, T5ForConditionalGeneration

def main():
    # Initialize distributed training environment
    dist.init_process_group(backend='nccl')

    # Set up GPU-specific settings
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Load tokenizer and model on each GPU
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    model = model.to(device)

    # Wrap model in DistributedDataParallel for multi-GPU training
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # Set input data for each GPU
    input_data = [
        "Translate English to French: Hello, how are you? I feel great today, and let's go hiking",
        "Translate English to French: What is your name?",
        "translate English to German: Very bad.",
             "translate English to German: How long is it?", 
             "translate English to German: I don't like the cat.",
             "translate English to German: How is the movie?",
             "translate English to German: Let's go.",
             "translate English to German: Can we go now?"
    ]

    # Tokenize input on each GPU
    inputs = tokenizer.batch_encode_plus(input_data, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform inference on each GPU
    outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
    print(f"Rank: {local_rank} shape: {outputs.shape}")

    # Gather outputs from all GPUs
    gathered_outputs = [torch.zeros_like(outputs) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_outputs, outputs)
    if local_rank == 0:
        print(gathered_outputs)
        print(len(gathered_outputs))
        print(gathered_outputs[0])
        print("****"*10)


    if local_rank == 0:
        # Flatten the list of tensors
        gathered_outputs = gathered_outputs[0]
        # Convert the tensor of output ids into a list of id sequences
        gathered_outputs_list = [output.tolist() for output in gathered_outputs]

        # Decode each id sequence into a string
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in gathered_outputs_list]

        # Print the decoded outputs
        for decoded_output in decoded_outputs:
            print(f"Decoded output: {decoded_output}")


    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
