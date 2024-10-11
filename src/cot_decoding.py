import torch
from transformers import StaticCache
from torch.nn.attention import SDPBackend, sdpa_kernel

def decode_one_token(model, cur_token, cache_position, past_key_values, k=1):
    output = model(input_ids=cur_token, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    logits = output.logits[:, -1, :]

    kth_largest_index = torch.topk(logits, k, dim=-1).indices[..., -1]

    return kth_largest_index

def return_first_answer_only(output):
    return output.split("Q:")[0].strip()

def generate_with_cot_decoding(model, tokenizer, device, prompt, max_new_tokens, intial_token_k=1):
    # Encode the initial prompt
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)

    seq_length = initial_prompt_seq.shape[-1]
    
    # Initialize static KV Cache
    past_key_values = StaticCache(
        config=model.config,
        batch_size=1,
        max_cache_len=seq_length + (max_new_tokens * 2),
        device=model.device,
        dtype=model.dtype
    )

    cache_position = torch.arange(seq_length, device=device)

    # Tensor to store generated ids
    generated_ids = torch.zeros(seq_length + max_new_tokens + 1, dtype=torch.int, device=device)
    generated_ids[cache_position] = initial_prompt_seq.to(torch.int)

    # First step: Process the entire prompt and sample the first token
    sample_token = decode_one_token(model, initial_prompt_seq, cache_position, past_key_values, intial_token_k)

    generated_ids[seq_length] = sample_token.int()

    # Initialize variables
    count_new_tokens = 1
    cache_position = torch.tensor([seq_length + 1], device=device)

    # Subsequent steps: Pass only the last generated token with past_key_values
    while count_new_tokens < max_new_tokens:
        with sdpa_kernel(SDPBackend.MATH):
            next_token_id = torch.tensor([[sample_token]], device=device, dtype=torch.int)
            sample_token = decode_one_token(model, next_token_id, cache_position, past_key_values, k=1)
            
        generated_ids[cache_position] = sample_token.int()

        # Break if EOS is generated
        if sample_token.item() == tokenizer.eos_token_id:
            break

        count_new_tokens += 1
        cache_position += 1

    # Decode the generated tokens
    generated_ids = generated_ids[seq_length: seq_length + count_new_tokens]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return return_first_answer_only(output)