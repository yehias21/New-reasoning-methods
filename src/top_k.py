import torch

def top_k_sampling_with_temperature(logits, top_k, temperature=1.0):
    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Top-k transformation
    if top_k == 0 or top_k > logits.shape[-1]:
        top_k = logits.shape[-1]

    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    # Sampling
    sample_index = torch.multinomial(top_k_probs, num_samples=1)
    sample_token = top_k_indices.gather(-1, sample_index)[0]
    return sample_token

def top_k_sampling(model, tokenizer, device, prompt, max_new_tokens, top_k=10, temperature=1.0):
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Initialize variables
    generated_tokens = []
    count_new_tokens = 0
    past_key_values = None  # Initialize past_key_values

    while count_new_tokens < max_new_tokens:
        if past_key_values is None:
            # First step: Process the entire prompt
            outputs = model(input_ids=initial_prompt_seq)
        else:
            # Subsequent steps: Pass only the last generated token with past_key_values
            next_token_id = torch.tensor([[generated_tokens[-1]]], device=device, dtype=torch.long)
            outputs = model(input_ids=next_token_id, past_key_values=past_key_values)
        
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values  # Update past_key_values

        sample_token = top_k_sampling_with_temperature(logits, top_k=top_k, temperature=temperature)

        if sample_token.item() == tokenizer.eos_token_id:
            break

        # Append the new token
        generated_tokens.append(sample_token.item())
        count_new_tokens += 1

    # Decode the generated tokens
    fin_prompt_new_seq = torch.tensor([generated_tokens], device=device)
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    
    return output