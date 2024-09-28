import torch
from transformers import LogitsProcessor, LogitsProcessorList

def unconstrained_sampling_with_temperature(logits, temperature=1.0):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    sample_token = torch.multinomial(probs, num_samples=1)[0]
    return sample_token

def unconstrained_sampling(model, tokenizer, device, prompt, max_new_tokens, temperature=1.0):
    # Encode the initial prompt
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

        sample_token = unconstrained_sampling_with_temperature(logits, temperature=temperature)

        if sample_token.item() == tokenizer.eos_token_id:
            break

        # Append the new token
        generated_tokens.append(sample_token.item())
        count_new_tokens += 1

    # Decode the generated tokens
    fin_prompt_new_seq = torch.tensor([generated_tokens], device=device)
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)

    return output