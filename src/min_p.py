import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MinPLogitsWarper

def min_p_sampling_with_temperature(logits, min_p=0.1, temperature=1.0, min_tokens_to_keep=1, return_logits=False):
    assert 0 <= min_p <= 1, "min_p must be between 0 and 1"

    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Min-p transformation
    probs = torch.softmax(logits, dim=-1)
    p_max = torch.max(probs, dim=-1).values
    p_scaled = min_p * p_max
    min_p_mask = probs < p_scaled

    sorted_indices = torch.argsort(logits, descending=True, dim=-1)
    sorted_indices_to_remove = min_p_mask.gather(-1, sorted_indices)
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

    min_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    min_p_probs = torch.softmax(min_p_logits, dim=-1)

    # Sampling
    sample_token = torch.multinomial(min_p_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, min_p_logits
    return sample_token

def generate_with_min_p_sampling(model, tokenizer, device, prompt, max_new_tokens, min_p=0.9, temperature=1.0):
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

        sample_token = min_p_sampling_with_temperature(logits, min_p=min_p, temperature=temperature)

        if sample_token.item() == tokenizer.eos_token_id:
            break

        # Append the new token
        generated_tokens.append(sample_token.item())
        count_new_tokens += 1

    # Decode the generated tokens
    fin_prompt_new_seq = torch.tensor([generated_tokens], device=device)
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    
    return output

def test_min_p_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, min_p_logits = min_p_sampling_with_temperature(logits, min_p=0.1, temperature=1.0, return_logits=True)
    print(min_p_logits)
    print(min_p_logits[torch.where(min_p_logits != float('-inf'))])
    logit_processor = MinPLogitsWarper(min_p=0.1, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)
    print(logits_processed[torch.where(logits_processed != float('-inf'))])

    torch.testing.assert_close(min_p_logits, logits_processed)
    
if __name__ == "__main__":
    test_min_p_sampling()