import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TypicalLogitsWarper

def typical_sampling_with_temperature(logits, typical_p_mass=0.9, temperature=1.0, min_tokens_to_keep=1, return_logits=False):
    assert 0 <= typical_p_mass <= 1, "typical_p_mass must be between 0 and 1"

    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Typical sampling transformation
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.nansum(probs * log_probs, dim=-1, keepdim=True)

    neg_log_probs = -log_probs
    entropy_neg_log_probs_dist = torch.abs(entropy - neg_log_probs)
    entropy_neg_log_probs_dist_sorted_val, entropy_neg_log_probs_dist_sorted_indices = torch.sort(entropy_neg_log_probs_dist, descending=False, dim=-1)
    sorted_logits = logits.gather(-1, entropy_neg_log_probs_dist_sorted_indices)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)

    last_ind = (cumsum_sorted_probs < typical_p_mass).sum(dim=1)
    last_ind.clamp_(max=sorted_logits.shape[-1] - 1)
    # Comparison is done on the entropy-neg-log-probs distance rather than the sorted logits since the entrop-neg-log-probs is what is being sorted
    sorted_indices_to_remove = entropy_neg_log_probs_dist_sorted_val > entropy_neg_log_probs_dist_sorted_val.gather(1, last_ind.view(-1, 1))
    sorted_indices_to_remove[..., : min_tokens_to_keep] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(-1, entropy_neg_log_probs_dist_sorted_indices, sorted_indices_to_remove)
    typical_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    typical_p_probs = torch.softmax(typical_p_logits, dim=-1)
    
    # Sampling
    sample_token = torch.multinomial(typical_p_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, typical_p_logits
    return sample_token

def generate_with_typical_sampling(model, tokenizer, device, prompt, max_new_tokens, typical_p_mass=0.9, temperature=1.0):
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

        sample_token = typical_sampling_with_temperature(logits, typical_p_mass=typical_p_mass, temperature=temperature)

        if sample_token.item() == tokenizer.eos_token_id:
            break

        # Append the new token
        generated_tokens.append(sample_token.item())
        count_new_tokens += 1

    # Decode the generated tokens
    fin_prompt_new_seq = torch.tensor([generated_tokens], device=device)
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    
    return output

def test_typical_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, typical_p_logits = typical_sampling_with_temperature(logits, typical_p_mass=0.9, temperature=1.0, return_logits=True)
    print(typical_p_logits)
    logit_processor = TypicalLogitsWarper(mass=0.9, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)

    torch.testing.assert_close(typical_p_logits, logits_processed)
    
if __name__ == "__main__":
    test_typical_sampling()