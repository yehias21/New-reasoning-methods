import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EpsilonLogitsWarper)

# Based on https://arxiv.org/abs/2210.15191 - Truncation Sampling as Language Model Desmoothing
# Best values of epsilon for epsilon sampling are between 3e-4 and 9e-4 (Table 5 in Appendix A.2)

def epsilon_sampling_with_temperature(logits, epsilon=6e-4, temperature=1.0, min_tokens_to_keep=1, return_logits=False):
    assert 0 <= epsilon <= 1, "epsilon must be between 0 and 1"

    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Filtering
    probs = torch.softmax(logits, dim=-1)
    indices_to_remove = probs < epsilon

    # Keep top min_tokens_to_keep tokens
    top_k = min(min_tokens_to_keep, logits.shape[-1])
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    top_min_tokens_mask = logits < top_k_logits[..., -1]

    indices_to_remove = torch.logical_and(indices_to_remove, top_min_tokens_mask)
    masked_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    masked_probs = torch.softmax(masked_logits, dim=-1)

    # Sampling
    sample_token = torch.multinomial(masked_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, masked_logits
    return sample_token

def test_epsilon_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, epsilon_logits = epsilon_sampling_with_temperature(logits, epsilon=0.1, temperature=1.0, min_tokens_to_keep=1, return_logits=True)
    print(epsilon_logits)
    print(epsilon_logits[torch.where(epsilon_logits != float('-inf'))])
    logit_processor = EpsilonLogitsWarper(epsilon=0.1, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)
    print(logits_processed[torch.where(logits_processed != float('-inf'))])

    torch.testing.assert_close(epsilon_logits, logits_processed)
    
if __name__ == "__main__":
    test_epsilon_sampling()