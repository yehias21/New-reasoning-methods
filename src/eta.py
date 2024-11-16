import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EtaLogitsWarper

# Based on https://arxiv.org/abs/2210.15191 - Truncation Sampling as Language Model Desmoothing
# Best values of epsilon for eta sampling are between 3e-4 and 2e-3 (Table 5 in Appendix A.2)

def eta_sampling_with_temperature(logits, epsilon=6e-4, temperature=1.0, min_tokens_to_keep=1, div_by_log2=True, return_logits=False):
    assert 0 <= epsilon <= 1, "epsilon must be between 0 and 1"

    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Filtering
    probs = torch.softmax(logits, dim=-1)
    entropy = -1 * torch.nansum(probs * torch.log(probs), dim=-1)

    # HF implementation doesn't divide by log(2)
    if div_by_log2:
        entropy = entropy / math.log(2)   
    
    epsilon = torch.tensor(epsilon).to(logits.device)
    alpha = torch.sqrt(epsilon)
    eta = torch.min(epsilon, alpha * torch.exp(-entropy))
    indices_to_remove = probs < eta

    # Keep top min_tokens_to_keep tokens
    top_k = min(min_tokens_to_keep, logits.shape[-1])
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    top_min_tokens_mask = logits < top_k_logits[..., -1] # Indices in top-k are marked False

    # Don't remove tokens that are already in the top k
    indices_to_remove = torch.logical_and(indices_to_remove, top_min_tokens_mask)
    masked_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    masked_probs = torch.softmax(masked_logits, dim=-1)

    # Sampling
    sample_token = torch.multinomial(masked_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, masked_logits
    return sample_token

def test_eta_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, eta_logits = eta_sampling_with_temperature(logits, epsilon=0.1, temperature=1.0, min_tokens_to_keep=1, div_by_log2=False, return_logits=True)
    print(eta_logits)
    print(eta_logits[torch.where(eta_logits != float('-inf'))])
    logit_processor = EtaLogitsWarper(epsilon=0.1, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)
    print(logits_processed[torch.where(logits_processed != float('-inf'))])

    torch.testing.assert_close(eta_logits, logits_processed)
    
if __name__ == "__main__":
    test_eta_sampling()