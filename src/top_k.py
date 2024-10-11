import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TopKLogitsWarper

def top_k_sampling_with_temperature(logits, top_k=50, temperature=1.0, min_tokens_to_keep=1, return_logits=False):
    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    if top_k == 0 or top_k > logits.shape[-1]:
        top_k = logits.shape[-1]

    top_k = max(top_k, min_tokens_to_keep)

    # Top-k transformation
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)

    indices_to_remove = logits < top_k_logits[..., -1]
    masked_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    masked_probs = torch.softmax(masked_logits, dim=-1)

    # Sampling
    sample_token = torch.multinomial(masked_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, masked_logits
    return sample_token

def test_top_k_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, top_k_logits = top_k_sampling_with_temperature(logits, top_k=500, temperature=1.0, return_logits=True)
    print(top_k_logits)
    logit_processor = TopKLogitsWarper(top_k=500, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)

    torch.testing.assert_close(top_k_logits, logits_processed)
    
if __name__ == "__main__":
    test_top_k_sampling()