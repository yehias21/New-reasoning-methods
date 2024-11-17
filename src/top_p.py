import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TopPLogitsWarper


def top_p_sampling_with_temperature(logits, top_p=0.9, temperature=1.0, min_tokens_to_keep=1, return_logits=False):
    # Temperature scaling first
    logits = logits / (temperature + 1e-10)
    
    # Top-p transformation - Off by one error in this implementation
    # For e.g. if probs are 0.1, 0.2, 0.3, 0.1, 0.15, 0.1, 0.05
    # If sorted in descending, indices are [2, 1, 4, 0, 3, 5, 6]
    # cumsum sorted probs = 0.3, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0
    # sorted indices to remove = [False, False, False, False, False, True, True]
    # This is incorrect because as per the definition of top-p, 
    # it requires smallest possible sets of top words such that the sum of their probability is >= top_p
    # Here we are removing the last two words, but including the second last word is needed to have the cumsum prob >= top_p

    # sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    # sorted_probs = torch.softmax(sorted_logits, dim=-1)
    # cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # sorted_indices_to_remove = cumsum_sorted_probs > top_p
    # sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Top-p transformation - Correct Implementation
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumsum_sorted_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = False

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    top_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    top_p_probs = torch.softmax(top_p_logits, dim=-1)
    
    # Sampling
    sample_token = torch.multinomial(top_p_probs, num_samples=1)[0]

    if return_logits:
        return sample_token, top_p_logits
    return sample_token

def test_top_p_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, top_p_logits = top_p_sampling_with_temperature(logits, top_p=0.9, temperature=1.0, return_logits=True)
    print(top_p_logits)
    logit_processor = TopPLogitsWarper(top_p=0.9, min_tokens_to_keep=1)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)

    torch.testing.assert_close(top_p_logits, logits_processed)
    
if __name__ == "__main__":
    test_top_p_sampling()