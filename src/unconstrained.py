import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TemperatureLogitsWarper

def unconstrained_sampling_with_temperature(logits, temperature=1.0, return_logits=False):
    logits = logits / (temperature + 1e-10)
    probs = torch.softmax(logits, dim=-1)
    sample_token = torch.multinomial(probs, num_samples=1)[0]

    if return_logits:
        return sample_token, logits
    return sample_token

def generate_with_unconstrained_sampling(model, tokenizer, device, prompt, max_new_tokens, temperature=1.0):
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

def test_unconstrained_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    logits = model(input_ids).logits[:, -1, :]
    print(logits)

    sample_token, unconstrained_logits = unconstrained_sampling_with_temperature(logits, temperature=0.9, return_logits=True)
    print(unconstrained_logits)
    logit_processor = TemperatureLogitsWarper(temperature=0.9)
    logits_processed = logit_processor(input_ids, logits)
    print(logits_processed)

    torch.testing.assert_close(unconstrained_logits, logits_processed)
    
if __name__ == "__main__":
    test_unconstrained_sampling()