import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TemperatureLogitsWarper)


def unconstrained_sampling_with_temperature(logits, temperature=1.0, return_logits=False):
    logits = logits / (temperature + 1e-10)
    probs = torch.softmax(logits, dim=-1)
    sample_token = torch.multinomial(probs, num_samples=1)[0]

    if return_logits:
        return sample_token, logits
    return sample_token

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