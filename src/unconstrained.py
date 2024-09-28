import torch
from .utils import sampling_with_temperature
from transformers import LogitsProcessor, LogitsProcessorList

def unconstrained_sampling(model, tokenizer, device, prompt, max_new_tokens, temperature=1.0):
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    
    count_new_tokens = 0

    while count_new_tokens < max_new_tokens:
        logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sampling_with_temperature(logits, temperature=temperature)
        if sample_token == tokenizer.eos_token_id:
            break
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        count_new_tokens += 1

    fin_prompt_new_seq = fin_prompt_seq[:, initial_prompt_seq.shape[1]:]
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    
    return output

class UnconstrainedSamplingLogitProcessor(LogitsProcessor):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, input_ids, scores):
        scores = scores / (self.temperature + 1e-10)
        return scores

def unconstrained_sampling_with_logit_processor(model, tokenizer, device, prompt, max_new_tokens, temperature=1.0):
    initial_prompt_seq = tokenizer(prompt, return_tensors="pt").to(device)
    unconstrained_sampling_logit_processor = UnconstrainedSamplingLogitProcessor(temperature=temperature)
    logits_processor = LogitsProcessorList([unconstrained_sampling_logit_processor])

    fin_prompt_seq = model.generate(**initial_prompt_seq, logits_processor=logits_processor, max_new_tokens=max_new_tokens)

    fin_prompt_new_seq = fin_prompt_seq[:, initial_prompt_seq.input_ids.shape[1]:]
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    
    return output