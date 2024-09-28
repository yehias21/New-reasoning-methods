import torch
from .utils import sampling_with_temperature

def unconstrained_sampling(model, tokenizer, device, prompt, max_new_tokens, temperature=1.0):
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []
    
    count_new_tokens = 0

    while count_new_tokens < max_new_tokens:
        logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sampling_with_temperature(logits, temperature=temperature)
        if sample_token == tokenizer.eos_token_id:
            break
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        out_logits.append(logits)
        count_new_tokens += 1

    fin_prompt_new_seq = fin_prompt_seq[:, initial_prompt_seq.shape[1]:]
    output = tokenizer.decode(fin_prompt_new_seq[0], skip_special_tokens=True)
    out_logits = torch.stack(out_logits, dim=1)
    
    return output, fin_prompt_seq, out_logits