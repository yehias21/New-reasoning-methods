import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Based on https://arxiv.org/abs/2302.01318 - Accelerating Large Language Model Decoding with Speculative Sampling

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs

# Same as unconstrained.py
def unconstrained_sampling_with_temperature(logits, temperature=1.0, return_logits=False):
    logits = logits / (temperature + 1e-10)
    probs = torch.softmax(logits, dim=-1)
    sample_token = torch.multinomial(probs, num_samples=1)[0]

    if return_logits:
        return sample_token, logits
    return sample_token

def sample_from_draft_model(model, initial_prompt_seq, new_tokens, temperature=1.0):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token, sample_token_logits = unconstrained_sampling_with_temperature(sample_token_logits, temperature=temperature, return_logits=True)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits

def speculative_sampling(target_model, draft_model, tokenizer, device, prompt, max_new_tokens, lookahead=4, temperature=1.0, debug=False):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    draft_accepted_tokens = 0

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    target_len = n + max_new_tokens

    end_flag = 0

    while n < target_len:
        if end_flag == 1:
            break

        n_orig = n
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        
        for t in range(lookahead):
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                n += 1
                draft_accepted_tokens += 1

                if fin_prompt_seq[..., -1] == tokenizer.eos_token_id:
                    end_flag = 1
                    break

            ## Rejection
            else:
                if debug:
                    print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

                new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id[None,...]], dim=-1)
                accepted_flag = 0

                if fin_prompt_seq[..., -1] == tokenizer.eos_token_id:
                    end_flag = 1
                
                break

        if accepted_flag == 1 and end_flag == 0:
            if debug:
                print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

            last_target_logits = target_logits[:, -1, :]
            sample_token = unconstrained_sampling_with_temperature(last_target_logits, temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)

            if fin_prompt_seq[..., -1] == tokenizer.eos_token_id:
                end_flag = 1
                break

        n += 1

    total_generated_tokens = fin_prompt_seq.shape[-1] - initial_prompt_seq.shape[-1]
    acceptance_rate = round(draft_accepted_tokens / total_generated_tokens, 2)
    generated_ids = fin_prompt_seq[0, initial_prompt_seq.shape[-1]:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output, acceptance_rate

def test_speculative_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
    draft_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    prompt = "Once upon a time"
    target_len = 10
    output, acceptance_rate = speculative_sampling(target_model, draft_model, tokenizer, device, prompt, target_len, lookahead=4, temperature=1.0, debug=True)
    print("Output:", output)
    print(f"Acceptance rate: {acceptance_rate}")

if __name__ == "__main__":
    test_speculative_sampling()