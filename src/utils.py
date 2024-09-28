import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, dtype, hf_token, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True, token=hf_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    if isinstance(model.generation_config.eos_token_id, list):
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
    else:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer

def sampling_with_temperature(logits, temperature=1.0):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return torch.multinomial(probs, num_samples=1)[0]

def apply_chat_template(prompt, tokenizer):
    chat = [
        {"role": "user", "content": prompt},
    ]

    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

def fancy_print(*args, add_new_line=True):
    if add_new_line:
        print()
    for arg in args:
        print("="*50)
        print(arg)
    print("="*50)