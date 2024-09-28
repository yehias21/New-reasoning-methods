import torch

def sampling_with_temperature(logits, temperature=1.0):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return torch.multinomial(probs, num_samples=1)[0]

def apply_chat_template(prompt, tokenizer):
    chat = [
        {"role": "user", "content": prompt},
    ]

    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

def fancy_print(*args, add_new_line=False):
    if add_new_line:
        print()
    for arg in args:
        print("="*50)
        print(arg)
    print("="*50)