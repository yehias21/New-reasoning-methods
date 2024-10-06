import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_with_beam_search(model, tokenizer, device, prompt, max_new_tokens, beam_width=5, temperature=1.0):
    initial_prompt_seq = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Initialize beams with the initial prompt and their corresponding scores, past_key_values, and is_complete
    beams = [{
        "tokens": initial_prompt_seq,
        "score": 0.0,
        "past_key_values": None,
        "is_complete": False
    }]

    count_new_tokens = 0
    while count_new_tokens < max_new_tokens:
        all_candidates = []
        for beam in beams:
            if beam["is_complete"]:
                all_candidates.append(beam)
                continue

            if beam["past_key_values"] is None:
                # First step: Process the entire prompt
                model_input = {"input_ids": beam["tokens"]}
            else:
                # Subsequent steps: Pass only the last generated token with past_key_values
                last_token_id = beam["tokens"][:, -1].unsqueeze(-1)  # Shape (1, 1)
                model_input = {
                    "input_ids": last_token_id,
                    "past_key_values": beam["past_key_values"]
                }

            with torch.no_grad():
                outputs = model(**model_input)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            logits = logits / (temperature + 1e-10)

            log_probs = torch.log_softmax(logits, dim=-1)

            # Get the top beam_width tokens and their log probabilities
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

            for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                new_tokens = torch.cat([beam["tokens"], idx.view(1, 1)], dim=-1)
                new_score = beam["score"] + log_prob.item()

                if idx.item() == tokenizer.eos_token_id:
                    is_complete = True
                else:
                    is_complete = False

                all_candidates.append({
                    "tokens": new_tokens,
                    "score": new_score,
                    "past_key_values": past_key_values,
                    "is_complete": is_complete
                })

        # Select the top beam_width beams based on their scores
        ordered = sorted(all_candidates, key=lambda x: x["score"], reverse=True)
        beams = ordered[:beam_width]

        # Check if all beams have generated the eos_token_id
        if all(beam["tokens"][0, -1].item() == tokenizer.eos_token_id for beam in beams):
            break

        count_new_tokens += 1

    # Select the beam with the highest score
    best_beam = beams[0]
    generated_ids = best_beam["tokens"]
    final_prompt_seq = generated_ids[..., initial_prompt_seq.shape[-1]:]

    # Decode the generated tokens
    output = tokenizer.decode(final_prompt_seq[0], skip_special_tokens=True)
    return output

def test_beam_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    max_new_tokens = 50
    beam_width = 5

    generated_text = generate_with_beam_search(model, tokenizer, device, prompt, max_new_tokens, beam_width)
    print("Generated Text with Beam Search:")
    print(generated_text)

if __name__ == "__main__":
    test_beam_search_sampling()