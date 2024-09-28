import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.unconstrained import unconstrained_sampling
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("--model", type=str, required=True, help="Path/name of the model.")
    parser.add_argument("--prompt", type=str, required=True, help="Input sequence for the model.")
    parser.add_argument("--apply-chat-template", type=str, action=argparse.BooleanOptionalAction, default=False, help="Whether to apply the chat template to the prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Use temperature=0 for greedy decoding.")
    parser.add_argument("--method", type=str, choices=["unconstrained", "other_method"], default="unconstrained", help="Sampling method to use.")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Data type for the model.")

    args = parser.parse_args()

    if args.hf_token is None:
        args.hf_token = os.environ["HF_TOKEN"]

    if args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "float32":
        args.dtype = torch.float32

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype, trust_remote_code=True, token=args.hf_token).to(device)

    if args.apply_chat_template:
        args.prompt = apply_chat_template(args.prompt, tokenizer)

    fancy_print("Prompt:", args.prompt)

    # Generate output based on the selected method
    if args.method == "unconstrained":
        output_sequence, complete_sequence, logits = unconstrained_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        
        fancy_print("Output:", output_sequence)
    else:
        # Placeholder for other sampling methods
        raise NotImplementedError("The specified sampling method is not implemented.")

if __name__ == "__main__":
    main()
