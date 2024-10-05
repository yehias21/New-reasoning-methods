import argparse
import os
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.unconstrained import generate_with_unconstrained_sampling
from src.top_k import generate_with_top_k_sampling
from src.top_p import generate_with_top_p_sampling
from src.min_p import generate_with_min_p_sampling
from src.typical import generate_with_typical_sampling
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("--method", type=str, choices=["unconstrained", "top_k", "top_p", "min_p", "typical", "speculative"], default="unconstrained", help="Sampling method to use.")
    parser.add_argument("--model", type=str, required=True, help="Path/name of the model.")
    parser.add_argument("--draft-model", type=str, default=None, help="Path/name of the draft model (required for speculative decoding).")
    parser.add_argument("--prompt", type=str, required=True, help="Input sequence for the model.")
    parser.add_argument("--apply-chat-template", type=str, action=argparse.BooleanOptionalAction, default=False, help="Whether to apply the chat template to the prompt.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling parameter.")
    parser.add_argument("--typical_p_mass", type=float, default=None, help="Typical-p mass parameter.")
    parser.add_argument("--min_tokens_to_keep", type=int, default=1, help="Minimum number of tokens to keep when sampling.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use temperature=0 for greedy decoding.")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Data type for the model.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    # Load hf token from environment variable if not provided
    if args.hf_token is None:
        args.hf_token = os.environ["HF_TOKEN"]

    # Make dtype torch compatible
    if args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "float32":
        args.dtype = torch.float32

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.hf_token, device)

    if args.apply_chat_template:
        args.prompt = apply_chat_template(args.prompt, tokenizer)

    fancy_print("Prompt:", args.prompt)

    # Generate output based on the selected method
    if args.method == "unconstrained":
        with torch.no_grad():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_unconstrained_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                fancy_print("Output:", output_sequence)
    
    elif args.method == "top_k":
        if args.top_k is None:
            parser.error("The --top_k argument is required when using the top-k sampling method.")
        with torch.no_grad():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_top_k_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, top_k=args.top_k, temperature=args.temperature)
                fancy_print("Output:", output_sequence)

    elif args.method == "top_p":
        if args.top_p is None:
            parser.error("The --top_p argument is required when using the top-p sampling method.")
        with torch.no_grad():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_top_p_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature)
                fancy_print("Output:", output_sequence)

    elif args.method == "min_p":
        if args.min_p is None:
            parser.error("The --min_p argument is required when using the min-p sampling method.")
        with torch.no_grad():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_min_p_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, min_p=args.min_p, temperature=args.temperature)
                fancy_print("Output:", output_sequence)

    elif args.method == "typical":
        if args.typical_p_mass is None:
            parser.error("The --typical_p_mass argument is required when using the typical sampling method.")
        with torch.no_grad():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_typical_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, typical_p_mass=args.typical_p_mass, temperature=args.temperature)
                fancy_print("Output:", output_sequence)

    elif args.method == "speculative":
        if args.draft_model is None:    
            parser.error("The --draft-model argument is required when using the speculative decoding method.")
        else:
            print("Assuming draft model and target model share the same tokenizer...")
            draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=args.dtype, trust_remote_code=True, token=args.hf_token).to(device)
            output_sequence = speculative_sampling(model, draft_model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            fancy_print("Output:", output_sequence)
    
    else:
        raise NotImplementedError("The specified sampling method is not implemented.")

if __name__ == "__main__":
    main()
