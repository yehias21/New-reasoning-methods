import argparse
import os
import ast
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src.unconstrained import unconstrained_sampling_with_temperature
from src.top_k import top_k_sampling_with_temperature
from src.top_p import top_p_sampling_with_temperature
from src.min_p import min_p_sampling_with_temperature
from src.typical import typical_sampling_with_temperature
from src.epsilon import epsilon_sampling_with_temperature
from src.eta import eta_sampling_with_temperature
from src.beam_search import generate_with_beam_search
from src.cot_decoding import generate_with_cot_decoding
from src.constrained_json_decoding import constrained_json_sampling
from src.speculative import speculative_sampling
from src.medusa import generate_with_medusa
from src.utils import *
from src.generation_utils import *
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("--method", type=str, choices=["unconstrained", "top_k", "top_p", "min_p", "typical", "eta", "epsilon", "beam_search", "cot_decoding", "constrained_json", "speculative", "medusa"], default="unconstrained", help="Sampling method to use.")
    parser.add_argument("--model", type=str, required=True, help="Path/name of the model.")
    parser.add_argument("--draft-model", type=str, default=None, help="Path/name of the draft model (required for speculative decoding).")
    parser.add_argument("--medusa-model-heads", type=str, default=None, help="Path/name of the medusa model heads (required for medusa decoding).")
    parser.add_argument("--prompt", type=str, default=None, help="Input sequence for the model.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to the file containing the prompt.")
    parser.add_argument("--apply-chat-template", type=str, action=argparse.BooleanOptionalAction, default=False, help="Whether to apply the chat template to the prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Use temperature=0 for greedy decoding.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling parameter.")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon sampling parameter (for eta and epsilon sampling).")
    parser.add_argument("--beam_width", type=int, default=None, help="Beam width for beam search.")
    parser.add_argument("--typical_p_mass", type=float, default=None, help="Typical-p mass parameter.")
    parser.add_argument("--json_schema", type=str, help="Path to the JSON schema file for constrained JSON sampling.")
    parser.add_argument("--max_array_length", type=int, default=10, help="Maximum length of arrays in constrained JSON sampling.")
    parser.add_argument("--max_number_tokens", type=int, default=6, help="Maximum number of tokens for numbers in constrained JSON sampling.")
    parser.add_argument("--max_string_token_length", type=int, default=10, help="Maximum number of tokens for strings in constrained JSON sampling.")
    parser.add_argument("--lookahead", type=int, default=4, help="Lookahead for speculative decoding.")
    parser.add_argument("--medusa_choices", type=str, default=None, help="Path to the file containing the medusa choices.")
    parser.add_argument("--min_tokens_to_keep", type=int, default=1, help="Minimum number of tokens to keep when sampling.")
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

    # Check if prompt is provided
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt_file must be provided.")
    
    # Read prompt from file if not provided via --prompt
    if args.prompt is None and args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    if args.method != "medusa":
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.hf_token, device)

        # For Llama pre-trained only models
        if "Llama" in args.model and args.prompt and not args.apply_chat_template:
            args.prompt = "<|begin_of_text|>" + args.prompt

        if args.apply_chat_template:
            args.prompt = apply_chat_template(args.prompt, tokenizer)

    if args.medusa_choices is not None and args.method == "medusa":
        args.medusa_choices = ast.literal_eval(args.medusa_choices)

    fancy_print("Prompt:", args.prompt)

    # Generate output based on the selected method
    if args.method == "unconstrained":
        sampling_function = unconstrained_sampling_with_temperature
        sampling_params = {"temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)
    
    elif args.method == "top_k":
        if args.top_k is None:
            parser.error("The --top_k argument is required when using the top-k sampling method.")
        sampling_function = top_k_sampling_with_temperature
        sampling_params = {"top_k": args.top_k, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "top_p":
        if args.top_p is None:
            parser.error("The --top_p argument is required when using the top-p sampling method.")
        sampling_function = top_p_sampling_with_temperature
        sampling_params = {"top_p": args.top_p, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "min_p":
        if args.min_p is None:
            parser.error("The --min_p argument is required when using the min-p sampling method.")
        sampling_function = min_p_sampling_with_temperature
        sampling_params = {"min_p": args.min_p, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "typical":
        if args.typical_p_mass is None:
            parser.error("The --typical_p_mass argument is required when using the typical sampling method.")
        sampling_function = typical_sampling_with_temperature
        sampling_params = {"typical_p_mass": args.typical_p_mass, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "epsilon":
        if args.epsilon is None:
            parser.error("The --epsilon argument is required when using the epsilon sampling method.")
        sampling_function = epsilon_sampling_with_temperature
        sampling_params = {"epsilon": args.epsilon, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "eta":
        if args.epsilon is None:
            parser.error("The --epsilon argument is required when using the eta sampling method.")
        sampling_function = eta_sampling_with_temperature
        sampling_params = {"epsilon": args.epsilon, "temperature": args.temperature}
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, sampling_function=sampling_function, sampling_params=sampling_params)
                fancy_print("Output:", output_sequence)

    elif args.method == "beam_search":
        if args.beam_width is None:
            parser.error("The --beam_width argument is required when using the beam search method.")
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = generate_with_beam_search(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, beam_width=args.beam_width, temperature=args.temperature)
                fancy_print("Output:", output_sequence)

    elif args.method == "cot_decoding":
        if args.num_return_sequences > 1:
            parser.error("The CoT decoding method only supports single-sequence generation with varying top index for the first token.")
        with torch.inference_mode():
            for intial_token_k in range(1, 11):
                output_sequence = generate_with_cot_decoding(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, intial_token_k=intial_token_k)
                if intial_token_k == 1:
                    fancy_print(f"Greedy path\nOutput with intial_token_k={intial_token_k}:", output_sequence)
                else:
                    fancy_print(f"Output with intial_token_k={intial_token_k}:", output_sequence)
    
    elif args.method == "constrained_json":
        if args.json_schema is None:
            parser.error("The --json_schema argument is required when using the constrained JSON sampling method.")
        if args.temperature == 0:
            parser.error("The temperature should be greater than 0 for constrained JSON sampling.")
        with open(args.json_schema, 'r') as f:
            json_schema = json.load(f)
        with torch.inference_mode():
            for _ in range(args.num_return_sequences):
                output_sequence = constrained_json_sampling(
                    model,
                    tokenizer,
                    args.prompt,
                    json_schema,
                    max_array_length=args.max_array_length,
                    max_number_tokens=args.max_number_tokens,
                    temperature=args.temperature,
                    max_string_token_length=args.max_string_token_length
                )
                fancy_print("Output:", json.dumps(output_sequence, indent=2))

    elif args.method == "speculative":
        if args.draft_model is None:    
            parser.error("The --draft-model argument is required when using the speculative decoding method.")
        else:
            with torch.inference_mode():
                for _ in range(args.num_return_sequences):
                    print(f"Using model: {args.model}")
                    print(f"Using draft model: {args.draft_model}")
                    print(f"Using lookahead: {args.lookahead}")
                    print("Assuming draft model and target model share the same tokenizer...")
                    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=args.dtype, trust_remote_code=True, token=args.hf_token).to(device)
                    output_sequence, acceptance_rate = speculative_sampling(model, draft_model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, lookahead=args.lookahead, temperature=args.temperature)
                    fancy_print("Output:", output_sequence)
                    fancy_print("Acceptance rate:", acceptance_rate)

    elif args.method == "medusa":
        if args.medusa_model_heads is None:    
            parser.error("The --medusa-model-heads argument is required when using the medusa decoding method.")
        else:
            with torch.inference_mode():
                for _ in range(args.num_return_sequences):
                    print(f"Using model: {args.model}")
                    print(f"Using draft model: {args.medusa_model_heads}")
                    if args.medusa_choices is not None:
                        print(f"Using medusa choices: {args.medusa_choices}")

                    if args.epsilon is not None:
                        sampling = "eta"
                        epsilon = args.epsilon
                        top_p = None
                    elif args.top_p is not None:
                        sampling = "nucleus"
                        epsilon = None
                        top_p = args.top_p
                    else:
                        sampling = "eta"
                        epsilon = 0.09
                        top_p = None
                        
                    output_sequence = generate_with_medusa(args.model, args.medusa_model_heads, device, args.prompt, max_new_tokens=args.max_new_tokens, medusa_choices=args.medusa_choices, dtype=args.dtype, temperature=args.temperature, sampling=sampling, epsilon=epsilon, top_p=top_p)
                    fancy_print("Output:", output_sequence)
    
    else:
        raise NotImplementedError("The specified sampling method is not implemented.")

if __name__ == "__main__":
    main()
