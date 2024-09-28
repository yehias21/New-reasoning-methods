import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.unconstrained import unconstrained_sampling, unconstrained_sampling_with_logit_processor
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("--method", type=str, choices=["unconstrained", "speculative"], default="unconstrained", help="Sampling method to use.")
    parser.add_argument("--model", type=str, required=True, help="Path/name of the model.")
    parser.add_argument("--draft-model", type=str, default=None, help="Path/name of the draft model (required for speculative decoding).")
    parser.add_argument("--prompt", type=str, required=True, help="Input sequence for the model.")
    parser.add_argument("--apply-chat-template", type=str, action=argparse.BooleanOptionalAction, default=False, help="Whether to apply the chat template to the prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use temperature=0 for greedy decoding.")
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

    print("using dtype:", args.dtype)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.hf_token, device)

    if args.apply_chat_template:
        args.prompt = apply_chat_template(args.prompt, tokenizer)

    fancy_print("Prompt:", args.prompt)

    # Generate output based on the selected method
    if args.method == "unconstrained":
        output_sequence = unconstrained_sampling(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        fancy_print("Output:", output_sequence)

        output_sequence_from_logit_processor = unconstrained_sampling_with_logit_processor(model, tokenizer, device, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        fancy_print("Output from logit processor:", output_sequence_from_logit_processor)
    
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
