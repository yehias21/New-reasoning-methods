import argparse

from vllm import LLM, SamplingParams

from conversation_format import get_conv_template

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Vicuna model with optional Medusa mode')
parser.add_argument('--mode', type=str, default='default', choices=['default', 'medusa'],
                   help='Mode to run the model in (default or medusa)')
args = parser.parse_args()

# Initialize the model
if args.mode == 'default':
    model = LLM(model="lmsys/vicuna-7b-v1.3")
else:
    model = LLM(model="lmsys/vicuna-7b-v1.3", speculative_model="abhigoyal/vllm-medusa-vicuna-7b-v1.3", num_speculative_tokens=5)

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=512,
    stop=["</s>", "USER:", "ASSISTANT:"]  # Common stop tokens for Vicuna
)

# Get Vicuna conversation template
conv = get_conv_template("vicuna_v1.1")  # v1.1 template works for v1.3 model

def generate_response(prompt: str) -> str:
    # Add user message to conversation
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    
    # Get the full prompt with conversation history
    full_prompt = conv.get_prompt()
    print(full_prompt)
    
    # Generate response
    outputs = model.generate([full_prompt], sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    
    # Add the response to conversation history
    conv.messages[-1][1] = generated_text
    
    return generated_text

if __name__ == "__main__":
    # Example usage
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    for prompt in prompts:
        print(f"\nHuman: {prompt}")
        response = generate_response(prompt)
        print(f"Assistant: {response}")
