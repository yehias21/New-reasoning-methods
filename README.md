# LLM Sampling Library

## Overview

The LLM Sampling Library is a Python package designed to facilitate text generation using various sampling methods with large language models (LLMs). This library provides a simple command-line interface (CLI) for users to generate text based on input prompts, utilizing models from the Hugging Face Transformers library.

## Features

- **Multiple Sampling Methods**: Implement various sampling techniques, including unconstrained sampling.
- **Easy Integration**: Load models from Hugging Face with minimal setup.
- **Customizable Parameters**: Adjust parameters such as temperature and maximum token generation to fine-tune the output.
- **Chat Template Support**: Optionally apply chat templates to prompts for enhanced interaction.

In all my implementations, I have added temperature scaling to the logits before applying any sampling methods. This is in accordance to the [GPT-2 implementation](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/sample.py#L61C5-L72C14) and how [HuggingFace's implementation of sampling](https://github.com/huggingface/transformers/blob/acc394c4f5e1283c19783581790b3dc3105a3697/src/transformers/generation/utils.py#L825C9-L847C23) also does it.

## Installation

To install the required dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Usage

To generate text using the library, you can use the following command:

```bash
python generate.py --model <model_name> --prompt "<input_prompt>" --apply-chat-template --temperature <temperature> --method <sampling_method> --max_new_tokens <max_new_tokens> --hf-token <hugging_face_token> --dtype <data_type>
```

### Parameters

- `--model`: The path or name of the Hugging Face model to use.
- `--prompt`: The input sequence for the model.
- `--temperature`: Sampling temperature (default is 1.0).
- `--method`: The sampling method to use (e.g., `unconstrained`).
- `--max_new_tokens`: Maximum number of new tokens to generate (default is 500).
- `--hf-token`: Your Hugging Face token for model access.
- `--dtype`: Data type for the model (options: `bfloat16`, `float16`, `float32`).

## Example

To generate text using the unconstrained sampling method with a temperature of 0.5, you can use the following command:

```bash
python generate.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "Tell me a joke about the man who walks into a bar" --apply-chat-template --temperature 0 --method unconstrained
```
