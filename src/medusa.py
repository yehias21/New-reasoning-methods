import os
import sys
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.conversation_format import get_conv_template
from src.medusa_utils import (evaluate_posterior, format_input,
                          generate_candidates, generate_medusa_buffers,
                          initialize_medusa, initialize_past_key_values,
                          reset_medusa_mode, tree_decoding,
                          update_inference_inputs)
from src.modeling_llama_kv import LlamaForCausalLM


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x

class MedusaConfig(PretrainedConfig):
    model_type = "medusa"

    def __init__(self,
                 hidden_size: int = 4096,
                 vocab_size: int = 32001,
                 num_heads: int = 5,
                 num_hidden_layers: int = 1,
                 max_paths: int = 64,
                 topk: int = 10,
                 truncated_vocab_size: Optional[int] = None,
                 **kwargs):

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_paths = max_paths
        self.topk = topk
        self.max_seq_len = int(2**20)
        self.truncated_vocab_size = vocab_size if truncated_vocab_size is None\
            else truncated_vocab_size
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["MedusaModel"]

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "MedusaConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)
        for k in list(config_dict.keys()):
            if 'num' in k:
                if 'heads' in k:
                    config_dict["num_heads"] = config_dict.pop(k)
                elif 'layers' in k:
                    config_dict["num_hidden_layers"] = config_dict.pop(k)
        return cls.from_dict(config_dict, **kwargs)

class MedusaModel(nn.Module):
    """This class implements the Medusa draft model from the paper: https://arxiv.org/abs/2401.10774
    Reference implementation: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/medusa.py"""

    def __init__(self, config: MedusaConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size=config.hidden_size,
                          num_layers=config.num_hidden_layers)
            for _ in range(config.num_heads)
        ])
        self.orig_vocab_size = config.vocab_size

        self.lm_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_heads)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        hiddent_states_per_head = [block(hidden_states) for block in self.blocks]
        logits_per_head = [lm_head(hs) for lm_head, hs in zip(self.lm_heads, hiddent_states_per_head)]
        return logits_per_head

class MedusaLMHead(nn.Module):
    def __init__(self, base_model: PreTrainedModel, tokenizer: AutoTokenizer, medusa_heads: MedusaModel, medusa_config: MedusaConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.medusa_heads = medusa_heads
        self.medusa_config = medusa_config
        self.tokenizer = tokenizer

        self.medusa_heads.to(self.base_model.dtype).to(self.base_model.device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs.last_hidden_state[0]
            base_model_logits = self.base_model.lm_head(hidden_states)
            medusa_logits = self.medusa_heads(hidden_states)
            logits = torch.stack([base_model_logits] + medusa_logits, dim=0)
            return logits, base_model_logits

    def generate(
            self, 
            input_ids: torch.LongTensor, 
            attention_mask: Optional[torch.Tensor] = None,
            temperature: float = 1.0,
            max_new_tokens: int = 128,
            medusa_choices: List[Tuple[int, ...]] = None,
            epsilon: float = 0.09,
            top_p: float = 0.8,
            sampling: str = 'eta'
    ) -> str:
        print("Temperature", temperature)
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(medusa_choices, device=self.base_model.device)
        
        self.medusa_choices = medusa_choices
        self.medusa_buffers = medusa_buffers

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # print("Before sending", past_key_values)
        medusa_logits, logits = initialize_medusa(input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values)

        new_token = 0
        last_round_token = 0

        # print("Medusa logits")
        # print(medusa_logits.shape)
        # print("Logits")
        # print(logits.shape)

        for idx in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                epsilon=epsilon,
                top_p=top_p,
                sampling=sampling,
            )
            
            # print(tree_candidates)

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # print(logits.shape)
            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, epsilon**0.5, epsilon, top_p=top_p, sampling=sampling)

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
        
        output = self.tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        return output

def generate_with_medusa(base_model_name, medusa_model_heads_name, device, prompt, max_new_tokens, medusa_choices, dtype=torch.bfloat16, temperature=1.0):
    base_model = LlamaForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = MedusaConfig.from_pretrained(medusa_model_heads_name)
    print(config)
    medusa_heads = MedusaModel(config=config)
    model_weights = hf_hub_download(repo_id=medusa_model_heads_name, filename="model.safetensors")
    state_dict = load_file(model_weights, device=device)
    medusa_heads.load_state_dict(state_dict)
    medusa_lm_model = MedusaLMHead(base_model=base_model, tokenizer=tokenizer, medusa_heads=medusa_heads, medusa_config=config).to(device)

    conv = get_conv_template("vicuna_v1.1")
    inp_prompt = format_input(conv, prompt)
    inp_ids = tokenizer([inp_prompt], return_tensors="pt").input_ids.to(device)

    vicuna_7b_medusa_choices = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]

    if medusa_choices is None:
        medusa_choices = vicuna_7b_medusa_choices

    output = medusa_lm_model.generate(inp_ids, max_new_tokens=max_new_tokens, medusa_choices=medusa_choices, temperature=temperature)
    return output

if __name__ == "__main__":
    # Get Vicuna conversation template
    conv = get_conv_template("vicuna_v1.1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configs
    # Assuming 3 heads (4 tokens generated at max)
    vicuna_7b_medusa_choices = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]


    medusa_buffers = generate_medusa_buffers(vicuna_7b_medusa_choices, device=device)
    print(medusa_buffers)
    # sys.exit()

    # Load models
    base_model_name = "lmsys/vicuna-7b-v1.3"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
    base_model_medusa = LlamaForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_name = "abhigoyal/vllm-medusa-vicuna-7b-v1.3"
    config = MedusaConfig.from_pretrained(model_name)
    print(config)
    medusa_heads = MedusaModel(config=config)
    model_weights = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    state_dict = load_file(model_weights, device=device)
    medusa_heads.load_state_dict(state_dict)
    print(medusa_heads)

    # Prompt
    # inp = "What is the capital of 1) France and 2) India?"
    # inp = "What is the capital of France?"
    inp = "Tell me a joke about the man who walks into a bar"

    inp_prompt = format_input(conv, inp)
    inp_ids = tokenizer([inp_prompt], return_tensors="pt").input_ids.to(device)

    medusa_lm_model = MedusaLMHead(base_model=base_model_medusa, tokenizer=tokenizer, medusa_heads=medusa_heads, medusa_config=config).to(device)
    
    with torch.inference_mode():        
        out_logits, base_model_logits = medusa_lm_model(input_ids=inp_ids)
        print(inp_prompt)
        print(out_logits)
        print(out_logits.shape)

        toks_argmax = torch.argmax(out_logits[..., -1, :], dim=-1)
        print(toks_argmax)
        print(tokenizer.batch_decode(toks_argmax, skip_special_tokens=True))

        # Normal decoding
        out = base_model.generate(inp_ids, do_sample=False, max_new_tokens=100)
        print(tokenizer.batch_decode(out, skip_special_tokens=True))


    # # medusa_lm_model.generate(inp_ids, max_steps=10, medusa_choices=vicuna_7b_medusa_choices)
    # last_output = ""
    # for output in medusa_lm_model.generate(inp_ids, max_steps=512, medusa_choices=vicuna_7b_medusa_choices):
    #     current_output = output["text"]
    #     # Only print the new part
    #     new_text = current_output[len(last_output):]
    #     print(new_text, end="", flush=True)
    #     last_output = current_output
    # print()
    output = medusa_lm_model.generate(inp_ids, max_new_tokens=512, medusa_choices=vicuna_7b_medusa_choices)
    print(output)