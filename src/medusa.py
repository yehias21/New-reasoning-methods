from typing import Iterable, List, Optional, Tuple, Union
import os
import sys
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from conversation_format import get_conv_template, Conversation

TOPK = 10 # topk for sparse tree (10 is a placeholder and it is sufficient)

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
    def __init__(self, base_model: PreTrainedModel, medusa_heads: MedusaModel, medusa_config: MedusaConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.medusa_heads = medusa_heads
        self.medusa_config = medusa_config

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
    
def format_input(conv: Conversation, prompt: str) -> str:
    # Add user message to conversation
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    
    # Get the full prompt with conversation history
    full_prompt = conv.get_prompt()
    return full_prompt

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

if __name__ == "__main__":
    # Get Vicuna conversation template
    conv = get_conv_template("vicuna_v1.1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configs
    # Assuming 3 heads (4 tokens generated at max)
    vicuna_7b_medusa_choices = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]


    medusa_buffers = generate_medusa_buffers(vicuna_7b_medusa_choices, device=device)
    print(medusa_buffers)
    sys.exit()

    # Load models
    base_model_name = "lmsys/vicuna-7b-v1.3"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
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
    inp = "What is the capital of France?"
    inp_prompt = format_input(conv, inp)
    inp_ids = tokenizer([inp_prompt], return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        medusa_lm_model = MedusaLMHead(base_model=base_model, medusa_heads=medusa_heads, medusa_config=config).to(device)
        
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
