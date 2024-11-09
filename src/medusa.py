import types
from typing import Iterable, List, Optional, Tuple, Union
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
            hidden_states = outputs.last_hidden_state
            logits = [self.base_model.lm_head(hidden_states)]
            logits.extend(self.medusa_heads(hidden_states))
            logits = torch.stack(logits, dim=0)
            return logits
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    medusa_lm_model = MedusaLMHead(base_model=base_model, medusa_heads=medusa_heads, medusa_config=config).to(device)
    inp = "Hello, how are you?"
    inp_ids = tokenizer(inp, return_tensors="pt").input_ids.to(device)
    out = medusa_lm_model(input_ids=inp_ids)
    print(out)
    print(out.shape)