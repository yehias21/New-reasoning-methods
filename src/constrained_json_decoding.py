# Adapted from https://github.com/1rgs/jsonformer

import json
from typing import Any, Dict

import torch
from transformers import (LogitsWarper, PreTrainedModel, PreTrainedTokenizer, StoppingCriteria)


class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, _) -> bool:
        if len(input_ids[0]) <= self.prompt_length:
            return False
        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)
        return '"' in last_token

class NumberStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int, precision: int = 3):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        decoded = self.tokenizer.decode(input_ids[0][self.prompt_length:], skip_special_tokens=True)
        if decoded.count(".") > 1:
            return True
        if decoded.count(".") == 1 and len(decoded.strip().split(".")[1]) > self.precision:
            return True
        if len(decoded) > 1 and any(c.isdigit() for c in decoded) and decoded[-1] in [" ", "\n"]:
            return True
        return False

class OutputNumbersTokens(LogitsWarper):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id).strip()
            if token_str == "" or (all(c.isdigit() or c == "." for c in token_str) and token_str.count(".") <= 1):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")
        return scores

class ConstrainedJsonDecoding:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: dict,
        prompt: str,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt
        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)
        self.generation_marker = "|GENERATION|"
        self.max_array_length = max_array_length
        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length
        self.value = {}

    def generate_number(self):
        prompt = self.get_prompt()
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]))],
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = response[..., input_tokens.shape[1]:]
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        response = response.strip().rstrip(".")
        return float(response)

    def generate_boolean(self):
        prompt = self.get_prompt()
        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.forward(input_tensor.to(self.model.device))
        logits = output.logits[0, -1]
        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")
        result = logits[true_token_id] > logits[false_token_id]
        return result.item()

    def generate_string(self):
        prompt = self.get_prompt() + '"'
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if (len(response[0]) >= len(input_tokens[0]) and (response[0][:len(input_tokens[0])] == input_tokens).all()):
            response = response[0][len(input_tokens[0]):]
        if response.shape[0] == 1:
            response = response[0]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response.split('"')[0].strip()

    def generate_object(self, properties: dict, obj: dict) -> dict:
        for key, schema in properties.items():
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(self, schema: dict, obj: dict, key: str = None):
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else: # if key is an empty list or None
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else: # if key is an empty list or None
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else: # if key is an empty list or None
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj) 
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: dict, obj: list) -> list:
        for _ in range(self.max_array_length):
            element = self.generate_value(item_schema, obj)
            obj[-1] = element
            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt")
            output = self.model.forward(input_tensor.to(self.model.device))
            logits = output.logits[0, -1]
            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]
            found_comma = False
            found_close_bracket = False
            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
                if ',' in decoded_token:
                    found_comma = True
                    break
                if ']' in decoded_token:
                    found_close_bracket = True
                    break
            if found_close_bracket or not found_comma:
                break
        return obj

    def get_prompt(self):
        template = "{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")
        return template.format(prompt=self.prompt, schema=json.dumps(self.json_schema), progress=progress)

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data

def constrained_json_sampling(model, tokenizer, prompt, json_schema, max_array_length=10, max_number_tokens=6, temperature=1.0, max_string_token_length=10):
    constrained_json_decoding = ConstrainedJsonDecoding(
        model,
        tokenizer,
        json_schema,
        prompt,
        max_array_length=max_array_length,
        max_number_tokens=max_number_tokens,
        temperature=temperature,
        max_string_token_length=max_string_token_length,
    )
    return constrained_json_decoding()

def test_constrained_json_sampling():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    prompt = "Generate a JSON object describing a person with name as Shreyansh, age 26, who is works at Level AI and likes to play cricket and chess."
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    result = constrained_json_sampling(
        model,
        tokenizer,
        prompt,
        json_schema,
        max_array_length=3,
        max_number_tokens=2,
        temperature=1,
        max_string_token_length=5
    )

    print("Generated JSON:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_constrained_json_sampling()
