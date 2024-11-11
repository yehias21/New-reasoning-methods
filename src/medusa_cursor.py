import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict
import copy

class MedusaModel:
    def __init__(
        self,
        base_model_path: str,
        medusa_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load Medusa heads from checkpoint
        medusa_ckpt = torch.load(medusa_checkpoint_path, map_location=device)
        self.medusa_heads = self._load_medusa_heads(medusa_ckpt)
        self.num_heads = len(self.medusa_heads)
        print(f"Loaded {self.num_heads} Medusa heads")
        
    def _load_medusa_heads(self, checkpoint: Dict) -> torch.nn.ModuleList:
        """Load Medusa heads from checkpoint"""
        heads = torch.nn.ModuleList()
        for key, value in checkpoint.items():
            if key.startswith("medusa_head"):
                heads.append(
                    torch.nn.Linear(
                        self.base_model.config.hidden_size,
                        self.base_model.config.vocab_size
                    ).to(self.device)
                )
                heads[-1].load_state_dict(value)
        return heads

    def _get_medusa_logits(
        self, 
        hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get logits from all Medusa heads"""
        return [head(hidden_states) for head in self.medusa_heads]

    def tree_sample(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        tree_beam_size: int = 3,
        tree_depth: int = 3,
        verbose: bool = True
    ) -> str:
        """
        Perform tree sampling with Medusa heads
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_tokens = []
        current_length = input_ids.shape[1]
        
        while current_length < max_tokens:
            with torch.no_grad():
                # Get base model outputs
                outputs = self.base_model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get last hidden state
                last_hidden = outputs.hidden_states[-1][:, -1:]
                
                # Get logits from base model and Medusa heads
                base_logits = outputs.logits[:, -1:]
                medusa_logits = self._get_medusa_logits(last_hidden)
                
                # Build tree candidates
                candidates = self._build_tree_candidates(
                    base_logits,
                    medusa_logits,
                    temperature,
                    top_p,
                    tree_beam_size,
                    tree_depth
                )
                
                if verbose:
                    print("\nTree candidates:")
                    for i, cand in enumerate(candidates):
                        tokens = self.tokenizer.decode(cand["token_ids"])
                        print(f"Candidate {i}: {tokens} (score: {cand['score']:.3f})")
                
                # Select best candidate
                best_candidate = max(candidates, key=lambda x: x["score"])
                best_tokens = best_candidate["token_ids"]
                
                # Append to generated sequence
                input_ids = torch.cat([input_ids, best_tokens.unsqueeze(0)], dim=1)
                generated_tokens.extend(best_tokens.tolist())
                
                if verbose:
                    print(f"\nSelected tokens: {self.tokenizer.decode(best_tokens)}")
                
                current_length = input_ids.shape[1]
                
                # Check for EOS token
                if best_tokens[-1] == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated_tokens)

    def _build_tree_candidates(
        self,
        base_logits: torch.Tensor,
        medusa_logits: List[torch.Tensor],
        temperature: float,
        top_p: float,
        beam_size: int,
        depth: int
    ) -> List[Dict]:
        """Build tree candidates using base model and Medusa heads"""
        candidates = []
        
        # Process base logits
        base_probs = F.softmax(base_logits / temperature, dim=-1)
        base_probs = self._top_p_filtering(base_probs, top_p)
        
        # Sample from base distribution
        base_samples = torch.multinomial(base_probs[0], beam_size)
        
        for base_token in base_samples:
            # Initialize candidate
            candidate = {
                "token_ids": base_token.unsqueeze(0),
                "score": base_probs[0, base_token].log().item()
            }
            
            # Extend with Medusa heads
            current_score = candidate["score"]
            current_tokens = [base_token]
            
            for d in range(min(depth, len(medusa_logits))):
                head_logits = medusa_logits[d]
                head_probs = F.softmax(head_logits / temperature, dim=-1)
                head_probs = self._top_p_filtering(head_probs, top_p)
                
                # Sample from head
                head_token = torch.multinomial(head_probs[0], 1)[0]
                current_score += head_probs[0, head_token].log().item()
                current_tokens.append(head_token)
            
            candidate["token_ids"] = torch.tensor(current_tokens)
            candidate["score"] = current_score
            candidates.append(candidate)
        
        return candidates

    def _top_p_filtering(
        self,
        probs: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to probabilities"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize
        return probs / probs.sum(dim=-1, keepdim=True)

def test_medusa():
    """Test function to demonstrate Medusa model usage"""
    # Initialize model
    model = MedusaModel(
        base_model_path="lmsys/vicuna-7b-v1.3",  # Replace with actual model path
        medusa_checkpoint_path="medusa_lm_head.pt",  # Replace with actual checkpoint
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test prompt
    prompt = "Once upon a time,"
    
    print(f"Generating text for prompt: {prompt}")
    print("-" * 50)
    
    # Generate text
    output = model.tree_sample(
        prompt,
        max_tokens=50,
        temperature=0.8,
        top_p=0.9,
        tree_beam_size=3,
        tree_depth=3,
        verbose=True
    )
    
    print("\nFinal generated text:")
    print("-" * 50)
    print(prompt + output)

if __name__ == "__main__":
    test_medusa() 