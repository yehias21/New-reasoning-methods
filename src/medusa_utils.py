# Taken from https://github.com/FasterDecoding/Medusa
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.conversation_format import Conversation
from src.eta import eta_sampling_with_temperature
from src.top_p import top_p_sampling_with_temperature

TOPK = 10 # topk for sparse tree (10 is a placeholder and it is sufficient)

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

class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)
    
def initialize_past_key_values(model: PreTrainedModel):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=model.device,
        dtype=model.dtype,
    )
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data

def reset_medusa_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None

def initialize_medusa(input_ids, model, medusa_attn_mask, past_key_values):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    logits, base_model_logits = model(input_ids, past_key_values=past_key_values)
    model.base_model.model.medusa_mask = medusa_attn_mask
    return logits, base_model_logits

def generate_candidates(medusa_logits, logits, tree_indices, retrieve_indices, temperature = 0, epsilon = 0.09, top_p=0.8, sampling = 'eta'):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - medusa_logits (torch.Tensor): Logits from a specialized Medusa structure, aiding in candidate selection.
    - logits (torch.Tensor): Standard logits from a language model.
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - temperature (float, optional): Controls the diversity of the sampling process. Defaults to 0.
    - epsilon (float, optional): Scaling factor for the entropy-based threshold in eta sampling. Defaults to 0.09.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    if temperature == 0:
        candidates_logit = torch.argmax(logits[-1]).unsqueeze(0)
    else:
        if sampling == 'eta':
            candidates_logit = eta_sampling_with_temperature(logits=logits[-1], temperature=temperature, epsilon=epsilon).unsqueeze(0)
        elif sampling == 'nucleus':
            candidates_logit = top_p_sampling_with_temperature(logits=logits[-1], temperature=temperature, top_p=top_p).unsqueeze(0)
        else:
            raise NotImplementedError
    # Extract the TOPK candidates from the medusa logits.
    candidates_medusa_logits = torch.topk(medusa_logits[:, -1], TOPK, dim = -1).indices

    # Combine the selected candidate from the original logits with the topk medusa logits.
    # print(candidates_logit.shape)
    # print(candidates_logit)
    # print(candidates_medusa_logits.shape)
    candidates = torch.cat([candidates_logit.view(-1), candidates_medusa_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, tree_candidates

def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = medusa_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    tree_medusa_logits, tree_logits = model(
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    tree_logits = tree_logits.unsqueeze(0)
    # print(tree_medusa_logits.shape)
    # print(tree_logits.shape)
    # print(retrieve_indices)
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    medusa_logits = tree_medusa_logits[:, retrieve_indices]
    # print(medusa_logits.shape)
    return medusa_logits, logits

def evaluate_posterior(logits, candidates, temperature, epsilon = 0.09, top_p = 0.8, sampling = 'eta'):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - epsilon (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
        
    if sampling == 'eta':
        alpha = epsilon**0.5
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_eta_posterior_mask(logits, candidates, temperature, alpha, epsilon)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_eta_posterior_mask(logits, candidates, temperature, alpha, epsilon):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        alpha   (float): The minimum threshold for probabilities to be considered in sampling.
        epsilon (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * alpha , 
            torch.exp(-entropy) * epsilon,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask

def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_length : accept_length + 1
    ]
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, medusa_logits, new_token