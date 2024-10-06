1. Add Static Cache for KV Cache in each sampling method - https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format
2. Use torch.compile to compile the model for faster inference - https://huggingface.co/docs/transformers/en/llm_optims
3. Implement Speculative Decoding
4. Implement other entropy based sampling methods
5. Explore Mixture of Depths