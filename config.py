GPT_CONFIG = {
    # 124M
    "small": {
        "vocab_size": 50257,  # vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    },
    # 355M
    "medium": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 1024,
        "n_heads": 12,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    # 774M
    "large": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 1280,
        "n_heads": 12,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    # 1558M
    "xlarge": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 1600,
        "n_heads": 12,
        "n_layers": 48,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
}
