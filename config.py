from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocabulary_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    number_of_layer: int = 12
    d_ff: int = 3072
    head: int = 12
    embedding: int = 768
    dropout: float = 0.0
    bias: bool = True
