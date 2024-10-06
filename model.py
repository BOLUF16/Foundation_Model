import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNormalization(nn.Module):
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 10**-6)


class MultiHeadAttention(nn.Module):
   
    def __init__(self, config):
        super().__init__()
        assert config.embedding % config.head == 0, "Embedding size must be divisible by the number of heads"
        self.head = config.head
        self.embedding = config.embedding  # Fixed the typo

        self.d_k = config.embedding // config.head

        self.w_q = nn.Linear(config.embedding, config.embedding, bias=config.bias)
        self.w_k = nn.Linear(config.embedding, config.embedding, bias=config.bias)
        self.w_v = nn.Linear(config.embedding, config.embedding, bias=config.bias)
        self.w_o = nn.Linear(config.embedding, config.embedding, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask):
        # Linear projections for query, key, and value
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape into (batch_size, head, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1, 2)  # Corrected value transformation

        # Scaled dot-product attention
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask (if present)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # Weighted sum of values
        x = attention_scores @ value

        # Concatenate and pass through the output linear layer
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head * self.d_k)

        return self.w_o(x)

    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layer = nn.Linear(config.embedding, config.d_ff, bias = config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(config.d_ff, config.embedding, bias = config.bias)
        

    def forward(self, x):
       
        return self.output_layer(self.dropout(self.gelu(self.input_layer(x))))
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer1 = LayerNormalization(config.embedding, bias=config.bias)
        self.attention = MultiHeadAttention(config)
        self.layer2 = LayerNormalization(config.embedding, bias = config.bias)
        self.mlp = MLP(config)
    
    def forward(self,x, mask = None):
        layer_output_1 = self.layer1(x + self.attention(x,x,x,mask))
        layer_output_2 = self.layer2(layer_output_1  + self.mlp(layer_output_1))
        return layer_output_2
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocabulary_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            word_token_embedding = nn.Embedding(config.vocabulary_size, config.embedding),
            word_position_embedding = nn.Embedding(config.block_size, config.embedding),
            dropout = nn.Dropout(config.dropout),
            list_of_blocks = nn.ModuleList([Block(config) for _ in range(config.number_of_layer)]),
            layer_Normalization = LayerNormalization(config.embedding, config.bias)
        ))
        self.lm_head = nn.Linear(config.embedding, config.vocabulary_size, bias=False)
        self.transformer.word_token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("output_layer.weight"):
                torch.nn.init.normal_(p, mean=0.0, std = 0.02/math.sqrt(2  * config.number_of_layer))

        
        print("number of parameters: %.2fM" % (self.get_num_parameter()/1e6,))

    def get_num_parameter(self, non_embedding = True):

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.word_position_embedding.weight.numel()
        
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        device = idx.device
        block, sequence_length = idx.size()
        assert sequence_length <= self.config.block_size, f"Cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"          
        position = torch.arange(0, sequence_length, dtype=torch.long, device = device)
        token_embedding = self.transformer.word_token_embedding(idx)
        position_embedding = self.transformer.word_position_embedding(position)
        x = self.transformer.dropout(token_embedding + position_embedding)
        for block in self.transformer.list_of_blocks:
            x = block(x)
        x = self.transformer.layer_Normalization(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=100)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss   
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.word_position_embedding.weight = nn.Parameter(self.transformer.word_position_embedding.weight[:block_size])
        for block in self.transformer.list_of_blocks:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.adamw).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused = True) if use_fused else dict()
        optimizer = torch.optim.adamw(optim_groups, lr = learning_rate, betas = betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

    
        return optimizer
    
    def generate(self, idx, max_new_tokens, temperature = 0.5, top_k = None):

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)

            idx_next =  torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx