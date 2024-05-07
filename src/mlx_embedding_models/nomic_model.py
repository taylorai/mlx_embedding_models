# TODO: This doesn't work (doesn't match with sentence transformers output AT ALL)

import tempfile
from typing import Tuple, Union, Optional
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig
from .load_utils import convert_nomic_bert

class NomicBertEmbeddings(nn.Module):
    def __init__(self, config):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings if config.rotary_emb_fraction <= 0 else 0
        self.type_vocab_size = config.type_vocab_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.n_embd
        ) if config.type_vocab_size > 0 else None
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.n_embd
        ) if self.max_position_embeddings > 0 else None
        

    def __call__(
        self, 
        input_ids: mx.array, 
        position_ids: Optional[mx.array] = None, 
        token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        words = self.word_embeddings(input_ids)
        if token_type_ids is not None and self.token_type_embeddings is not None:
            words += self.token_type_embeddings(token_type_ids)
        if position_ids is not None and self.position_embeddings is not None:
            words += self.position_embeddings(position_ids)

        return self.norm(words)

class NomicBertMLP(nn.Module):
    def __init__(
        self,
        config: AutoConfig
    ):
        super().__init__()
        multiple_of = 256
        in_features, out_features = config.n_embd, config.n_embd
        hidden_features = config.n_inner
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of

        self.fc11 = nn.Linear(in_features, hidden_features, bias=config.mlp_fc1_bias)
        self.fc12 = nn.Linear(in_features, hidden_features, bias=config.mlp_fc1_bias)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=config.mlp_fc2_bias)

    def __call__(self, x):
        y = self.fc11(x)
        gate = self.fc12(x)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y
    
class NomicBertAttention(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.norm_factor = math.sqrt(self.head_dim)
        self.rotary_emb_dim = int(self.head_dim * config.rotary_emb_fraction)
        
        if self.rotary_emb_dim > 0:
            self.rotary_emb = nn.RoPE(
                dims=self.rotary_emb_dim,
                traditional=config.rotary_emb_interleaved,
                base=config.rotary_emb_base, # increase to extrapolate to docs > 2048 tokens
                scale=1.0
            )
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.qkv_proj_bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.qkv_proj_bias)
    
    @staticmethod
    def shape(x: mx.array):
        return x.transpose(0, 2, 1, 3)
    
    @staticmethod
    def merge_heads(x: mx.array):
        pass
    
    def __call__(self, x, mask):
        qkv = self.Wqkv(x)
        B, S, _ = qkv.shape
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] # b, s, h, d

        q, k, v = map(self.shape, (q, k, v)) # b, h, s, d

        if self.rotary_emb_dim > 0:
            q, k = self.rotary_emb(q), self.rotary_emb(k)

        attention_scores = q @ k.transpose(0, 1, 3, 2) / self.norm_factor
        if mask is not None:
            attention_scores += mask

        attentions_probs: mx.array = mx.softmax(attention_scores, axis=-1) # B, H, S, S
        attn_output = attentions_probs @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, -1)

        return self.out_proj(attn_output)

class NomicBertLayer(nn.Module):
    """
    A transformer encoder layer with RoPE.
    """

    def __init__(
        self,
        config: AutoConfig,
    ):
        super().__init__()
        self.attn = NomicBertAttention(config)
        # self.attention = FastAttention(config)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = NomicBertMLP(config)

    def __call__(self, x, mask):
        attention_out = self.attn(x, mask)
        add_and_norm = self.norm1(x + attention_out)
        mlp_out = self.mlp(add_and_norm)
        return self.norm2(add_and_norm + mlp_out)


class NomicBertEncoder(nn.Module):
    def __init__(
        self, config: AutoConfig
    ):
        super().__init__()
        self.layers = [
            NomicBertLayer(config)
            for _ in range(config.n_layer)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x

class LMHead(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd, bias=config.mlp_fc1_bias)
        self.activation = nn.SiLU()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=config.mlp_fc1_bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense(x)
        x = self.activation(x)
        x = self.ln(x)
        x = self.decoder(x)
        return x

class NomicBert(nn.Module):
    def __init__(self, config: AutoConfig, lm_head: bool = False, pooler: bool = False):
        self.embeddings = NomicBertEmbeddings(config)
        self.encoder = NomicBertEncoder(config)
        
        if lm_head:
            self.lm_head = LMHead(config)
        else:
            self.lm_head = None
        if pooler:
            self.pooler = nn.Linear(config.n_embd, config.n_embd)
        else:
            self.pooler = None

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids=token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        # mlm output
        if self.lm_head is not None:
            y = self.encoder(x, attention_mask)
            return self.lm_head(y), None # no pooler output
        
        # pooler output
        elif self.pooler is not None:
            y = self.encoder(x, attention_mask)
            return y, mx.tanh(self.pooler(y[:, 0]))
        
        else:
            y = self.encoder(x, attention_mask)
            return y, None
    
    @classmethod
    def from_pretrained(cls, model_path: str, lm_head: bool = False, pooler: bool = False):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # check if it's a bert or roberta model
        tensors = convert_nomic_bert(model_path, lm_head=lm_head)
        model = cls(config, lm_head=lm_head)
        
        # use npz extension
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f, **tensors)
            f.seek(0)
            model.load_weights(f.name)
        return model