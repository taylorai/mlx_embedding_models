# modeling code adapted from mlx-examples (MIT License)
# https://github.com/ml-explore/mlx-examples

import tempfile
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig, BertConfig, RobertaConfig
from .load_utils import convert

class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        config: Union[BertConfig, RobertaConfig],
        # dims: int,
        # num_heads: int,
        # mlp_dims: Optional[int] = None,
        # layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        # mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            bias=True
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, config: Union[BertConfig, RobertaConfig],
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config)
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: Union[BertConfig, RobertaConfig]):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: Union[BertConfig, RobertaConfig]):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            config
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        return y, mx.tanh(self.pooler(y[:, 0]))
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        config = AutoConfig.from_pretrained(model_path)
        # check if it's a bert or roberta model
        if isinstance(config, BertConfig):
            config = BertConfig.from_pretrained(model_path)
        elif isinstance(config, RobertaConfig):
            config = RobertaConfig.from_pretrained(model_path)
        model = cls(config)
        tensors = convert(model_path)
        # use npz extension
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f, **tensors)
            f.seek(0)
            model.load_weights(f.name)
        return model