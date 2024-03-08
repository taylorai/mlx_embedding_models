import os
import numpy as np
from .model import Bert
from .registry import registry
from transformers import AutoTokenizer
from typing import Literal, Optional
import awkward as ak
import mlx.core as mx
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pool(
    pooling_strategy: Literal["mean", "cls", "first", "max"],
    normalize: bool,
    last_hidden_state: np.ndarray,  # B, L, D
    pooler_output: Optional[np.ndarray] = None,  # B, D
    mask: Optional[np.ndarray] = None,  # B, L
) -> np.ndarray:
    """
    Pool output fron a sentence transformer model into one embedding.
    Expects numpy arrays as input, so be sure to convert the MLX tensors.
    : last_hidden_state: B, L, D
    : pooler_output: B, D
    : mask: B, L
    """
    # hiddens: B, L, D; mask: B, L
    if mask is None:
        mask = np.ones(last_hidden_state.shape[:2])
    if pooling_strategy == "mean":
        pooled = np.sum(
            last_hidden_state * np.expand_dims(mask, -1), axis=1
        ) / np.sum(mask, axis=-1, keepdims=True)
    elif pooling_strategy == "max":
        pooled = np.max(
            last_hidden_state * np.expand_dims(mask, -1), axis=1
        )
    elif pooling_strategy == "first":
        pooled = last_hidden_state[:, 0, :]
    elif pooling_strategy == "cls":
        if pooler_output is None:
            # use first token w/ no pooling linear layer
            pooled = last_hidden_state[:, 0, :]
        else:
            pooled = pooler_output
    else:
        raise NotImplementedError(
            f"pooling strategy {pooling_strategy} not implemented"
        )
    if normalize:
        pooled = pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)

    return pooled

class EmbeddingModel:
    """
    SentenceTransformers-compatible model for encoding sentences
    with MLX.
    """
    def __init__(
        self,
        model_path: str,
        pooling_strategy: Literal["mean", "cls", "first"],
        normalize: bool,
        max_length: int,
        lm_head: bool = False,
    ):
        self.model = Bert.from_pretrained(model_path, lm_head=lm_head)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.max_length = max_length

    @classmethod
    def from_registry(cls, model_name: str, normalize: bool = True):
        """
        Initialize from the model registry.
        """
        model_config = registry[model_name]
        return cls(
            model_path=model_config["repo"],
            pooling_strategy=model_config["pooling_strategy"],
            normalize=normalize,
            max_length=model_config["max_length"],
            lm_head=model_config.get("lm_head", False),
        )
    
    def _tokenize(self, sentences) -> ak.Array:
        """
        Tokenize a list of sentences as a jagged array.
        """
        tokenized = self.tokenizer(
            sentences,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        
        # convert each key to a jagged array
        batch = {
            k: ak.Array(tokenized[k]) for k in tokenized
        }
        return batch
    
    def _sort_inputs(self, tokens: dict[str, ak.Array]) -> ak.Array:
        """
        Sort inputs by length for efficient batching.
        Returns sorted batch, plus indices to reverse the sort.
        """
        lengths = ak.num(tokens["input_ids"], axis=1)
        sorted_indices = np.argsort(-1 * lengths)
        reverse_indices = np.argsort(sorted_indices)
        return {
            k: tokens[k][sorted_indices, :]
            for k in tokens
        }, reverse_indices
    
    def _pad_array(
        self, 
        arr: ak.Array, 
        pad_id: int,
        length: int
    ) -> list[list[int]]:
        """
        Pad a jagged array to a target length.
        """
        arr = ak.pad_none(arr, target=length, axis=-1, clip=True)
        arr = ak.fill_none(arr, pad_id)
        return arr.to_list()
    
    def _construct_batch(self, batch: dict[str, ak.Array]) -> dict[str, mx.array]:
        """
        Pad a batch of tokenized sentences and convert to MLX tensors.
        """
        tensor_batch = {}
        pad_id = self.tokenizer.pad_token_id
        max_length = int(max(ak.num(batch["input_ids"], axis=1)))
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k not in batch:
                continue
            tensor_batch[k] = mx.array(
                self._pad_array(batch[k], pad_id, max_length)
            )
        return tensor_batch
    
    def encode(
        self, 
        sentences, 
        batch_size=64, 
        show_progress=True, 
        **kwargs
    ):
        """
        Encode a list of sentences into embeddings.
        """
        tokens = self._tokenize(sentences)
        sorted_tokens, reverse_indices = self._sort_inputs(tokens)
        output_embeddings = None
        for i in tqdm.tqdm(
            range(0, len(sentences), batch_size),
            disable=not show_progress,
        ):
            # slice out batch & convert to MLX tensors
            batch = {
                k: sorted_tokens[k][i:i + batch_size]
                for k in sorted_tokens
            }
            batch = self._construct_batch(batch)
            last_hidden_state, pooler_output = self.model(**batch)
            embs = pool(
                self.pooling_strategy,
                self.normalize,
                np.array(last_hidden_state),
                np.array(pooler_output),
            )
            if output_embeddings is None:
                output_embeddings = embs
            else:
                output_embeddings = np.concatenate(
                    [output_embeddings, embs], axis=0
                )

        return output_embeddings[reverse_indices]
    
# TODO: implement this
class SpladeModel:
    pass