import os
import numpy as np
from .model import Bert
from .nomic_model import NomicBert
from .registry import registry
from transformers import AutoTokenizer
from typing import Literal, Optional
import awkward as ak
import mlx.core as mx
import tqdm
from scipy.sparse import csr_matrix
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEQ_LENS = np.arange(16, 128, 16).tolist() + np.arange(128, 512, 32).tolist() + [512]

def pool(
    pooling_strategy: Literal["mean", "cls", "first", "max"],
    normalize: bool,
    last_hidden_state: mx.array,  # B, L, D
    pooler_output: Optional[mx.array] = None,  # B, D
    mask: Optional[mx.array] = None,  # B, L
) -> mx.array:
    """
    Pool output fron a sentence transformer model into one embedding.
    Use MLX tensors as input, return MLX tensor as output.
    : last_hidden_state: B, L, D
    : pooler_output: B, D
    : mask: B, L
    """
    # hiddens: B, L, D; mask: B, L
    if mask is None:
        mask = mx.ones(last_hidden_state.shape[:2])
    if pooling_strategy == "mean":
        pooled = mx.sum(
            last_hidden_state * mx.expand_dims(mask, -1), axis=1
        ) / mx.sum(mask, axis=-1, keepdims=True)
    elif pooling_strategy == "max":
        pooled = mx.max(
            last_hidden_state * mx.expand_dims(mask, -1), axis=1
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
        pooled = pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)

    return pooled

class EmbeddingModel:
    """
    SentenceTransformers-compatible model for encoding sentences
    with MLX.
    """
    def __init__(
        self,
        model_path: str,
        pooling_strategy: Literal["mean", "cls", "first", "max"],
        normalize: bool,
        max_length: int,
        nomic_bert: bool = False,
        lm_head: bool = False,
    ):
        if nomic_bert:
            self.model = NomicBert.from_pretrained(model_path, lm_head=lm_head)
        else:
            self.model = Bert.from_pretrained(model_path, lm_head=lm_head)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.max_length = max_length

    @classmethod
    def from_registry(cls, model_name: str):
        """
        Initialize from the model registry.
        """
        model_config = registry[model_name]
        return cls(
            model_path=model_config["repo"],
            pooling_strategy=model_config["pooling_strategy"],
            normalize=model_config["normalize"],
            max_length=model_config["max_length"],
            lm_head=model_config.get("lm_head", False),
            nomic_bert="nomic" in model_name,
        )
    
    def _tokenize(
        self, 
        sentences,
        min_length: Optional[int] = None # if provided, we add [MASK] tokens to short queries
    ) -> ak.Array:
        """
        Tokenize a list of sentences as a jagged array.
        """
        if min_length is not None:
            tokenized = self.tokenizer.batch_encode_plus(
                sentences,
                padding=False,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_length - 2,
            )
            
            # convert each key to a jagged array
            batch = {
                k: ak.Array(tokenized[k]) for k in tokenized
            }

            # pad to min length with MASK tokens
            if min_length is not None:
                batch["input_ids"] = self._pad_array(batch["input_ids"], self.tokenizer.mask_token_id, min_length - 2)
                batch["attention_mask"] = self._pad_array(batch["attention_mask"], 1, min_length - 2)
                if "token_type_ids" in batch:
                    batch["token_type_ids"] = self._pad_array(batch["token_type_ids"], 0, min_length - 2)

            # add special tokens
            batch["input_ids"] = ak.concatenate(
                [
                    np.ones((len(batch["input_ids"]), 1), dtype=np.int64) * self.tokenizer.cls_token_id,
                    batch["input_ids"],
                    np.ones((len(batch["input_ids"]), 1), dtype=np.int64) * self.tokenizer.sep_token_id,
                ],
                axis=1,
            )
            batch["attention_mask"] = ak.concatenate(
                [
                    np.ones((len(batch["attention_mask"]), 1), dtype=np.int64),
                    batch["attention_mask"],
                    np.ones((len(batch["attention_mask"]), 1), dtype=np.int64),
                ],
                axis=1,
            )
            if "token_type_ids" in batch:
                batch["token_type_ids"] = ak.concatenate(
                    [
                        np.zeros((len(batch["token_type_ids"]), 1), dtype=np.int64),
                        batch["token_type_ids"],
                        np.zeros((len(batch["token_type_ids"]), 1), dtype=np.int64),
                    ],
                    axis=1,
                )
        else:
            tokenized = self.tokenizer.batch_encode_plus(
                sentences,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
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
        sorted_lengths = np.array(lengths[sorted_indices])
        # round sorted lengths to nearest SEQ_LEN
        sorted_lengths = np.array([
            [x for x in SEQ_LENS if x >= l][0] for l in sorted_lengths
        ])
        return {
            k: tokens[k][sorted_indices, :]
            for k in tokens
        }, reverse_indices, sorted_lengths
    
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
        return arr.to_numpy()
    
    def _construct_batch(
        self, 
        batch: dict[str, ak.Array]
    ) -> dict[str, mx.array]:
        """
        Pad a batch of tokenized sentences and convert to MLX tensors.
        """
        tensor_batch = {}
        pad_id = self.tokenizer.pad_token_id
        longest = int(max(ak.num(batch["input_ids"], axis=1)))
        longest = SEQ_LENS[np.argmax(np.array(SEQ_LENS) > longest)]
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k not in batch:
                continue
            tensor_batch[k] = mx.array(
                self._pad_array(batch[k], pad_id, longest)
            )
            # print(k, "is type", tensor_batch[k].dtype)
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
        from collections import Counter
        tokens = self._tokenize(sentences)
        sorted_tokens, reverse_indices, lengths = self._sort_inputs(tokens)
        # print("lengths:", Counter(lengths))
        output_embeddings = []
        pbar = tqdm.tqdm(total=len(sentences), disable=not show_progress)
        for seq_len in sorted(SEQ_LENS, reverse=True): # biggest first
            pbar.set_postfix({"seq_len": seq_len})
            # create chunk of all sentences with length == seq_len
            chunk = {
                k: sorted_tokens[k][lengths == seq_len]
                for k in sorted_tokens
            }

            # iterate over batches within chunk
            for i in range(0, len(chunk["input_ids"]), batch_size):
                batch = {
                    k: chunk[k][i:i + batch_size]
                    for k in chunk
                }
                batch = self._construct_batch(batch)
                last_hidden_state, pooler_output = self.model(**batch)
                embs = pool(
                    self.pooling_strategy,
                    self.normalize,
                    last_hidden_state,
                    pooler_output
                )
                mx.eval(embs)
                output_embeddings.append(embs)
                pbar.update(len(batch["input_ids"]))
                del batch
            # we're done with this seqlen, clear the cache
            mx.metal.clear_cache()
        
        # concatenate embeddings and reverse the sort
        output_embeddings = mx.concatenate(output_embeddings, axis=0)
        output_embeddings = np.array(output_embeddings, copy=False)
        return output_embeddings[reverse_indices]

class SpladeModel(EmbeddingModel):
    def __init__(
        self,
        model_path: str,
        top_k: int,
        min_query_length: int = 16 # experimental: add MASK to short queries to allocate more computation
    ):
        super().__init__(
            model_path,
            pooling_strategy="max",
            normalize=False,
            max_length=512,
            nomic_bert=False,
            lm_head=True
        )
        self.top_k = top_k
        self.min_query_length = min_query_length

    @classmethod
    def from_registry(cls, model_name: str, top_k: int = 64, min_query_length: int = None):
        """
        Initialize from the model registry.
        """
        model_config = registry[model_name]
        return cls(
            model_path=model_config["repo"],
            top_k=top_k,
            min_query_length=min_query_length
        )

    @staticmethod
    def _create_sparse_embedding(
        activations: mx.array,
        top_k: int,
    ):
        B, V = activations.shape
        topk_indices = mx.argpartition(activations, -top_k, axis=-1)[:, :-top_k]
        activations[mx.arange(B).reshape(-1, 1), topk_indices] = 0
        return activations
    
    def encode(
        self, 
        sentences, 
        batch_size=16,
        show_progress=True, 
        **kwargs
    ):
        tokens = self._tokenize(sentences, min_length=self.min_query_length)
        sorted_tokens, reverse_indices = self._sort_inputs(tokens)
        output_embeddings = []
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
            mlm_output, _ = self.model(**batch)
            # try pooling with mlx instead
            embs = mx.max(mlm_output * mx.expand_dims(batch["attention_mask"], -1), axis=1)
            del batch
            # embs = np.log(1 + np.maximum(embs, 0))
            embs = mx.log(1 + mx.maximum(embs, 0))
            if self.top_k > 0:
                embs = self._create_sparse_embedding(embs, self.top_k)
            mx.eval(embs)
            output_embeddings.append(embs)
        sparse_embs = mx.concatenate(output_embeddings, axis=0)
        return np.array(sparse_embs, copy=False).astype(np.float16)[reverse_indices]