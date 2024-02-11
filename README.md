# MLX Embedding Models
Run text embeddings on your Apple Silicon GPU. Supports any BERT- or RoBERTa-based embedding model, with a curated registry of high-performing models that just work off the shelf.

Get started by installing from PyPI:
```
pip install mlx-embedding-models
```

Then get started in a few lines of code:
```python
from mlx_embedding_models.embedding import EmbeddingModel
model = EmbeddingModel.from_registry("bge-small")
texts = [
    "isn't it nice to be inside such a fancy computer",
    "the horse raced past the barn fell"
]
embs = model.encode(texts)
print(embs.shape)
# 2, 384
```