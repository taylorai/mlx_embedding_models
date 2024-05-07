registry = {
    # 3 layers, 384-dim
    "bge-micro": {
        "repo": "TaylorAI/bge-micro-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True, 
        "ndim": 384,
    },
    # 6 layers, 384-dim
    "gte-tiny": {
        "repo": "TaylorAI/gte-tiny",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "minilm-l6": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "snowflake-xs": {
        "repo": "Snowflake/snowflake-arctic-embed-xs",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 384-dim
    "minilm-l12": {
        "repo": "sentence-transformers/all-MiniLM-L12-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "bge-small": {
        "repo": "BAAI/bge-small-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first", # cls token, not pooler output
        "normalize": True,
        "ndim": 384,
    },
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 768-dim
    "bge-base": {
        "repo": "BAAI/bge-base-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1": {
        "repo": "nomic-ai/nomic-embed-text-v1",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1.5": {
        "repo": "nomic-ai/nomic-embed-text-v1.5",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
        "apply_ln": True,
    },
    # 24 layers, 1024-dim
    "bge-large": {
        "repo": "BAAI/bge-large-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024,
    },
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "max_length": 8192,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    # SPARSE MODELS #
    "distilbert-splade": {
        "repo": "raphaelsty/distilbert-splade",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "neuralcherche-sparse-embed": {
        "repo": "raphaelsty/neural-cherche-sparse-embed",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "opensearch": {
        "repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "bert-base-uncased": { # mainly here as a baseline
        "repo": "bert-base-uncased",
        "max_length": 512,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "naver-splade-distilbert": {
        "repo": "naver/splade-v3-distilbert",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    }
}