from transformers import (
    BertModel, DistilBertModel, BertForMaskedLM, DistilBertForMaskedLM,
    BertConfig, DistilBertConfig
)

def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    key = key.replace(
        "cls.predictions.transform.dense.",
        "lm_head.dense.")
    key = key.replace(
        "cls.predictions.transform.LayerNorm.",
        "lm_head.ln.")
    key = key.replace(
        "cls.predictions.decoder",
        "lm_head.decoder")
    return key

def replace_key_distilbert(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace("transformer.", "encoder.")
    key = key.replace("embeddings.LayerNorm", "embeddings.norm")
    key = key.replace(".attention.q_lin.", ".attention.query_proj.")
    key = key.replace(".attention.k_lin.", ".attention.key_proj.")
    key = key.replace(".attention.v_lin.", ".attention.value_proj.")
    key = key.replace(".attention.out_lin.", ".attention.out_proj.")
    key = key.replace(".sa_layer_norm.", ".ln1.")
    key = key.replace(".ffn.lin1.", ".linear1.")
    key = key.replace(".ffn.lin2.", ".linear2.")
    key = key.replace(".output_layer_norm.", ".ln2.")
    key = key.replace("vocab_transform", "lm_head.dense")
    key = key.replace("vocab_layer_norm", "lm_head.ln")
    key = key.replace("vocab_projector", "lm_head.decoder")
    key = key.replace("distilbert.", "")

    return key


def convert(bert_model: str, lm_head: bool = False):
    if not lm_head:
        model = BertModel.from_pretrained(bert_model)
    else:
        model = BertForMaskedLM.from_pretrained(bert_model)
    # save the tensors
    tensors = {
        replace_key(key): tensor.numpy() for key, tensor in model.state_dict().items()
    }
    # print([n for n, p in tensors.items()])
    return tensors

def convert_distilbert(distilbert_model: str, lm_head: bool = False):
    if not lm_head:
        model = DistilBertModel.from_pretrained(distilbert_model)
    else:
        model = DistilBertForMaskedLM.from_pretrained(distilbert_model)
    # save the tensors
    tensors = {
        replace_key_distilbert(key): tensor.numpy() for key, tensor in model.state_dict().items()
    }
    # print([n for n, p in tensors.items()])
    return tensors

def bert_config_from_distilbert(distilbert_config: DistilBertConfig) -> BertConfig:
    return BertConfig(
        vocab_size=distilbert_config.vocab_size,
        hidden_size=distilbert_config.dim,
        num_hidden_layers=distilbert_config.n_layers,
        num_attention_heads=distilbert_config.n_heads,
        intermediate_size=distilbert_config.hidden_dim,
        hidden_act=distilbert_config.activation,
        hidden_dropout_prob=distilbert_config.dropout,
        attention_probs_dropout_prob=distilbert_config.attention_dropout,
        max_position_embeddings=distilbert_config.max_position_embeddings,
        type_vocab_size=0,
        initializer_range=distilbert_config.initializer_range,
        layer_norm_eps=1e-12,
    )
