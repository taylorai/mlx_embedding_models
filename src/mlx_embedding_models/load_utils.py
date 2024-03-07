from transformers import BertModel, DistilBertModel, BertForMaskedLM, DistilBertForMaskedLM

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