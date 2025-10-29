from inference_engine.utils.checkpoint_utils import (
    prepare_weights,
    load_weights_into_module,
)
from transformers import AutoTokenizer, AutoConfig
import torch
from typing import Optional, Tuple
from inference_engine.model_executor import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)


def get_stop_token_ids(model_type, model_path=""):
    if model_type.lower() == "llama":
        if (
            "llama-3" in model_path.lower() or "llama3" in model_path.lower()
        ) and "30b" not in model_path.lower():
            # llama3
            return [128001, 128009]
        return []
    elif model_type.lower() == "qwen":
        return [151645]
    elif model_type.lower() == "falcon":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif model_type.lower() == "mpt":
        if "mpt" and "chat" in model_path:
            return [50278, 0]
        else:
            return []
    elif model_type.lower() == "nvila":
        return [151645]
    else:
        raise ValueError(f"model type {model_type} is not supported")


def build_and_load_model(
    model_path: str,
    orig_model_path: str,
    device: str = "cuda:0",
    ppl_eval: bool = False,
) -> Tuple[LlamaForCausalLM, AutoTokenizer]:

    config = AutoConfig.from_pretrained(model_path)
    if ppl_eval:
        setattr(config, "test_ppl_forward", True)
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    model_cls = get_model_cls_from_name(model_path)
    model = model_cls(config)
    weight_dict = prepare_weights(model_name_or_path=model_path)

    mk, uek = load_weights_into_module(
        model.model.embed_tokens, prefix="model.embed_tokens", weight_dict=weight_dict
    )
    assert len(mk) == 0, f"Missing keys in embed_tokens: {mk}"
    mk, uek = load_weights_into_module(
        model.model.norm, prefix="model.norm", weight_dict=weight_dict
    )
    assert len(mk) == 0, f"Missing keys in model.norm: {mk}"
    mk, uek = load_weights_into_module(
        model.lm_head, prefix="lm_head", weight_dict=weight_dict
    )
    assert config.tie_word_embeddings == (len(mk) > 0), (
        f"tie_word_embeddings={config.tie_word_embeddings}, "
        f"but missing_keys_in_lm_head={len(mk)}>0 mismatch"
    )
    for i, block in enumerate(model.model.layers):
        prefix = f"model.layers.{i}"
        mk, uek = load_weights_into_module(
            block, prefix=prefix, weight_dict=weight_dict
        )
        assert len(mk) == 0, f"Missing keys in {prefix}: {mk}"

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    model = model.to(device).to(torch.half).eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def get_model_cls_from_name(model_name):
    if "llama" in model_name.lower():
        return LlamaForCausalLM
    elif "qwen3" in model_name.lower():
        return Qwen3ForCausalLM
    elif "qwen2" in model_name.lower() or "qwen" in model_name.lower():
        return Qwen2ForCausalLM
    else:
        raise ValueError(f"model type {model_name} is not supported")
