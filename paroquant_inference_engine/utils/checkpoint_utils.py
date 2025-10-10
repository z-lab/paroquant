from transformers import AutoModelForCausalLM
import torch
from paroquant_inference_engine.model_executor.modules.rotation_linear import RotateLinearInt4, RotateLinearMarlinInt4
import torch.nn as nn
from typing import Iterator, Tuple, Union, Optional
from collections.abc import Generator
from .convert_utils import transform_from_pt
from safetensors.torch import load_file, safe_open, save_file
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
import huggingface_hub.constants
import os
import tempfile
import glob
import filelock
import hashlib
import fnmatch
import time
from tqdm import tqdm
from pathlib import Path

def iter_linears_with_parent(module: nn.Module, prefix: str = "") -> Iterator[Tuple[str, nn.Module, nn.Linear]]:
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, module, name, child
        yield from iter_linears_with_parent(child, full)

def strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name

@torch.no_grad()
def replace_linears_from_pt(model: nn.Module, pt_dir: str, prefix="model.layers.", ignore_suffix=("lm_head",), marlin=False):
    items = list(iter_linears_with_parent(model))
    tqdm_list = tqdm(items, desc=f"Replacing Linear → RotateLinearInt4", total=len(items))
    for full, parent, key, lin in tqdm_list:
        short = strip_prefix(full, prefix)
        if short.split(".")[-1] in ignore_suffix:
            continue

        blob_path = f"{pt_dir}/{short}.pt"
        try:
            blob = torch.load(blob_path, map_location="cuda", weights_only=True)
        except FileNotFoundError:
            print(f'{blob_path} not found, exit')
            exit()

        w, rotation_pairs, rotation_angles, channel_scales, qscales, qzeros_float, qzeros = transform_from_pt(blob, include_qsz=True)
        lin.weight.copy_(w)
        if not marlin:
            new_rotate_linear = RotateLinearInt4.from_linear(lin, rotation_angles, rotation_pairs, channel_scales, qscales=qscales, qzeros=qzeros, rotate_weight=True, init_only=False)
        else:
            new_rotate_linear = RotateLinearMarlinInt4.from_linear(lin, rotation_angles, rotation_pairs, channel_scales, qscales=qscales, qzeros=qzeros, rotate_weight=True, init_only=False)
        new_rotate_linear.to('cpu')
        setattr(parent, key, new_rotate_linear)
        


def from_pt_to_ckpt(model_name: str, pt_path: str, ckpt_out_path: str, prefix="model.layers.", ignore_suffix=("lm_head",), krot=8, group_size=128):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu', torch_dtype=torch.float16)
    replace_linears_from_pt(model, pt_path, prefix, ignore_suffix)
    model.half()
    model.config.rotq_config = {"quant_method": "rotquant", "nbit":4, "krot": krot, "group_size": group_size}
    model.config.orig_model_name = model_name
    model.save_pretrained(ckpt_out_path)

# adapted from vLLM
class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

# adapted from vLLM
def get_lock(model_name_or_path: Union[str, Path],
             cache_dir: Optional[str] = None):
    temp_dir = tempfile.gettempdir()
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock

# adapted from vLLM
def safetensors_weights_dictionary(
    hf_weights_files: list[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    d = {}
    for st_file in hf_weights_files:
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys(): 
                d[name] = st_file
    return d

# adapted from vLLM
def prepare_weights(
    model_name_or_path: str,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    ignore_patterns: Optional[str] = None,
) -> tuple[str, list[str], bool]:
    """Prepare weights for the model.

    If the model is not local, it will be downloaded."""
    is_local = os.path.isdir(model_name_or_path)
    allow_patterns = ["*.safetensors"]

    if not is_local:
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            download_dir,
            allow_patterns,
            revision,
            ignore_patterns=ignore_patterns,
        )
    else:
        hf_folder = model_name_or_path

    hf_weights_files: list[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    weights_dict = safetensors_weights_dictionary(
        hf_weights_files
    )

    return weights_dict

# adapted from vLLM
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: list[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, list[str]]] = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Before we download we look at what is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    print("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=local_only,
        )
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            print("Time spent downloading weights for %s: %.6f seconds",
                        model_name_or_path, time_taken)
    return hf_folder

def load_weights_into_module(module, prefix, weight_dict):
    """load weights from weight_dict to modules"""
    from safetensors import safe_open
    
    tensors_by_file = {}
    for key in module.state_dict().keys():
        full_name = f"{prefix}.{key}" if prefix else key
        if full_name in weight_dict:
            filepath = weight_dict[full_name]
            if filepath not in tensors_by_file:
                tensors_by_file[filepath] = []
            tensors_by_file[filepath].append(full_name)

    module_state_dict = {}
    for filepath, tensor_names in tensors_by_file.items():
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for full_name in tensor_names:
                key_in_module = full_name[len(prefix) + 1:] if prefix else full_name
                module_state_dict[key_in_module] = f.get_tensor(full_name)
    model_keys = set(module.state_dict().keys())
    loaded_keys = set(module_state_dict.keys())
    missing_keys = list(model_keys - loaded_keys)
    unexpected_keys = list(loaded_keys - model_keys)
    module.to_empty(device="cuda")
    module.load_state_dict(module_state_dict, strict=False)
    module.to('cuda')
    return missing_keys, unexpected_keys

def find_module_prefix(model, module_instance):
    for name, mod in model.named_modules():
        if mod is module_instance:
            return name
    return ""