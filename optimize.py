import torch
import torch.nn as nn
import simple_parsing
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
import json
from typing import Literal, Optional
import sys

sys.path.append(Path(__file__).parent.as_posix())

from paroquant.optimize import (
    optimize_module,
    get_random_rotation_pairs,
)
from paroquant.module import (
    PseudoQuantizedLinear,
    reset_angles_by_mask,
)
from paroquant.util import (
    set_module_by_name,
    load_model,
    move_embed,
    load_tokenizer,
    get_blocks,
    get_calib_dataset,
    get_mixed_calib_dataset,
    catch_first_layer_input,
    get_named_linears,
    empty_cache,
    logger,
    CachedTensorShards,
)
from paroquant.convert_utils import transform_to_kernel_data


@dataclass(kw_only=True)
class Config:
    # Huggingface model path.
    model: str
    # The parameters to optimize at each stage and the corresponding learning rates,
    # e.g., --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6"
    params: list[str]
    # The number of epochs for each stage of optimization,
    # e.g., --epochs 10 10
    epochs: list[int]

    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    # Loss function to use.
    loss: Literal["mse", "smooth_l1"] = "smooth_l1"

    # Quantization & rotation group size.
    group_size: int
    # Bit width.
    n_bit: int
    # Number of rotations.
    num_rotations: int

    skipped_modules: list[str] = field(default_factory=list)

    # Calibration datasets. If more than one dataset is provided,
    # they will be sampled evenly and shuffled.
    datasets: list[str]
    val_dataset: str
    train_size: int
    validation_size: int
    batch_size: int
    val_batch_size: Optional[int] = None  # Defaults to batch_size if not set.
    seqlen: int

    # Number of shards to cache the input/output tensors. At any time, only one shard
    # will be moved to GPU for optimization. The rest will be kept in CPU memory.
    # Increasing this reduces GPU memory usage but increases training time.
    cache_shards: int = 1

    # Directory to save state dicts of optimized linear layers.
    output_dir: str

    # Whether to resume from previously saved results in `output_dir`.
    resume: bool = False
    # Whether to enable gradient checkpointing.
    checkpointing: bool = False

    seed: int


def main():
    args = simple_parsing.parse(
        Config, add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    print(args)

    # Store the results in a subdirectory.
    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Currently only support single GPU training.
    device = "cuda"

    # Determine which params to optimize.
    params_to_optimize: list[dict[str, float]] = []
    for params in args.params:
        params = params.strip().split(",")
        param_dict = {}
        for param in params:
            param, lr = param.strip().split(":")
            param_dict[param.strip()] = float(lr.strip())
        params_to_optimize.append(param_dict)
    print(f"Parameters to optimize: {params_to_optimize}")

    # Save args to output directory
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model.
    model = load_model(args.model, device_map="cpu", dtype=torch.float16)
    move_embed(model, device)
    tokenizer = load_tokenizer(args.model)
    blocks = get_blocks(model)

    # Get calibration dataset.
    samples = get_mixed_calib_dataset(
        args.datasets,
        tokenizer=tokenizer,
        n_samples=args.train_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="train",
    )
    samples = torch.stack(samples, dim=0).to(device)

    val_samples = get_calib_dataset(
        args.val_dataset,
        tokenizer=tokenizer,
        n_samples=args.validation_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="validation",
    )
    val_samples = torch.stack(val_samples, dim=0).to(device)

    # Capture first layer's input.
    logger.info("Capturing first layer input...")
    blocks[0].to(device)
    og_layer_input_batches, kwargs = catch_first_layer_input(
        model,
        blocks,
        samples,
        batch_size=args.batch_size,
    )
    val_batch_size = args.val_batch_size or args.batch_size
    og_layer_val_input_batches, _ = catch_first_layer_input(
        model,
        blocks,
        val_samples,
        batch_size=val_batch_size,
    )
    blocks[0].cpu()

    del samples
    empty_cache()

    @torch.no_grad()
    def forward_layer_batch(
        layer: nn.Module,
        input_batched: list[torch.Tensor],
        kwargs: dict,
        store_device: torch.device,
        dtype: torch.dtype = torch.float16,
        cast_to_dtype: torch.dtype = torch.float16,
    ) -> list[torch.Tensor]:
        output_batched = []

        layer.to(device)
        for input_batch in input_batched:
            output = layer(input_batch.to(dtype).to(device), **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dtype != cast_to_dtype:
                output = output.to(cast_to_dtype)
            if output.device != store_device:
                output = output.to(store_device)
            output_batched.append(output)
        layer.cpu()

        empty_cache()
        return output_batched

    def set_checkpointing_enabled(module: nn.Module, enable: bool) -> None:
        for linear in get_named_linears(
            module, subclass=PseudoQuantizedLinear
        ).values():
            linear.enable_checkpoint = enable

    # Layerwise, multi-stage optimization.
    for layer_idx, layer in enumerate(tqdm(blocks)):
        empty_cache()
        logger.info(f"Capturing original layer output...")
        # Original output of this layer.
        og_layer_output_batches = forward_layer_batch(
            layer, og_layer_input_batches, kwargs, store_device="cpu"
        )
        og_layer_val_output_batches = forward_layer_batch(
            layer, og_layer_val_input_batches, kwargs, store_device="cpu"
        )

        if layer_idx > 0:
            layer_input_batches = new_layer_output_batches
            layer_val_input_batches = new_layer_val_output_batches
        else:
            layer_input_batches = og_layer_input_batches
            layer_val_input_batches = og_layer_val_input_batches

        train_input_batches = layer_input_batches
        train_output_batches = og_layer_output_batches

        train_input_batches = CachedTensorShards(
            train_input_batches, args.cache_shards, target_device=device
        )
        train_output_batches = CachedTensorShards(
            train_output_batches, args.cache_shards, target_device=device
        )

        val_input_batches = [b.to(device) for b in layer_val_input_batches]
        val_output_batches = [b.to(device) for b in og_layer_val_output_batches]

        # Freeze all parameters
        for param in layer.parameters():
            param.requires_grad = False

        linear_modules = get_named_linears(layer)
        if args.resume:
            all_files_exist = True
            for name in linear_modules.keys():
                file_name = f"{layer_idx}.{name}.pt"
                file_path = output_dir / file_name
                if not file_path.exists() and name not in args.skipped_modules:
                    all_files_exist = False
                    break
        else:
            all_files_exist = False

        if not all_files_exist:
            logger.info(f"Initializing rotation parameters...")

        for name, old_module in linear_modules.items():
            if name in args.skipped_modules:
                continue

            if all_files_exist:
                existing_result_file = output_dir / f"{layer_idx}.{name}.pt"
                sd = torch.load(existing_result_file, map_location=device)
                new_module = PseudoQuantizedLinear.from_state_dict(sd)
                set_module_by_name(layer, name, new_module)
                continue

            old_module.to(device)

            num_pairs_factor = 0.5
            weight = old_module.weight.float()
            weight_grouped = weight.view(weight.shape[0], -1, args.group_size).permute(
                1, 0, 2
            )

            all_pairs = get_random_rotation_pairs(
                weight_grouped,
                group_size=args.group_size,
                num_rotations=args.num_rotations,
                num_pairs_factor=num_pairs_factor,
                seed=args.seed + layer_idx,
            )

            all_pairs = [
                torch.tensor(pairs, device="cpu", dtype=torch.int32)
                for pairs in all_pairs
            ]
            initial_angles = [
                torch.zeros(pairs.shape[0], device="cpu") for pairs in all_pairs
            ]
            initial_scales = torch.ones(
                1, weight.shape[1], dtype=torch.float16, device=device
            )

            npairs, angles, mask = transform_to_kernel_data(
                all_pairs,
                initial_angles,
                group_size=args.group_size,
            )
            npairs = npairs.to(device)
            angles = angles.to(device)
            mask = mask.to(device)
            rotation_pairs = [npairs, angles, mask]
            channel_scales = initial_scales

            new_module = PseudoQuantizedLinear(
                old_module,
                rotation_pairs,
                channel_scales,
                group_size=args.group_size,
                n_bits=args.n_bit,
                num_rotations=args.num_rotations,
            )

            set_module_by_name(layer, name, new_module)
            old_module.cpu()

        if not all_files_exist:
            layer.to(device).float()

            set_checkpointing_enabled(layer, args.checkpointing)
            for step, step_params_dict in enumerate(params_to_optimize):
                empty_cache()
                optim_params = []
                new_modules = get_named_linears(layer, subclass=PseudoQuantizedLinear)
                for new_module in new_modules.values():
                    new_module.set_optim_enabled(
                        **{param_name: True for param_name in step_params_dict.keys()},
                    )
                    for param_name, lr in step_params_dict.items():
                        optim_params.append(
                            dict(
                                params=new_module.get_optim_params(param_name),
                                lr=lr,
                                weight_decay=args.weight_decay,
                                betas=args.betas,
                                eps=args.eps,
                            )
                        )

                logger.info(
                    f"Optimizing layer {layer_idx}, step {step + 1}/{len(params_to_optimize)}: "
                    f"{', '.join([k for k in step_params_dict])}"
                )

                optimize_module(
                    layer,
                    (train_input_batches, train_output_batches),
                    (val_input_batches, val_output_batches),
                    kwargs,
                    optim_params,
                    loss_fn=args.loss,
                    n_iter=args.epochs[step],
                    early_stop=None,
                    post_optim_callback=reset_angles_by_mask,
                )

            set_checkpointing_enabled(layer, False)

            del (
                train_input_batches,
                train_output_batches,
                val_input_batches,
                val_output_batches,
            )
            empty_cache()

        else:
            logger.info(
                f"Skipping optimization for layer {layer_idx}: already been optimized."
            )

        layer.half().to(device)

        logger.info("Capturing new layer output...")
        new_layer_output_batches = forward_layer_batch(
            layer,
            layer_input_batches,
            kwargs,
            store_device="cpu",
        )
        new_layer_val_output_batches = forward_layer_batch(
            layer,
            layer_val_input_batches,
            kwargs,
            store_device="cpu",
        )

        og_layer_input_batches = og_layer_output_batches
        og_layer_val_input_batches = og_layer_val_output_batches

        if all_files_exist:
            layer.cpu()
            continue

        # Save the optimized result
        for name, module in get_named_linears(
            layer, subclass=PseudoQuantizedLinear
        ).items():
            result_file = output_dir / f"{layer_idx}.{name}.pt"
            torch.save(
                module.state_dict(),
                result_file,
            )

        layer.cpu()


if __name__ == "__main__":
    main()
