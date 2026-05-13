from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from .util import CosineAnnealingParam, RETAINED_KWARG_KEYS, logger, to_device


def _get_independent_channel_pairs(
    pairs: torch.Tensor, dim: int, num_rotations: int, num_pairs_each: int
) -> list[list[tuple[int, int]]]:
    pairs = pairs.cpu().tolist()
    rotations_pairs = []
    # Record the available pairs in a matrix.
    available = torch.ones(dim, dim)
    available.fill_diagonal_(0)

    for _ in range(num_rotations):
        independent_pairs = []
        # We use a copy to track the available pairs in this rotation.
        # It's different from the original available matrix.
        available_in_rotation = available.clone()
        # Greedily select pairs that are independent
        for i, j in pairs:
            if len(independent_pairs) == num_pairs_each:
                break
            if available_in_rotation[i, j] == 0:
                continue
            # Simply select the first available pair.
            independent_pairs.append((i, j))
            # Selecting (i, j) in this rotation prohibits
            # selecting any other pairs that share i or j
            # in this rotation.
            available_in_rotation[i, :] = 0
            available_in_rotation[j, :] = 0
            available_in_rotation[:, i] = 0
            available_in_rotation[:, j] = 0
            # Mark the pair as unavailable for future selections.
            # i and j are still available for any other pairs
            # in next rotations.
            available[i, j] = 0
            available[j, i] = 0

        rotations_pairs.append(independent_pairs)

    return rotations_pairs


def get_random_rotation_pairs(
    sensitivity_input: torch.Tensor,
    group_size: int,
    num_rotations: int,
    num_pairs_factor: float,
    seed: int,
) -> list[tuple[int, int]]:
    sorted_pairs: list[list[tuple[int, int]]] = []
    rand = random.Random(seed)
    group_num = sensitivity_input.shape[0]

    for group_idx in range(group_num):
        sorted_pairs.append([])
        for i in range(group_size):
            for j in range(i + 1, group_size):
                sorted_pairs[group_idx].append((i, j))

        rand.shuffle(sorted_pairs[group_idx])

    sorted_pairs = torch.tensor(sorted_pairs, device=sensitivity_input.device)

    pairs_k_groups = [[] for _ in range(num_rotations)]
    num_pairs_per_group = int(group_size * num_pairs_factor)

    for i in range(group_num):
        offset = i * group_size
        pairs_g_k_groups = _get_independent_channel_pairs(
            sorted_pairs[i], group_size, num_rotations, num_pairs_per_group
        )
        for r_idx in range(num_rotations):
            pairs_g = pairs_g_k_groups[r_idx]
            for j, (col1, col2) in enumerate(pairs_g):
                pairs_g[j] = (col1 + offset, col2 + offset)
            pairs_k_groups[r_idx].extend(pairs_g)

    return pairs_k_groups


def optimize_module(
    module: nn.Module,
    train_set_batches: tuple[Iterator[tuple[torch.Tensor, ...]], Iterator[torch.Tensor]],
    val_set_batches: tuple[Iterator[tuple[torch.Tensor, ...]], Iterator[torch.Tensor]],
    train_kwargs: dict | list[dict],
    val_kwargs: dict | list[dict],
    optim_params: list[dict],
    *,
    loss_fn: Literal["mse", "smooth_l1"],
    n_iter: int,
    gradient_accumulation_steps: int = 1,
    early_stop: int | None,
    post_optim_callback: Callable[[nn.Module], None] | None = None,
    metric_logger: Callable[[dict[str, float], int], None] | None = None,
    start_step: int = 0,
) -> int:
    train_args_batches, train_output_batches = train_set_batches
    val_args_batches, val_output_batches = val_set_batches

    if gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be a positive integer, got {gradient_accumulation_steps}")

    num_train_batches = len(train_args_batches)
    total_steps = n_iter * math.ceil(num_train_batches / gradient_accumulation_steps)
    schedulers = [
        CosineAnnealingParam(
            start_value=param_group["lr"],
            end_value=param_group["lr"] / 20,
            T_max=total_steps,
        )
        for param_group in optim_params
    ]
    optimizer = torch.optim.AdamW(optim_params)
    scaler = torch.amp.GradScaler()
    if loss_fn == "mse":
        loss = nn.MSELoss()
    elif loss_fn == "smooth_l1":
        loss = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    progress_bar = tqdm(total=n_iter, unit="iter")

    def module_output(args: tuple[torch.Tensor, ...], kwargs: dict | list[dict], batch_idx: int) -> torch.Tensor:
        batch_kwargs = kwargs
        if isinstance(kwargs, list):
            batch_kwargs = {
                key: value.copy() if isinstance(value, dict) else value for key, value in kwargs[batch_idx].items()
            }
            for key in RETAINED_KWARG_KEYS:
                if key in batch_kwargs:
                    batch_kwargs[key] = to_device(batch_kwargs[key], args[0].device)
        out = module(*args, **batch_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def val_loss_batches(
        args_batches: Iterator[tuple[torch.Tensor, ...]],
        output_batches: Iterator[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = None
        for batch_idx, (args_batch, output_batch) in enumerate(zip(args_batches, output_batches)):
            output_q = module_output(args_batch, val_kwargs, batch_idx)
            loss_value = loss(output_batch, output_q)
            if total_loss is None:
                total_loss = loss_value
            else:
                total_loss += loss_value
        return total_loss / len(args_batches)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        original_val_loss = val_loss_batches(val_args_batches, val_output_batches)

    best_val_loss = original_val_loss
    best_sd = deepcopy(module.state_dict())
    current_step = start_step
    if metric_logger is not None:
        metric_logger(
            {
                "val_loss": original_val_loss.item(),
                "best_val_loss": best_val_loss.item(),
            },
            current_step,
        )

    early_stop_counter = 0

    for _ in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        accum_loss = 0.0
        window_size = min(gradient_accumulation_steps, num_train_batches)
        for batch_idx, (args_batch, output_batch) in enumerate(
            zip(train_args_batches, train_output_batches),
            start=1,
        ):
            if accum_counter == 0:
                remaining_batches = num_train_batches - batch_idx + 1
                window_size = min(gradient_accumulation_steps, remaining_batches)

            with torch.amp.autocast("cuda"):
                output_q = module_output(args_batch, train_kwargs, batch_idx - 1)
                batch_loss = loss(output_batch, output_q)
                loss_value = batch_loss / window_size

            scaler.scale(loss_value).backward()
            accum_loss += batch_loss.item()
            accum_counter += 1

            if accum_counter == window_size:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

                for i, scheduler in enumerate(schedulers):
                    optimizer.param_groups[i]["lr"] = scheduler.step()

                if post_optim_callback:
                    post_optim_callback(module)

                current_step += 1
                if metric_logger is not None:
                    metric_logger({"loss": accum_loss / window_size}, current_step)
                accum_loss = 0.0

        with torch.no_grad(), torch.amp.autocast("cuda"):
            val_loss_value = val_loss_batches(val_args_batches, val_output_batches)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_sd = deepcopy(module.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop is not None and early_stop_counter >= early_stop:
                break

        if metric_logger is not None:
            metric_logger(
                {
                    "val_loss": val_loss_value.item(),
                    "best_val_loss": best_val_loss.item(),
                },
                current_step,
            )

        progress_bar.set_postfix(
            val_loss=val_loss_value.item(),
            val_og_loss=original_val_loss.item(),
        )
        progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Best val loss: {best_val_loss.item()}, Original val loss: {original_val_loss.item()}")

    module.load_state_dict(best_sd)
    return current_step
