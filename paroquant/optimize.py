import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from copy import deepcopy
from typing import Literal, Optional, Iterator, Callable
import random

from .util import empty_cache, logger


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
    train_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    val_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    kwargs: dict,
    optim_params: list[dict],
    *,
    loss_fn: Literal["mse", "smooth_l1"],
    n_iter: int,
    early_stop: Optional[int],
    post_optim_callback: Optional[Callable[[nn.Module], None]] = None,
) -> None:
    train_input_batches, train_output_batches = train_set_batches
    val_input_batches, val_output_batches = val_set_batches

    lr_schedulers = []
    t_max = n_iter * len(train_input_batches)
    for param_group in optim_params:
        optimizer = torch.optim.AdamW([{k: v for k, v in param_group.items()}])
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=param_group["lr"] / 20,
        )
        lr_schedulers.append(scheduler)

    optimizer = torch.optim.AdamW(optim_params)
    scaler = torch.amp.GradScaler()
    if loss_fn == "mse":
        loss = nn.MSELoss()
    elif loss_fn == "smooth_l1":
        loss = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    progress_bar = tqdm(total=n_iter, unit="iter")

    def module_output(input: torch.Tensor) -> torch.Tensor:
        out = module(input, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def loss_batches(
        input_batches: Iterator[torch.Tensor], output_batches: Iterator[torch.Tensor]
    ) -> torch.Tensor:
        total_loss = None
        for input_batch, output_batch in zip(input_batches, output_batches):
            output_q = module_output(input_batch)
            loss_value = loss(output_batch, output_q)
            if total_loss is None:
                total_loss = loss_value
            else:
                total_loss += loss_value
        return total_loss / len(input_batches)

    with torch.no_grad():
        original_val_loss = loss_batches(val_input_batches, val_output_batches)

    best_val_loss = original_val_loss
    best_sd = deepcopy(module.state_dict())

    early_stop_counter = 0

    for _ in range(n_iter):
        for input_batch, output_batch in zip(train_input_batches, train_output_batches):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output_q = module_output(input_batch)
                loss_value = loss(output_batch, output_q)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            for i, scheduler in enumerate(lr_schedulers):
                scheduler.step()
                optimizer.param_groups[i]["lr"] = scheduler.get_lr()[0]

            if post_optim_callback:
                post_optim_callback(module)

        with torch.no_grad():
            val_loss_value = loss_batches(val_input_batches, val_output_batches)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_sd = deepcopy(module.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop is not None and early_stop_counter >= early_stop:
                break

        empty_cache()
        progress_bar.set_postfix(
            val_loss=val_loss_value.item(),
            val_og_loss=original_val_loss.item(),
        )
        progress_bar.update(1)

    progress_bar.close()
    logger.info(
        f"Best val loss: {best_val_loss.item()}, Original val loss: {original_val_loss.item()}"
    )

    module.load_state_dict(best_sd)
