from __future__ import annotations

import torch


def _align_shape(pair, angle, grp_size=128, inc_mask=False):
    """Align rotation pairs into fixed-size groups with dummy padding."""
    assert pair.size(0) == angle.size(0)
    assert pair.size(1) == 2

    grp = 0
    pair_ptr = 0

    pair_groups = []
    angle_groups = []
    mask_groups = []
    while True:
        if pair_ptr >= pair.size(0):
            break
        s = torch.zeros((grp_size), dtype=torch.int32)
        count = 0
        temp_pairs = torch.zeros((grp_size // 2, 2), dtype=torch.int32, device=pair.device)
        temp_angle = torch.zeros((grp_size // 2), dtype=torch.float, device=angle.device)
        temp_mask = torch.zeros((grp_size // 2), dtype=torch.int32, device=angle.device)
        while count < grp_size // 2:
            if (
                pair_ptr < pair.size(0)
                and pair[pair_ptr, 0] - grp * grp_size < grp_size
                and pair[pair_ptr, 1] - grp * grp_size < grp_size
            ):
                temp_pairs[count, :] = pair[pair_ptr, :]
                temp_angle[count] = angle[pair_ptr]
                if s[pair[pair_ptr, 0] % grp_size] == 1 or s[pair[pair_ptr, 1] % grp_size] == 1:
                    raise ValueError("illegal pair")
                s[pair[pair_ptr, :] % grp_size] = 1
                pair_ptr += 1
            else:
                t_pair = torch.tensor([-1, -1])
                for i in range(grp_size):
                    if s[i] == 0:
                        t_pair[0] = i
                        s[i] = 1
                        break
                for i in range(grp_size):
                    if s[i] == 0:
                        t_pair[1] = i
                        s[i] = 1
                        break
                if t_pair[0] == -1 or t_pair[1] == -1:
                    raise ValueError("can't find a dummy pair")
                temp_pairs[count, :] = t_pair
                temp_angle[count] = float(0)
                temp_mask[count] = 1
            count += 1
        grp += 1
        pair_groups.append(temp_pairs)
        angle_groups.append(temp_angle)
        mask_groups.append(temp_mask)

    rotation_pairs = torch.cat(pair_groups, dim=0).view(-1).contiguous()
    rotation_pairs = rotation_pairs % grp_size
    angles = torch.cat(angle_groups, dim=0)
    if inc_mask:
        masks = torch.cat(mask_groups, dim=0)
        return rotation_pairs, angles, masks
    return rotation_pairs, angles


def transform_to_kernel_data(
    pairs_group: list[torch.Tensor],
    angles_group: list[torch.Tensor],
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert optimization pair/angle lists to kernel-ready format with masks."""
    assert len(pairs_group) == len(angles_group)
    pairs, angles, masks = [], [], []
    for i in range(len(pairs_group)):
        pair, angle, mask = _align_shape(pairs_group[i], angles_group[i], grp_size=group_size, inc_mask=True)
        pairs.append(pair)
        angles.append(angle)
        masks.append(mask)

    return (
        torch.stack(pairs, dim=0).to(dtype=torch.short),
        torch.stack(angles, dim=0).to(dtype=torch.float),
        torch.stack(masks, dim=0).to(dtype=torch.bool),
    )
