import torch

from .util import clamp_ste, round_ste


def _align_shape(pair, angle, grp_size=128, inc_mask=False):
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
        temp_pairs = torch.zeros(
            (grp_size // 2, 2), dtype=torch.int32, device=pair.device
        )
        temp_angle = torch.zeros(
            (grp_size // 2), dtype=torch.float, device=angle.device
        )
        temp_mask = torch.zeros((grp_size // 2), dtype=torch.int32, device=angle.device)
        while count < grp_size // 2:
            if (
                pair_ptr < pair.size(0)
                and pair[pair_ptr, 0] - grp * grp_size < grp_size
                and pair[pair_ptr, 1] - grp * grp_size < grp_size
            ):
                temp_pairs[count, :] = pair[pair_ptr, :]
                temp_angle[count] = angle[pair_ptr]
                if (
                    s[pair[pair_ptr, 0] % grp_size] == 1
                    or s[pair[pair_ptr, 1] % grp_size] == 1
                ):
                    raise ValueError("illigal pair")
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

    rotation_pairs = torch.cat(pair_groups, axis=0).view(-1).contiguous()
    rotation_pairs = rotation_pairs % grp_size
    angles = torch.cat(angle_groups, axis=0)
    if inc_mask:
        masks = torch.cat(mask_groups, axis=0)
        return rotation_pairs, angles, masks
    return rotation_pairs, angles


def transform_from_pt(pt_path, krot=8, include_qsz=False):
    pt = torch.load(pt_path)
    weight = pt["weight"]
    group_size = pt["quantizer.group_size"].item()
    n_bit = pt["quantizer.n_bit"].item()
    in_feat = weight.size(1)
    pairs = []
    angles = []
    from_kernel_form = pt.get("pairs_grouped", None)

    if from_kernel_form is None:
        for i in range(krot):
            pairs_name = "pairs_grouped." + str(i)
            pair = pt[pairs_name]
            angle_name = "angles_grouped." + str(i)
            angle = pt[angle_name]
            pair, angle = _align_shape(pair, angle, group_size)
            assert (
                pair.size(0) * pair.size(1) == in_feat and angle.size(0) * 2 == in_feat
            )
            pair = pair
            pair = pair.view(-1).contiguous()
            pair = pair % group_size
            pairs.append(pair)
            angle = angle
            angles.append(angle)
        rotation_pairs = torch.stack(pairs, dim=0)
        rotation_angles = torch.stack(angles, dim=0)
    else:
        rotation_pairs = pt["pairs_grouped"]
        rotation_angles = pt["angles_grouped"]

    channel_scales = pt["channel_scales"].float()
    channel_scales = 1 / channel_scales
    channel_scales.to(weight.dtype)
    weight = weight.cpu()
    qscales = pt["quantizer.scale"]
    qzeros = pt["quantizer.zero_point_float"]
    qmin = 0
    qmax = 2**n_bit - 1
    round_zero_point = clamp_ste(-round_ste(qzeros), qmin, qmax)
    if include_qsz:
        return (
            weight,
            rotation_pairs,
            rotation_angles,
            channel_scales,
            qscales,
            qzeros,
            round_zero_point,
        )
    else:
        return weight, rotation_pairs, rotation_angles, channel_scales


def transform_to_kernel_data(
    pairs_group: list[torch.Tensor],
    angles_group: list[torch.Tensor],
    group_size: int = 128,
):
    assert len(pairs_group) == len(angles_group)
    krot = len(pairs_group)
    pairs = []
    angles = []
    masks = []

    for i in range(krot):
        pair, angle, mask = _align_shape(
            pairs_group[i], angles_group[i], grp_size=group_size, inc_mask=True
        )
        pairs.append(pair)
        angles.append(angle)
        masks.append(mask)

    rotation_pairs = torch.stack(pairs, dim=0).to(dtype=torch.short)
    angles = torch.stack(angles, dim=0).to(dtype=torch.float)
    masks = torch.stack(masks, dim=0).to(dtype=torch.bool)
    return rotation_pairs, angles, masks


def check_pairs_independent(pairs_group, group_size):
    krot = pairs_group.size(0)
    h = pairs_group.size(1)
    num_groups = h // group_size
    for i in range(krot):
        for j in range(num_groups):
            s = [0] * group_size
            for k in range(group_size):
                offset = j * group_size
                if s[pairs_group[i, k + offset].item()] == 1:
                    return False
                s[pairs_group[i, k + offset]] = 1
    return True
