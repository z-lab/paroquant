import torch, torch.nn as nn
from inference_engine.model_executor.modules.qmodule import WQLinear
from inference_engine.utils.rotation_utils import rotate_tensor, quantizer
from inference_engine.utils.quantization_utils import pseudo_quantize_tensor
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
)


class Rotation(nn.Module):
    def __init__(
        self,
        dim,
        krot=8,
        group_size=128,
        rotation_angles=None,
        rotation_pairs=None,
        channel_scales=None,
    ):
        super().__init__()
        if krot != 8 or group_size != 128:
            raise NotImplementedError()
        assert dim % group_size == 0

        KROT = krot
        GROUP_SIZE = group_size
        num_groups = dim // GROUP_SIZE
        device = "cuda"

        if rotation_angles is None:
            rotation_angles = torch.zeros(
                KROT, dim // 2, device="cuda", dtype=torch.float16
            )

        if rotation_pairs is None:
            single_group_idx_ij = torch.randperm(
                GROUP_SIZE, dtype=torch.int, device=device
            )
            one_krot_layer_idx_ij = single_group_idx_ij.repeat(num_groups)
            rotation_pairs = one_krot_layer_idx_ij.unsqueeze(0).expand(KROT, -1)

        if channel_scales is None:
            channel_scales = torch.ones(1, dim, dtype=torch.half)

        self.register_buffer("theta", rotation_angles.clone().contiguous())
        self.register_buffer("pairs", rotation_pairs.clone().short().contiguous())
        self.register_buffer("channel_scales", channel_scales.clone().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.ops.rotation.rotate(
            x, self.pairs, self.theta, self.channel_scales
        )
        return output


class RotateLinearInt4(nn.Module):

    def __init__(
        self,
        in_feat,
        out_feat,
        bias,
        dtype,
        rotation_angles=None,
        rotation_pairs=None,
        channel_scales=None,
    ):
        super().__init__()
        self.in_features = in_feat
        self.out_features = out_feat
        self.rotation = Rotation(
            in_feat,
            rotation_angles=rotation_angles,
            rotation_pairs=rotation_pairs,
            channel_scales=channel_scales,
        )
        self.qlinear = WQLinear(
            4, 128, self.in_features, self.out_features, bias, dtype=dtype
        )

    @classmethod
    def from_linear(
        cls,
        linear,
        rotation_angles=None,
        rotation_pairs=None,
        channel_scales=None,
        qscales=None,
        qzeros=None,
        rotate_weight=False,
        init_only=True,
    ):
        rotate_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.dtype,
            rotation_angles,
            rotation_pairs,
            channel_scales,
        )
        if rotate_weight:
            with torch.no_grad():
                w = linear.weight.data.detach().clone()
                w = rotate_tensor(w, rotation_angles, rotation_pairs, channel_scales)
                if qscales is not None:
                    w = pseudo_quantize_tensor(
                        w, 4, q_group_size=128, qzeros=qzeros, qscales=qscales
                    )
                    qzeros = qzeros.view(w.size(0), -1)
                    qscales = qscales.view(w.size(0), -1)
                linear.weight.copy_(w)
        if init_only:
            return rotate_linear
        linear.to(torch.half)

        assert (qscales is None) == (qzeros is None)
        rotate_linear.qlinear = quantizer(
            linear, 4, 128, qscales=qscales, qzeros=qzeros
        )
        return rotate_linear

    @torch.no_grad()
    def forward(self, x):
        x = self.rotation(x)
        x = self.qlinear(x)
        return x

    def buffer_name(self, buffer_name: str):
        if buffer_name == "qlinear.qweight":
            return self.qlinear.qweight
        elif buffer_name == "qlinear.scaled_zeros":
            return self.qlinear.scaled_zeros
        elif buffer_name == "qlinear.scales":
            return self.qlinear.scales
        elif buffer_name == "rotation.theta":
            return self.rotation.theta
        elif buffer_name == "rotation.pairs":
            return self.rotation.pairs
        elif buffer_name == "rotation.channel_scales":
            return self.rotation.channel_scales
        raise ValueError(f"Invalid buffer name: {buffer_name}")


class AWQMarlinLinear(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dtype=torch.float16
    ):
        super().__init__()
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.pack_factor = 32 // self.w_bit

        assert self.in_features % self.group_size == 0
        assert self.out_features % self.pack_factor == 0

        num_groups = self.in_features // self.group_size
        self.register_buffer(
            "qweight",
            torch.zeros(
                (self.in_features, self.out_features // self.pack_factor),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (num_groups, self.out_features // self.pack_factor),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros((num_groups, self.out_features), dtype=dtype),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype))
        else:
            self.bias = None

        self.quant_type = scalar_types.uint4
        self.workspace = None
        self.g_idx = None
        self.g_idx_sort_indices = None
        self.is_marlin_converted = False

    def convert_to_marlin(self):
        if self.is_marlin_converted:
            return
        device = self.qweight.device
        self.workspace = marlin_make_workspace_new(device)
        self.g_idx = marlin_make_empty_g_idx(device)
        self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        marlin_qweight = ops.awq_marlin_repack(
            self.qweight,
            size_k=self.in_features,
            size_n=self.out_features,
            num_bits=self.w_bit,
        )
        self.qweight = marlin_qweight

        marlin_scales = marlin_permute_scales(
            self.scales,
            size_k=self.in_features,
            size_n=self.out_features,
            group_size=self.group_size,
        )
        self.scales = marlin_scales

        marlin_zp = awq_to_marlin_zero_points(
            self.qzeros,
            size_k=self.in_features // self.group_size,
            size_n=self.out_features,
            num_bits=self.w_bit,
        )
        self.qzeros = marlin_zp

        self.is_marlin_converted = True

    @torch.no_grad()
    def forward(self, inputs):
        if not self.is_marlin_converted:
            self.convert_to_marlin()
        return apply_awq_marlin_linear(
            input=inputs,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=self.quant_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            bias=self.bias,
        )


class RotateLinearAWQMarlinInt4(nn.Module):

    def __init__(
        self,
        in_feat,
        out_feat,
        bias,
        dtype,
        rotation_angles=None,
        rotation_pairs=None,
        channel_scales=None,
    ):
        super().__init__()
        self.in_features = in_feat
        self.out_features = out_feat
        self.rotation = Rotation(
            in_feat,
            rotation_angles=rotation_angles,
            rotation_pairs=rotation_pairs,
            channel_scales=channel_scales,
        )
        self.qlinear = AWQMarlinLinear(
            4, 128, self.in_features, self.out_features, bias, dtype=dtype
        )

    @torch.no_grad()
    def forward(self, x):
        x = self.rotation(x)
        x = self.qlinear(x)
        return x

    def buffer_name(self, buffer_name: str):
        if buffer_name == "qlinear.qweight":
            return self.qlinear.qweight
        elif buffer_name == "qlinear.qzeros":
            return self.qlinear.qzeros
        elif buffer_name == "qlinear.scales":
            return self.qlinear.scales
        elif buffer_name == "rotation.theta":
            return self.rotation.theta
        elif buffer_name == "rotation.pairs":
            return self.rotation.pairs
        elif buffer_name == "rotation.channel_scales":
            return self.rotation.channel_scales
        raise ValueError(f"Invalid buffer name: {buffer_name}")
