/******************************************************************************
 * Copyright (c) 2025, Haisheng Chen.
 ******************************************************************************/

#include "rotation.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <torch/extension.h>

template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT,
          bool USE_SCALE>
__global__ void
rotate(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
       const int16_t *__restrict__ idx_ij, const scalar_t *__restrict__ theta,
       const scalar_t *__restrict__ scales, int s, int h) {
  // shared buffer
  static_assert(CTA_M % 2 == 0, "CTA_M must be even");
  static_assert(GROUP_SIZE % 2 == 0, "GROUP_SIZE must be even");
  __shared__ scalar_t x_grp[CTA_M * GROUP_SIZE];

  int j = blockIdx.x;  // row‐block
  int g = blockIdx.y;  // group
  int t = threadIdx.x; // pair within group

  // load & scale into shared memory
  RotateAccess<scalar_t>::template load_group<CTA_M, GROUP_SIZE, USE_SCALE>(
      x_grp, x, scales, s, h, j, g, t);

  // fetch sin/cos + idx into registers
  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(
      reg_theta, reg_idx, idx_ij, theta, h, g, t);
  __syncthreads();

// apply all KROT rotations
#pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M>(x_grp, reg_idx[r],
                                                      reg_theta[r]);
    __syncthreads();
  }

  // write back
  RotateAccess<scalar_t>::template store_group<CTA_M, GROUP_SIZE>(out, x_grp, s,
                                                                  h, j, g, t);
}

// C++ launcher
template <int KROT, int CTA_M, int GROUP_SIZE>
torch::Tensor rotate_launcher(at::Tensor x, at::Tensor idx_ij, at::Tensor theta,
                              at::Tensor scales) {
  int h = x.size(-1);
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  int seq_len = x.numel() / x.size(-1);
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::empty(x.sizes(), options);
  bool has_scale = scales.defined() && scales.numel() > 0;

  dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
  dim3 block(pn);

  // Launch kernel
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto dtype = x.scalar_type();
  switch (dtype) {
  case at::kFloat: {
    if (has_scale) {
      rotate<float, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
          x.data_ptr<float>(), out.data_ptr<float>(),
          idx_ij.data_ptr<int16_t>(), theta.data_ptr<float>(),
          scales.data_ptr<float>(), seq_len, h);
    } else {
      rotate<float, CTA_M, GROUP_SIZE, KROT, false><<<grid, block, 0, stream>>>(
          x.data_ptr<float>(), out.data_ptr<float>(),
          idx_ij.data_ptr<int16_t>(), theta.data_ptr<float>(), nullptr, seq_len,
          h);
    }
    break;
  }
  case at::kHalf: {
    __half *x_ptr = reinterpret_cast<__half *>(x.data_ptr<c10::Half>());
    __half *out_ptr = reinterpret_cast<__half *>(out.data_ptr<c10::Half>());
    __half *theta_ptr = reinterpret_cast<__half *>(theta.data_ptr<c10::Half>());
    __half *scales_ptr = nullptr;
    if (has_scale) {
      scales_ptr = reinterpret_cast<__half *>(scales.data_ptr<c10::Half>());
    }
    if (has_scale) {
      rotate<__half, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
          x_ptr, out_ptr, idx_ij.data_ptr<int16_t>(), theta_ptr, scales_ptr,
          seq_len, h);
    } else {
      rotate<__half, CTA_M, GROUP_SIZE, KROT, false>
          <<<grid, block, 0, stream>>>(x_ptr, out_ptr,
                                       idx_ij.data_ptr<int16_t>(), theta_ptr,
                                       nullptr, seq_len, h);
    }
    break;
  }
  default:
    TORCH_CHECK(false, "rotate only supports Float and Half, but got ", dtype);
  }
  return out;
}

// Group size = 128
torch::Tensor rotate_k16g128(at::Tensor x, at::Tensor idx, at::Tensor th,
                             at::Tensor sc) {
  return rotate_launcher<16, 4, 128>(x, idx, th, sc);
}
torch::Tensor rotate_k8g128(at::Tensor x, at::Tensor idx, at::Tensor th,
                            at::Tensor sc) {
  return rotate_launcher<8, 4, 128>(x, idx, th, sc);
}
torch::Tensor rotate_k4g128(at::Tensor x, at::Tensor idx, at::Tensor th,
                            at::Tensor sc) {
  return rotate_launcher<4, 4, 128>(x, idx, th, sc);
}
torch::Tensor rotate_k2g128(at::Tensor x, at::Tensor idx, at::Tensor th,
                            at::Tensor sc) {
  return rotate_launcher<2, 4, 128>(x, idx, th, sc);
}
torch::Tensor rotate_k1g128(at::Tensor x, at::Tensor idx, at::Tensor th,
                            at::Tensor sc) {
  return rotate_launcher<1, 4, 128>(x, idx, th, sc);
}

// Group size = 64
torch::Tensor rotate_k16g64(at::Tensor x, at::Tensor idx, at::Tensor th,
                            at::Tensor sc) {
  return rotate_launcher<16, 4, 64>(x, idx, th, sc);
}
torch::Tensor rotate_k8g64(at::Tensor x, at::Tensor idx, at::Tensor th,
                           at::Tensor sc) {
  return rotate_launcher<8, 4, 64>(x, idx, th, sc);
}
torch::Tensor rotate_k4g64(at::Tensor x, at::Tensor idx, at::Tensor th,
                           at::Tensor sc) {
  return rotate_launcher<4, 4, 64>(x, idx, th, sc);
}
torch::Tensor rotate_k2g64(at::Tensor x, at::Tensor idx, at::Tensor th,
                           at::Tensor sc) {
  return rotate_launcher<2, 4, 64>(x, idx, th, sc);
}
torch::Tensor rotate_k1g64(at::Tensor x, at::Tensor idx, at::Tensor th,
                           at::Tensor sc) {
  return rotate_launcher<1, 4, 64>(x, idx, th, sc);
}

torch::Tensor rotate_dynamic(at::Tensor x, at::Tensor idx, at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt,
                             int64_t group_size = 128) {
  int64_t krot = theta.size(0);
  TORCH_CHECK(krot == idx.size(0), "theta.size(0) must equal idx_ij.size(0)");
  at::Tensor scales = scales_opt.value_or(at::Tensor());

  if (group_size == 128) {
    switch (krot) {
    case 16:
      return rotate_k16g128(x, idx, theta, scales);
    case 8:
      return rotate_k8g128(x, idx, theta, scales);
    case 4:
      return rotate_k4g128(x, idx, theta, scales);
    case 2:
      return rotate_k2g128(x, idx, theta, scales);
    case 1:
      return rotate_k1g128(x, idx, theta, scales);
    default:
      TORCH_CHECK(false, "Unsupported KROT = ", krot,
                  "; compiled variants: 1/2/4/8");
    }
  } else if (group_size == 64) {
    switch (krot) {
    case 16:
      return rotate_k16g64(x, idx, theta, scales);
    case 8:
      return rotate_k8g64(x, idx, theta, scales);
    case 4:
      return rotate_k4g64(x, idx, theta, scales);
    case 2:
      return rotate_k2g64(x, idx, theta, scales);
    case 1:
      return rotate_k1g64(x, idx, theta, scales);
    default:
      TORCH_CHECK(false, "Unsupported KROT = ", krot,
                  "; compiled variants: 1/2/4/8");
    }
  }

  TORCH_CHECK(false, "Unexpected group_size: ", group_size);
}

// C++ launcher
torch::Tensor rotate_k8g128half_launcher(at::Tensor x, at::Tensor idx_ij,
                                         at::Tensor theta, at::Tensor scales) {
  int h = x.size(-1);
  const int KROT = 8;
  const int CTA_M = 4;
  const int GROUP_SIZE = 128;
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  int seq_len = x.numel() / x.size(-1);
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::empty(x.sizes(), options);
  dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
  dim3 block(pn);

  // Launch kernel
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  __half *x_ptr = reinterpret_cast<__half *>(x.data_ptr<c10::Half>());
  __half *out_ptr = reinterpret_cast<__half *>(out.data_ptr<c10::Half>());
  __half *theta_ptr = reinterpret_cast<__half *>(theta.data_ptr<c10::Half>());
  __half *scales_ptr = reinterpret_cast<__half *>(scales.data_ptr<c10::Half>());

  rotate<__half, CTA_M, GROUP_SIZE, KROT, true>
      <<<grid, block, 0, stream>>>(x_ptr, out_ptr, idx_ij.data_ptr<int16_t>(),
                                   theta_ptr, scales_ptr, seq_len, h);
  return out;
}

TORCH_LIBRARY(rotation, m) {
  m.def("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales=None, "
        "int group_size=128) -> Tensor");
}

TORCH_LIBRARY_IMPL(rotation, CUDA, m) { m.impl("rotate", &rotate_dynamic); }
