/******************************************************************************
 * Copyright (c) 2025, Haisheng Chen.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename scalar_t> struct RotateAccess;

template <> struct RotateAccess<float> {
  // load & scale into shared buffer
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void
  load_group(float *__restrict__ x_grp, const float *__restrict__ x,
             const float *__restrict__ scales, const int s, const int h,
             const int j, const int g, const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
    float scale0 = USE_SCALE ? scales[base0] : float(1);
    float scale1 = USE_SCALE ? scales[base1] : float(1);
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[t * CTA_M + i] = x[row * h + base0] * scale0;
        x_grp[(t + GROUP_SIZE / 2) * CTA_M + i] = x[row * h + base1] * scale1;
      }
    }
  }

  // load sin/cos and idx into regs
  template <int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(float reg_theta[KROT], int reg_idx[KROT],
                                     const int16_t *__restrict__ idx_ij,
                                     const float *__restrict__ theta,
                                     const int h, const int g, const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = theta[r * h / 2 + g * GROUP_SIZE / 2 + t];
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h +
                                                  g * GROUP_SIZE + 2 * t);
    }
  }

  // apply one Givens rotation
  template <int CTA_M>
  __device__ static void apply_one(float *__restrict__ x_grp, const int ij,
                                   const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
#pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i * CTA_M + m];
      float xj = x_grp[j * CTA_M + m];
      x_grp[i * CTA_M + m] = xi * c_ + xj * s_;
      x_grp[j * CTA_M + m] = xi * (-s_) + xj * c_;
    }
  }

  template <int CTA_M, int GROUP_SIZE>
  __device__ static void
  store_group(float *__restrict__ out, const float *__restrict__ x_grp,
              const int s, const int h, const int j, const int g, const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = x_grp[t * CTA_M + i];
        out[row * h + base1] = x_grp[(t + GROUP_SIZE / 2) * CTA_M + i];
      }
    }
  }
};

template <> struct RotateAccess<__half> {
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void
  load_group(__half *__restrict__ x_grp, const __half *__restrict__ x,
             const __half *__restrict__ scales, const int s, const int h,
             const int j, const int g, const int t) {
    // load two elements at once with __half2
    const int offset = GROUP_SIZE * g + 2 * t;
    __half scale_i, scale_j;
    if constexpr (USE_SCALE) {
      __half2 scale_pair = *reinterpret_cast<const __half2 *>(scales + offset);
      scale_i = __low2half(scale_pair);
      scale_j = __high2half(scale_pair);
    } else {
      scale_i = __half(1);
      scale_j = __half(1);
    }

#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        __half2 x2 = *reinterpret_cast<const __half2 *>(x + row * h + offset);
        __half lo = __hmul(__low2half(x2), scale_i);
        __half hi = __hmul(__high2half(x2), scale_j);
        x_grp[(2 * t) * CTA_M + i] = lo;
        x_grp[(2 * t + 1) * CTA_M + i] = hi;
      }
    }
  }

  template <int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(float reg_theta[KROT], int reg_idx[KROT],
                                     const int16_t *__restrict__ idx_ij,
                                     const __half *__restrict__ theta,
                                     const int h, const int g, const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] =
          static_cast<float>(theta[r * h / 2 + g * GROUP_SIZE / 2 + t]);
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h +
                                                  g * GROUP_SIZE + 2 * t);
    }
  }

  template <int CTA_M>
  __device__ static void apply_one(__half *__restrict__ x_grp, const int ij,
                                   const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;

    float s_, c_;
    __sincosf(theta, &s_, &c_);

#pragma unroll
    for (int m = 0; m < CTA_M / 2; ++m) {

      __half2 *pi2 = reinterpret_cast<__half2 *>(x_grp + i * CTA_M + m * 2);
      __half2 *pj2 = reinterpret_cast<__half2 *>(x_grp + j * CTA_M + m * 2);
      __half2 xi_h2 = *pi2;
      __half2 xj_h2 = *pj2;

      float2 xi = __half22float2(xi_h2);
      float2 xj = __half22float2(xj_h2);

      float2 yi, yj;
      yi.x = fmaf(c_, xi.x, s_ * xj.x); // c*xi + s*xj
      yi.y = fmaf(c_, xi.y, s_ * xj.y);

      yj.x = fmaf(c_, xj.x, -s_ * xi.x); // c*xj - s*xi
      yj.y = fmaf(c_, xj.y, -s_ * xi.y);

      *pi2 = __floats2half2_rn(yi.x, yi.y);
      *pj2 = __floats2half2_rn(yj.x, yj.y);
    }
  }

  template <int CTA_M, int GROUP_SIZE>
  __device__ static void
  store_group(__half *__restrict__ out, const __half *__restrict__ x_grp,
              const int s, const int h, const int j, const int g, const int t) {
    const int base = GROUP_SIZE * g + 2 * t;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        __half2 out2;
        out2.x = x_grp[(2 * t) * CTA_M + i];
        out2.y = x_grp[(2 * t + 1) * CTA_M + i];
        *reinterpret_cast<__half2 *>(out + row * h + base) = out2;
      }
    }
  }
};