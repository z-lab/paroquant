#pragma once
#include <torch/extension.h>

torch::Tensor rotate_dynamic(at::Tensor x, at::Tensor idx, at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt);