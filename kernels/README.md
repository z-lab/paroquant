# Fast Givens Transform

This repository provides a fast Givens transform implementation for PyTorch, leveraging custom CUDA kernels for improved performance.

## Installation

To install the package, navigate to the root directory of this repository and run:

```bash
pip install .
```

## Usage

The core functionality is exposed through the `fast_givens_transform` function and `RotateTensorFunc` autograd function.

### `fast_givens_transform`

This function applies the Givens rotation using the custom CUDA kernel.

### `RotateTensorFunc`

This is a `torch.autograd.Function` that allows for backpropagation through the Givens rotation operation.

### `transform_from_ckpt`

This is a utility function for converting checkpoint file (e.g. 0.self_attn.q_proj) to kernel compatible data

