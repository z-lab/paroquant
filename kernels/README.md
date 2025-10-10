#  Scaled Pairwise Rotation

This repository provides a scaled pairwise rotation implementation for PyTorch, leveraging custom CUDA kernels for improved performance.

## Installation

To install the package, navigate to the root directory of this directory and run:

```bash
pip install .
```

## Usage

The core functionality is exposed through the `scaled_pairwise_rotation` function and `RotateTensorFunc` autograd function.

### `scaled_pairwise_rotation`

This function applies the Givens rotation using the custom CUDA kernel.

### `RotateTensorFunc`

This is a `torch.autograd.Function` that allows for backpropagation through the Givens rotation operation.