// Empty pybind module — rotation kernel registers itself via TORCH_LIBRARY in rotation.cu.
// This file exists so that `from paroquant_kernels import _C` loads the compiled .so.
#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
