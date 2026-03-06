// Empty pybind module — the rotation kernel registers itself via TORCH_LIBRARY in rotation.cu.
// JIT-compiled by torch.utils.cpp_extension.load() on first import.
#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
