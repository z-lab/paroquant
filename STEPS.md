git clone git@github.com:keskinonur/paroquant-export.git
cd paroquant-export

git checkout codex/export-compat
uv venv
uv pip install -e ".[export,mlx]"

git clone https://github.com/ggml-org/llama.cpp ./src/llama.cpp
brew install cmake
cmake -S ./src/llama.cpp -B ./src/llama.cpp/build -DGGML_METAL=ON
cmake --build ./src/llama.cpp/build -j

export HF_TOKEN=hf_ABCxyz

set -a; source .env; set +a

uv run python -m paroquant.cli.export \
--model z-lab/Qwen3.5-9B-PARO \
--output-dir output/qwen3.5-9b-export \
--targets hf,mlx,gguf,ollama,lmstudio \
--text-only \
--gguf-quants Q4_K_M,Q8_0 \
--llama-cpp-dir ./src/llama.cpp
