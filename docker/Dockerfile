ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

LABEL org.opencontainers.image.source=https://github.com/z-lab/paroquant
LABEL org.opencontainers.image.licenses=MIT

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    curl \
    ca-certificates \
    libibverbs-dev \
    libnuma-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0 10.0 12.0 12.1"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ---- chat: interactive vLLM inference ----
FROM base AS chat
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ARG VLLM_VERSION=""
ARG CUDA_TOOLKIT=""
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$VLLM_VERSION" ] && [ -n "$CUDA_TOOLKIT" ]; then \
        pip install -e ".[vllm]" "vllm==${VLLM_VERSION}" \
            --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TOOLKIT}" \
            --extra-index-url "https://download.pytorch.org/whl/${CUDA_TOOLKIT}"; \
    else \
        pip install -e ".[vllm]"; \
    fi
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
ENTRYPOINT ["python", "-m", "paroquant.cli.chat"]

# ---- serve: OpenAI-compatible vLLM API server ----
FROM base AS serve
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ARG VLLM_VERSION=""
ARG CUDA_TOOLKIT=""
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$VLLM_VERSION" ] && [ -n "$CUDA_TOOLKIT" ]; then \
        pip install -e ".[vllm]" "vllm==${VLLM_VERSION}" \
            --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TOOLKIT}" \
            --extra-index-url "https://download.pytorch.org/whl/${CUDA_TOOLKIT}"; \
    else \
        pip install -e ".[vllm]"; \
    fi
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
ENTRYPOINT ["python", "-m", "paroquant.cli.serve"]

# ---- optim: optimization & evaluation ----
FROM base AS optim
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[optim,eval]"
CMD ["/bin/bash"]

# ---- eval: reasoning task evaluation ----
FROM base AS eval
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ./experiments/tasks/reasoning/requirements.txt
RUN pip install -e . --no-cache-dir --no-deps
CMD ["/bin/bash"]
