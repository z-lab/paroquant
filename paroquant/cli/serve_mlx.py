"""OpenAI-compatible MLX server that auto-loads ParoQuant models."""

from __future__ import annotations

import mlx_lm.server
from mlx_lm.server import ModelProvider
from mlx_lm.utils import load_tokenizer

from paroquant.inference.backends.mlx.load import load as paro_load


def _paro_load(self, model_path, adapter_path=None, draft_model_path=None):
    """Override that uses the ParoQuant loader for model weights."""
    model_path = self.default_model_map.get(model_path, model_path)
    if self.model_key == (model_path, adapter_path, draft_model_path):
        return self.model, self.tokenizer

    actual_path = model_path if model_path != "default_model" else self.cli_args.model
    model, _, _ = paro_load(actual_path, force_text=True)
    tokenizer = load_tokenizer(actual_path)

    if self.cli_args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    tokenizer._tool_call_start = None
    tokenizer._tool_call_end = None

    self.model_key = (model_path, adapter_path, draft_model_path)
    self.model = model
    self.tokenizer = tokenizer
    self.draft_model = None
    self.is_batchable = False

    return self.model, self.tokenizer


def main():
    ModelProvider.load = _paro_load
    mlx_lm.server.main()


if __name__ == "__main__":
    main()
