import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

from paroquant.util import (
    load_model,
    load_tokenizer,
    get_blocks,
    get_named_linears,
)
from paroquant.module import PseudoQuantizedLinear

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    dtype = torch.float16
    device = "cuda"
    result_dir = Path(args.result_dir)

    model = load_model(args.model, device_map=device, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    blocks = get_blocks(model)

    for i, layer in enumerate(tqdm(blocks)):
        layer = layer.to(device)
        for name, module in get_named_linears(layer).items():
            result_file = result_dir / f"{i}.{name}.pt"
            if not result_file.exists():
                raise Exception(f"Result file not found: {result_file}")
            sd = torch.load(result_file, weights_only=False, map_location=device)
            qlayer = PseudoQuantizedLinear.from_state_dict(sd)
            weight = qlayer.pseudo_weight()
            module.weight.data.copy_(weight)

    # Save the new model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {args.output_path}")
