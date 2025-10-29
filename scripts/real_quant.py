from inference_engine.utils.checkpoint_utils import from_pt_to_ckpt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert training output (.pt weights) to model checkpoint"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="path to the training output (.pt checkpoints) dir",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="checkpoint output path"
    )
    args = parser.parse_args()

    from_pt_to_ckpt(args.model, args.result_dir, args.output_path)
