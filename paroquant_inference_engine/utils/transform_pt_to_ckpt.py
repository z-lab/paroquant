from paroquant_inference_engine.utils.checkpoint_utils import from_pt_to_ckpt
import argparse

def main(args):
    from_pt_to_ckpt(args.hf_model_name, args.pt_path, args.ckpt_out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert training output (.pt weights) to model checkpoint")
    parser.add_argument("--hf_model_name", type=str, required=True)
    parser.add_argument("--pt_path", type=str, required=True, help='path to the training output (.pt weights) dir')
    parser.add_argument("--ckpt_out_path", type=str, required=True, help='checkpoint output path')
    args = parser.parse_args()
    main(args)