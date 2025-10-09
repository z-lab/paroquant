import argparse
import random
import transformers
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset


def get_wikitext2(seed, seqlen, tokenizer):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc


def get_c4(seed, seqlen, tokenizer):
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(seed)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return valenc


def get_test_tokens(name, seed, seqlen, tokenizer):
    if name == "wikitext2":
        return get_wikitext2(seed, seqlen, tokenizer).input_ids
    elif name == "c4":
        return get_c4(seed, seqlen, tokenizer).input_ids
    else:
        raise ValueError(f"Unknown dataset {name}")


def model_from_hf_path(path, device_map="auto"):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if "AWQ" in path:
        model = model.model

    return model


def main(args):
    datasets = ["wikitext2", "c4"]
    model = model_from_hf_path(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for dataset in datasets:
        input_tok = get_test_tokens(
            dataset, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer
        )
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, : (args.seqlen * nsamples)].view(nsamples, args.seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(
                input,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            output = output[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"{dataset} perplexity: {ppl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--seqlen", type=int)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    main(args)
