import torch
from lm_eval.base import BaseLM

class LMEvalAdaptor(BaseLM):

    def __init__(self, model: torch.nn.Module, tokenizer, batch_size=1, max_length=None):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = batch_size
        self._max_length = self.model.config.max_position_embeddings if max_length is None else max_length

    @property
    def eot_token_id(self):
        # End of Text token ID
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor):
        outputs = []
        for i in range(inps.shape[0]):
            inp_single = inps[i].unsqueeze(0)
            
            with torch.no_grad():
                h = self.model.model.ppl_forward(tokens=inp_single, start_pos=0, chunk_prefilling=False)
                logits = self.model.lm_head(h)
            outputs.append(logits)

        return torch.cat(outputs, dim=0)


    def _model_generate(self, context: torch.Tensor, max_length: int, eos_token_id: int):
        assert context.shape[0] == 1, "Generation only supports a batch size of 1."
        
        generated_tokens = []
        with torch.no_grad():
            logits = self.model(tokens=context, start_pos=0)
            for i in range(max_length):
                next_token = torch.argmax(logits.squeeze(1), dim=-1)

                if next_token.item() == eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_token = next_token.unsqueeze(0)
                start_pos = context.shape[1] + i
                
                logits = self.model(tokens=input_token, start_pos=start_pos)
        if not generated_tokens:
            return context
        
        return torch.cat([
            context,
            torch.tensor([generated_tokens], device=self.device, dtype=torch.long)
        ], dim=1)