"""
eval_lm.py -- zero-shot benchmarks via lm-evaluation-harness (v0.4.x) for the teacher and any converted checkpoint,
plus a perplexity-vs-length extrapolation test.

  # baseline (the unmodified pretrained model run through THIS implementation -> also validates the re-implementation)
  python eval_lm.py --pretrained state-spaces/mamba2-130m --out results/teacher.json
  # converted model
  python eval_lm.py --ckpt runs/D_shift_tern/last.pt --out results/D_shift_tern.json
  # add --ppl-length to also run the length-extrapolation test
"""
import argparse, json, math, os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ssm_core import load_pretrained, load_checkpoint, QuantConfig, get_tokenizer

DEFAULT_TASKS = "lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,wikitext"

try:
    from lm_eval.api.model import LM
except ImportError:  # allow --help and standalone long-context evaluation without lm-eval installed
    class LM:
        pass


class SSMEvalLM(LM):
    def __init__(self, model, tok, batch_size=16, max_len=2048, device="cuda", dtype=torch.bfloat16, stride=1024):
        super().__init__()
        self.model, self.tok = model.eval(), tok
        self.bs, self.max_len, self.dev, self.dtype, self.stride = batch_size, max_len, device, dtype, stride
        self.eot = tok.eos_token_id

    def _enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _encode_pair(self, ctx, cont):                                    # same trailing-space handling as HFLM
        n_sp = len(ctx) - len(ctx.rstrip())
        if n_sp:
            ctx, cont = ctx[:-n_sp], ctx[-n_sp:] + cont
        whole = self._enc(ctx + cont)
        c_enc = self._enc(ctx)
        return (c_enc or [self.eot]), whole[len(c_enc):]

    @torch.no_grad()
    def _logits(self, ids_batch):
        n = max(len(x) for x in ids_batch)
        t = torch.zeros(len(ids_batch), n, dtype=torch.long)
        for i, x in enumerate(ids_batch):
            t[i, : len(x)] = torch.tensor(x)
        with torch.autocast("cuda", dtype=self.dtype):
            return self.model(t.to(self.dev))

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm=False):
        pairs = [self._encode_pair(*r.args) for r in requests]
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        res = [None] * len(pairs)
        for s in tqdm(range(0, len(order), self.bs), disable=disable_tqdm, desc="loglikelihood"):
            idx = order[s: s + self.bs]
            inps, metas = [], []
            for i in idx:
                ctx, cont = pairs[i]
                full = (ctx + cont)[-(self.max_len + 1):]
                inps.append(full[:-1])
                metas.append((len(full) - 1, len(cont), full[-len(cont):]))
            logits = self._logits(inps)
            for j, i in enumerate(idx):
                n, cl, cont_ids = metas[j]
                lp = F.log_softmax(logits[j, n - cl: n].float(), -1)
                tgt = torch.tensor(cont_ids, device=lp.device)
                ll = lp.gather(-1, tgt[:, None]).sum().item()
                res[i] = (ll, bool((lp.argmax(-1) == tgt).all().item()))
        return res

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        out = []
        for r in tqdm(requests, disable=disable_tqdm, desc="rolling"):
            ids = [self.eot] + self._enc(r.args[0])
            total, prev_end = 0.0, 0
            for begin in range(0, len(ids) - 1, self.stride):
                end = min(begin + self.max_len, len(ids) - 1)             # inputs ids[begin:end] predict ids[begin+1:end+1]
                trg = end - max(prev_end, begin)
                inp = ids[begin:end]
                logits = self._logits([inp])[0]
                lp = F.log_softmax(logits[-trg:].float(), -1)
                tgt = torch.tensor(ids[end - trg + 1: end + 1], device=lp.device)
                total += lp.gather(-1, tgt[:, None]).sum().item()
                prev_end = end
                if end == len(ids) - 1:
                    break
            out.append(total)
        return out

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("only loglikelihood-based tasks are supported")


@torch.no_grad()
def ppl_vs_length(model, tok, lengths=(512, 1024, 2048, 4096, 8192), n_seq=8, dataset="HuggingFaceFW/fineweb-edu",
                  subset="sample-10BT", val_modulo=100, dev="cuda", dtype=torch.float16):
    """Loss by position bucket on long held-out text (docs concatenated). Pretraining length was 2048:
    a stable recurrence should not blow up beyond it."""
    from datasets import load_dataset
    L = max(lengths)
    ds = load_dataset(dataset, name=subset, split="train", streaming=True)
    ds = ds.filter(lambda ex, idx: idx % val_modulo == 0, with_indices=True).take(50000)
    buf, seqs = [], []
    for ex in ds:
        buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= L + 1 and len(seqs) < n_seq:
            seqs.append(torch.tensor(buf[: L + 1])); buf = buf[L:]
        if len(seqs) >= n_seq:
            break
    sums = torch.zeros(L); cnt = 0
    for s in seqs:
        x, y = s[:-1][None].to(dev), s[1:].to(dev)
        with torch.autocast("cuda", dtype=dtype):
            lg = model(x)[0]
        sums += F.cross_entropy(lg.float(), y, reduction="none").cpu(); cnt += 1
    per_pos = sums / cnt
    edges = [0] + list(lengths)
    return {f"{edges[i]}-{edges[i+1]}": per_pos[edges[i]:edges[i+1]].mean().item() for i in range(len(lengths))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (debug)")
    ap.add_argument("--ppl-length", action="store_true")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--val-modulo", type=int, default=100)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Language-model evaluation requires CUDA in this implementation.")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if a.dtype == "auto" else (torch.bfloat16 if a.dtype == "bf16" else torch.float16)
    if a.ckpt:
        model, _ = load_checkpoint(a.ckpt, "cuda")
    else:
        model, _ = load_pretrained(a.pretrained or "state-spaces/mamba2-130m", QuantConfig())
        model = model.to("cuda")
    tok = get_tokenizer()
    lm = SSMEvalLM(model, tok, a.batch_size, dtype=amp_dtype)

    import lm_eval
    res = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","), num_fewshot=0, limit=a.limit)
    out = {"results": res["results"], "evaluation_protocol": {
        "lm_eval_tasks": a.tasks.split(","), "fewshot": 0, "limit": a.limit,
        "long_context_dataset": a.dataset, "long_context_subset": a.subset,
        "long_context_validation_rule": f"document_index % {a.val_modulo} == 0",
        "synthetic_evaluation_data": False, "dtype": str(amp_dtype),
    }}
    if a.ppl_length:
        out["loss_by_position"] = ppl_vs_length(
            model, tok, dataset=a.dataset, subset=a.subset, val_modulo=a.val_modulo, dtype=amp_dtype
        )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    for t, r in res["results"].items():
        print(t, {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and "stderr" not in k})
    if a.ppl_length:
        print("loss by position:", out["loss_by_position"])


if __name__ == "__main__":
    main()
