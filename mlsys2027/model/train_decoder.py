#!/usr/bin/env python
"""Train the cue-conditioned decoder: reconstruct, or abstain.

The decoder sees a compacted memory plus the cues available at query time and
must either reconstruct the original or say it cannot. Its primary metric is
NOT reconstruction fidelity -- it is whether it abstains on the examples where
the information is genuinely gone. A model that always emits text scores well
on BLEU and is useless, because a third of the corpus is unrecoverable by
construction.

Two heads on a shared decoder-only backbone:
  LM head          generate the reconstruction
  abstention head  P(reconstructable | compacted, cues), supervised by the
                   recoverability label the data generator computed

Loss (paper Eq. 1):
    L = L_recon  +  lambda_cal * L_abstain
  where L_recon is masked out on `unrecoverable` examples -- we never teach the
  model to invent content it cannot know. That masking is the whole design.

Single-node multi-GPU via accelerate/DDP. Defaults target one H100 node.

Usage:
  accelerate launch train_decoder.py --data ../data_gen/pairs.jsonl \
      --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --bs 8
"""
from __future__ import annotations

import argparse, json, math, os, random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROMPT = """You are reconstructing a degraded memory.

COMPACTED MEMORY (tier {tier}):
{compacted}

CUES AVAILABLE NOW:
  keywords: {keywords}
  tag: {tag}
  topic: {topic}
  recorded: day {t_days}
  related memories:
{neighbours}

If the cues let you restore what the compacted memory lost, write the restored
memory. If they do not, reply exactly: INSUFFICIENT.

RESTORED:"""


def render(ex: dict) -> str:
    c = ex["cues"]
    nb = "\n".join(f"    - {g}" for g in c.get("neighbour_gists", [])) or "    (none)"
    return PROMPT.format(tier=ex["tier"], compacted=ex["compacted"],
                         keywords=", ".join(c.get("keywords", [])) or "(none)",
                         tag=c.get("tag") or "(none)", topic=c.get("topic") or "(none)",
                         t_days=c.get("t_days"), neighbours=nb)


class Pairs(Dataset):
    def __init__(self, path, tok, max_len=1024, split="train", val_frac=0.1, seed=0):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        rng = random.Random(seed); rng.shuffle(rows)
        cut = int(len(rows) * (1 - val_frac))
        self.rows = rows[:cut] if split == "train" else rows[cut:]
        self.tok, self.max_len = tok, max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        src = render(ex)
        # unrecoverable -> the target IS the abstention token; we never supervise
        # the model to produce content it cannot have known.
        recoverable = ex["recoverability"] != "unrecoverable"
        tgt = ex["original"] if recoverable else "INSUFFICIENT"
        full = src + " " + tgt + self.tok.eos_token
        enc = self.tok(full, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        src_len = len(self.tok(src, truncation=True, max_length=self.max_len)["input_ids"])
        ids = enc["input_ids"][0]
        labels = ids.clone()
        labels[:src_len] = -100                     # loss only on the continuation
        return {"input_ids": ids, "labels": labels,
                "recoverable": torch.tensor(float(recoverable)),
                "src_len": src_len}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = b["input_ids"]; lab[i, :L] = b["labels"]; att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att,
            "recoverable": torch.stack([b["recoverable"] for b in batch]),
            "src_len": torch.tensor([b["src_len"] for b in batch])}


class DecoderWithAbstention(nn.Module):
    """Backbone LM + a scalar head predicting P(reconstructable)."""

    def __init__(self, base):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.lm = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        h = self.lm.config.hidden_size
        self.abstain = nn.Sequential(nn.Linear(h, h // 4), nn.GELU(),
                                     nn.Linear(h // 4, 1)).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, src_len):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                      labels=labels, output_hidden_states=True)
        hs = out.hidden_states[-1]
        # pool the prompt only: the decision must be made before generating
        idx = (src_len - 1).clamp(min=0).to(hs.device)
        pooled = hs[torch.arange(hs.size(0), device=hs.device), idx]
        return out.loss, self.abstain(pooled).squeeze(-1).float()


def main(a):
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from accelerate import Accelerator

    acc = Accelerator(gradient_accumulation_steps=a.grad_accum)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tr = Pairs(a.data, tok, a.max_len, "train")
    va = Pairs(a.data, tok, a.max_len, "val")
    acc.print(f"train {len(tr):,}  val {len(va):,}")

    f = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=f, num_workers=4)
    dl_va = DataLoader(va, batch_size=a.bs, shuffle=False, collate_fn=f, num_workers=2)

    model = DecoderWithAbstention(a.base)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(dl_tr) / a.grad_accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    model, opt, dl_tr, dl_va, sched = acc.prepare(model, opt, dl_tr, dl_va, sched)

    for ep in range(a.epochs):
        model.train()
        for i, b in enumerate(dl_tr):
            with acc.accumulate(model):
                recon, logit = model(b["input_ids"], b["attention_mask"],
                                     b["labels"], b["src_len"])
                # asymmetric: a confident wrong reconstruction costs far more
                # than an abstention, so weight the positive class down.
                w = torch.where(b["recoverable"] > 0.5, 1.0, a.abstain_weight)
                cal = (F.binary_cross_entropy_with_logits(
                    logit, b["recoverable"], reduction="none") * w).mean()
                loss = recon + a.lambda_cal * cal
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if i % a.log_every == 0:
                acc.print(f"ep{ep} step{i}/{len(dl_tr)} recon={recon.item():.3f} "
                          f"cal={cal.item():.3f}")

        model.eval(); tp = fp = fn = tn = 0
        with torch.no_grad():
            for b in dl_va:
                _, logit = model(b["input_ids"], b["attention_mask"],
                                 b["labels"], b["src_len"])
                pred = (torch.sigmoid(logit) > 0.5).float()
                y = b["recoverable"]
                tp += ((pred == 1) & (y == 1)).sum().item()
                fp += ((pred == 1) & (y == 0)).sum().item()
                fn += ((pred == 0) & (y == 1)).sum().item()
                tn += ((pred == 0) & (y == 0)).sum().item()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        acc.print(f"[ep{ep}] abstention gate -- precision {prec:.3f} recall {rec:.3f} | "
                  f"CONFABULATIONS (emitted on unrecoverable): {fp}")

        acc.wait_for_everyone()
        if acc.is_main_process:
            d = Path(a.out) / f"ep{ep}"; d.mkdir(parents=True, exist_ok=True)
            acc.unwrap_model(model).lm.save_pretrained(d); tok.save_pretrained(d)
            torch.save(acc.unwrap_model(model).abstain.state_dict(), d / "abstain.pt")
            acc.print(f"  saved -> {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data_gen/pairs.jsonl")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="ckpt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lambda-cal", type=float, default=1.0)
    p.add_argument("--abstain-weight", type=float, default=3.0,
                   help="cost multiplier on confabulation vs missed reconstruction")
    p.add_argument("--log-every", type=int, default=50)
    main(p.parse_args())
