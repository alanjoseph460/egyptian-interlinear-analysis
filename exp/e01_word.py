"""ByT5 fine-tuned as a per-word analyser. The encoder reads the whole sentence as UTF-8 bytes once.
For every word, the decoder cross-attends only to that word's encoder states (already contextual)
and writes `transliteration;lemma;UPOS;gloss`. Outputs are short, parallel over words and aligned
to the given word boundaries by construction.

Settings come from the environment so one file serves the size ladder:
MODEL, EPOCHS, LR, BS (position-layers per batch), LAYERS, SEED, HOLD (page, near, f2, none), BEAMS, CTX."""
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, ".")
import common

MODEL = os.environ.get("MODEL", "google/byt5-small")
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = float(os.environ.get("LR", 5e-4))
BS = int(os.environ.get("BS", 80000))  # position-layers per batch
N_ENC, N_DEC = (int(x) for x in os.environ.get("LAYERS", "12,4").split(","))
SEED = int(os.environ.get("SEED", 42))
HOLD = os.environ.get("HOLD", "page")
BEAMS = int(os.environ.get("BEAMS", 1))
CTX = int(os.environ.get("CTX", 0))  # neighbour words on each side that the decoder may also see
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 3))
MAX_TGT = 128  # bytes; the longest train analysis is 161, p99.9 is 93
OUT = os.environ.get("ERIS_OUT", "cache/e01")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PAD, EOS = 0, 1


def enc_bytes(s):
    return [b + 3 for b in s.encode("utf-8")]  # ByT5 ids: byte value + 3


def dec_bytes(ids):
    out = bytearray()
    for i in ids:
        if i == EOS:
            break
        if 3 <= i < 259:
            out.append(i - 3)
    return out.decode("utf-8", errors="ignore")


def prep(rows):
    """Per sentence: source ids and the byte span of each word."""
    items = []
    for r in rows:
        src, spans = [], []
        for k, w in enumerate(r["words"]):
            if k:
                src.append(32 + 3)
            a = len(src)
            src += enc_bytes(w)
            spans.append((a, len(src)))
        src.append(EOS)
        tgt = [enc_bytes(g)[:MAX_TGT - 1] + [EOS] for g in r["groups"]] if "groups" in r else None
        items.append((src, spans, tgt))
    return items


def collate(batch):
    """Returns source tensors, the gather index that builds one memory row per word, and labels."""
    L = max(len(b[0]) for b in batch)
    src = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, :len(b[0])] = torch.tensor(b[0])
    wb, ws, we = [], [], []
    for i, b in enumerate(batch):
        n = len(b[1])
        for k in range(n):
            a = b[1][max(0, k - CTX)][0]
            e = b[1][min(n - 1, k + CTX)][1]
            wb.append(i), ws.append(a), we.append(e)
    Lw = max(e - a for a, e in zip(ws, we))
    W = len(wb)
    pos = torch.arange(Lw).unsqueeze(0).expand(W, Lw)
    ws_t, we_t = torch.tensor(ws).unsqueeze(1), torch.tensor(we).unsqueeze(1)
    mmask = (pos + ws_t) < we_t
    idx_t = torch.where(mmask, pos + ws_t, torch.zeros_like(pos))
    idx_b = torch.tensor(wb).unsqueeze(1).expand(W, Lw)
    # with a context window the decoder needs to know which bytes are the target word
    own = None
    if CTX:
        own = torch.zeros(W, Lw, dtype=torch.long)
        j = 0
        for i, b in enumerate(batch):
            for k in range(len(b[1])):
                a, e = b[1][k]
                own[j, a - ws[j]:e - ws[j]] = 1
                j += 1
    labels = None
    if batch[0][2] is not None:
        tg = [t for b in batch for t in b[2]]
        Lt = max(len(t) for t in tg)
        labels = torch.full((W, Lt), -100, dtype=torch.long)
        for j, t in enumerate(tg):
            labels[j, :len(t)] = torch.tensor(t)
    return src, idx_b, idx_t, mmask, own, labels


def batches(items, budget, shuffle, rng):
    """Training: batches under a memory budget counted in position-layers, the measured driver
    (about 53 KB each): sentences x longest source x encoder layers + words x longest target x
    decoder layers. Prediction: at most `budget` words. Sentences of similar source length go together."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
        chunks = [order[i:i + 1600] for i in range(0, len(order), 1600)]
        order = [j for c in chunks for j in sorted(c, key=lambda j: len(items[j][0]))]
    else:
        order.sort(key=lambda j: len(items[j][0]))
    out, cur, nw, ms, mt = [], [], 0, 0, 0
    for j in order:
        n, ls = len(items[j][1]), len(items[j][0])
        lt = max(len(t) for t in items[j][2]) if shuffle else 0
        ms2, mt2 = max(ms, ls), max(mt, lt)
        cost = (len(cur) + 1) * ms2 * N_ENC + (nw + n) * mt2 * N_DEC if shuffle else nw + n
        if cur and cost > budget:
            out.append(cur)
            cur, nw, ms2, mt2 = [], 0, ls, lt
        cur.append(j)
        nw, ms, mt = nw + n, ms2, mt2
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL)
        for m in self.t5.modules():
            if hasattr(m, "act") and hasattr(m, "wi_0"):
                m.act = torch.nn.GELU(approximate="tanh")
        if CTX:
            self.own = torch.nn.Embedding(2, self.t5.config.d_model)
            torch.nn.init.zeros_(self.own.weight)

    def memory(self, src, idx_b, idx_t, mmask, own):
        h = self.t5.encoder(input_ids=src, attention_mask=(src != PAD).long()).last_hidden_state
        mem = h[idx_b, idx_t]
        if own is not None:
            mem = mem + self.own(own).to(mem.dtype)
        return mem

    def forward(self, src, idx_b, idx_t, mmask, own, labels):
        mem = self.memory(src, idx_b, idx_t, mmask, own)
        return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                       attention_mask=mmask.long(), labels=labels).loss

    @torch.no_grad()
    def generate(self, src, idx_b, idx_t, mmask, own):
        mem = self.memory(src, idx_b, idx_t, mmask, own)
        return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                                attention_mask=mmask.long(), max_new_tokens=MAX_TGT,
                                num_beams=BEAMS, do_sample=False)


def repair(s):
    """One group, four fields, no spaces: the grader zeroes any other shape."""
    f = s.replace(" ", "").split(";")
    f = (f + ["", "", "", ""])[:4]
    return ";".join(f)


def predict(net, items):
    net.eval()
    preds = [None] * len(items)
    for b in batches(items, 512, False, None):
        src, idx_b, idx_t, mmask, own, _ = collate([items[j] for j in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net.generate(src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev),
                               None if own is None else own.to(dev))
        out = out[:, 1:].tolist()  # drop the decoder start token
        j = 0
        for i in b:
            n = len(items[i][1])
            preds[i] = " ".join(repair(dec_bytes(o)) for o in out[j:j + n])
            j += n
    return preds


def main():
    train = common.load("train")
    test = common.load("test")
    hm = common.held_mask(train, HOLD)
    fit = [r for r, h in zip(train, hm) if not h]
    held = [r for r, h in zip(train, hm) if h]
    print(f"model {MODEL} hold {HOLD} fit {len(fit)} held {len(held)} epochs {EPOCHS} lr {LR} bs {BS} "
          f"beams {BEAMS} ctx {CTX} seed {SEED}", flush=True)
    fit_items, held_items, test_items = prep(fit), prep(held), prep(test)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    steps = EPOCHS * len(batches(fit_items, BS, True, random.Random(0)))
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    t_start = time.time()
    seen = {w for r in fit for w in r["words"]}
    for epoch in range(EPOCHS):
        t0 = time.time()
        net.train()
        tot, nb = 0.0, 0
        for b in batches(fit_items, BS, True, rng):
            src, idx_b, idx_t, mmask, own, labels = collate([fit_items[j] for j in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev),
                           None if own is None else own.to(dev), labels.to(dev))
            if epoch == 0 and nb < 3:
                print("dbg", tuple(src.shape), tuple(idx_b.shape), tuple(labels.shape),
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        msg = f"epoch {epoch + 1}: loss {tot / nb:.4f} train {time.time() - t0:.0f}s"
        if held and ((epoch + 1) % EVAL_EVERY == 0 or epoch + 1 == EPOCHS):
            t1 = time.time()
            hp = predict(net, held_items)
            sc = common.score_rows(hp, held)
            # transliteration accuracy by whether the spelling occurs in the fit part
            hit = {True: [0, 0], False: [0, 0]}
            for r, p in zip(held, hp):
                for w, g, q in zip(r["words"], r["groups"], p.split(" ")):
                    k = w in seen
                    hit[k][0] += g.split(";")[0] == q.split(";")[0]
                    hit[k][1] += 1
            msg += (f" | held s {sc['s']:.4f} T {sc['T']:.3f} L {sc['L']:.3f} P {sc['P']:.3f} G {sc['G']:.3f}"
                    f" | translit acc seen {hit[True][0] / max(1, hit[True][1]):.3f}"
                    f" unseen {hit[False][0] / max(1, hit[False][1]):.3f}"
                    f" (unseen share {hit[False][1] / (hit[False][1] + hit[True][1]):.3f})"
                    f" | decode {time.time() - t1:.0f}s")
        print(msg, flush=True)
    if held:
        common.save_preds(OUT, "held", [r["id"] for r in held], hp, sc)
        with open(f"{OUT}/score.txt", "w") as f:
            f.write(f"{sc['s']:.5f}\n")
    t1 = time.time()
    tp = predict(net, test_items)
    common.save_preds(OUT, "test", [r["id"] for r in test], tp)
    print(f"test decode {time.time() - t1:.0f}s | total {(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
