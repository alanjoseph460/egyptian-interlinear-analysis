"""Per-word ByT5 analyser, second version of e01 with the speed and augmentation options.

The encoder reads the sentence as bytes once; for every word the decoder cross-attends to that
word's encoder states and writes `transliteration;lemma;UPOS;gloss`.

Environment: MODEL, EPOCHS, LR, BS (position-layers per batch), SEED, HOLD (page, near, f2, none),
BEAMS (extra beam decode at the end), CTX (neighbour words visible to the decoder), STRIP (drop the
two constant lead bytes of every hieroglyph), DCH (decoder chunks by target length), NOISE (share
of training words that get one random sign edit), OPT (adamw or adafactor), WD."""
import os
import random
import re
import sys
import time

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, ".")
import common

MODEL = os.environ.get("MODEL", "google/byt5-small")
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = float(os.environ.get("LR", 5e-4))
BS = int(os.environ.get("BS", 80000))
SEED = int(os.environ.get("SEED", 42))
HOLD = os.environ.get("HOLD", "page")
BEAMS = int(os.environ.get("BEAMS", 1))
CTX = int(os.environ.get("CTX", 0))
STRIP = int(os.environ.get("STRIP", 1))
DCH = int(os.environ.get("DCH", 3))
NOISE = float(os.environ.get("NOISE", 0))
OPT = os.environ.get("OPT", "adamw")
WD = float(os.environ.get("WD", 0.01))
LEMMA_FIRST = int(os.environ.get("LEMMA_FIRST", 0))  # write lemma before transliteration
DEL_ONLY = int(os.environ.get("DEL_ONLY", 0))  # noise deletes signs only
LS = float(os.environ.get("LS", 0))  # label smoothing
DROPOUT = float(os.environ.get("DROPOUT", 0.1))  # T5 default is 0.1
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 3))
MAX_TGT = 128
OUT = os.environ.get("ERIS_OUT", "cache/e02")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
cfg = T5Config.from_pretrained(MODEL)
N_ENC, N_DEC = cfg.num_layers, cfg.num_decoder_layers

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PAD, EOS, SPACE = 0, 1, 35
SIGN = re.compile(r"<g>.*?</g>|.", re.S)
LEAD = b"\xf0\x93"  # every sign of the Egyptian Hieroglyphs block starts with these two bytes


def enc_bytes(s):
    return [b + 3 for b in s.encode("utf-8")]


def sign_ids(sign):
    b = sign.encode("utf-8")
    if STRIP and len(b) == 4 and b[:2] == LEAD:
        b = b[2:]
    return [x + 3 for x in b]


def dec_bytes(ids):
    out = bytearray()
    for i in ids:
        if i == EOS:
            break
        if 3 <= i < 259:
            out.append(i - 3)
    return out.decode("utf-8", errors="ignore")


def swap(g):
    """Exchange the first two fields when the lemma is written first."""
    if not LEMMA_FIRST:
        return g
    f = g.split(";")
    return ";".join([f[1], f[0]] + f[2:]) if len(f) >= 2 else g


def prep(rows):
    """Per sentence: for each word its signs as id lists, and one target per word."""
    items = []
    for r in rows:
        words = [[sign_ids(s) for s in SIGN.findall(w)] for w in r["words"]]
        tgt = [enc_bytes(swap(g))[:MAX_TGT - 1] + [EOS] for g in r["groups"]] if "groups" in r else None
        items.append((words, tgt, sum(len(s) for w in words for s in w) + len(words)))
    return items


def noisy(words, rng, pool):
    """One random sign edit in a share of the words: delete, substitute or insert a sign."""
    out = []
    for w in words:
        if rng.random() < NOISE:
            w = list(w)
            k = rng.randrange(len(w))
            op = rng.random()
            if (op < 0.4 or DEL_ONLY) and len(w) > 1:
                del w[k]
            elif DEL_ONLY:
                pass
            elif op < 0.8:
                w[k] = rng.choice(pool)
            else:
                w.insert(k, rng.choice(pool))
        out.append(w)
    return out


def build(words):
    src, spans = [], []
    for k, w in enumerate(words):
        if k:
            src.append(SPACE)
        a = len(src)
        for s in w:
            src += s
        spans.append((a, len(src)))
    src.append(EOS)
    return src, spans


def collate(batch, rng=None, pool=None):
    built = [build(noisy(b[0], rng, pool) if rng else b[0]) for b in batch]
    L = max(len(s) for s, _ in built)
    src = torch.full((len(batch), L), PAD, dtype=torch.long)
    wb, ws, we, own_a, own_e = [], [], [], [], []
    for i, (s, spans) in enumerate(built):
        src[i, :len(s)] = torch.tensor(s)
        n = len(spans)
        for k in range(n):
            a = spans[max(0, k - CTX)][0]
            e = spans[min(n - 1, k + CTX)][1]
            wb.append(i), ws.append(a), we.append(e)
            own_a.append(spans[k][0] - a), own_e.append(spans[k][1] - a)
    W, Lw = len(wb), max(e - a for a, e in zip(ws, we))
    pos = torch.arange(Lw).unsqueeze(0).expand(W, Lw)
    ws_t, we_t = torch.tensor(ws).unsqueeze(1), torch.tensor(we).unsqueeze(1)
    mmask = (pos + ws_t) < we_t
    idx_t = torch.where(mmask, pos + ws_t, torch.zeros_like(pos))
    idx_b = torch.tensor(wb).unsqueeze(1).expand(W, Lw)
    own = None
    if CTX:
        own = ((pos >= torch.tensor(own_a).unsqueeze(1)) & (pos < torch.tensor(own_e).unsqueeze(1))).long()
    labels = None
    if batch[0][1] is not None:
        tg = [t for b in batch for t in b[1]]
        labels = torch.full((W, max(len(t) for t in tg)), -100, dtype=torch.long)
        for j, t in enumerate(tg):
            labels[j, :len(t)] = torch.tensor(t)
    return src, idx_b, idx_t, mmask, own, labels


def batches(items, budget, shuffle, rng):
    """Training: batches under a memory budget in position-layers: sentences x longest source x
    encoder layers + target bytes x 1.3 x decoder layers (the decoder runs in DCH length chunks, so
    its padding is small). Prediction: at most `budget` words."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
        chunks = [order[i:i + 1600] for i in range(0, len(order), 1600)]
        order = [j for c in chunks for j in sorted(c, key=lambda j: items[j][2])]
    else:
        order.sort(key=lambda j: items[j][2])
    out, cur, nw, ms, tt = [], [], 0, 0, 0
    for j in order:
        n, ls = len(items[j][0]), items[j][2] + 2
        lt = sum(len(t) for t in items[j][1]) if shuffle else 0
        ms2 = max(ms, ls)
        cost = (len(cur) + 1) * ms2 * N_ENC + (tt + lt) * 1.3 * N_DEC if shuffle else nw + n
        if cur and cost > budget:
            out.append(cur)
            cur, nw, tt, ms2 = [], 0, 0, ls
        cur.append(j)
        nw, tt, ms = nw + n, tt + lt, ms2
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL, dropout_rate=DROPOUT)
        for m in self.t5.modules():
            if hasattr(m, "act") and hasattr(m, "wi_0"):
                m.act = torch.nn.GELU(approximate="tanh")
        if CTX:
            self.own = torch.nn.Embedding(2, cfg.d_model)
            torch.nn.init.zeros_(self.own.weight)

    def memory(self, src, idx_b, idx_t, own):
        h = self.t5.encoder(input_ids=src, attention_mask=(src != PAD).long()).last_hidden_state
        mem = h[idx_b, idx_t]
        if own is not None:
            mem = mem + self.own(own).to(mem.dtype)
        return mem

    def forward(self, src, idx_b, idx_t, mmask, own, labels):
        mem = self.memory(src, idx_b, idx_t, own)
        lens = (labels != -100).sum(1)
        order = torch.argsort(lens)
        total, ntok = 0.0, 0
        for ch in torch.tensor_split(order, DCH):
            if len(ch) == 0:
                continue
            lt, lw = int(lens[ch].max()), int(mmask[ch].sum(1).max())
            out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem[ch, :lw]),
                          attention_mask=mmask[ch, :lw].long(), labels=labels[ch, :lt])
            loss = out.loss
            if LS:
                loss = torch.nn.functional.cross_entropy(
                    out.logits.float().flatten(0, 1), labels[ch, :lt].flatten(), ignore_index=-100,
                    label_smoothing=LS)
            n = int(lens[ch].sum())
            total, ntok = total + loss * n, ntok + n
        return total / ntok

    @torch.no_grad()
    def generate(self, src, idx_b, idx_t, mmask, own, beams):
        mem = self.memory(src, idx_b, idx_t, own)
        return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                                attention_mask=mmask.long(), max_new_tokens=MAX_TGT,
                                num_beams=beams, do_sample=False)


def repair(s):
    f = s.replace(" ", "").split(";")
    return swap(";".join((f + ["", "", "", ""])[:4]))


def predict(net, items, beams=1):
    net.eval()
    preds = [None] * len(items)
    for b in batches(items, 512 if beams == 1 else 192, False, None):
        src, idx_b, idx_t, mmask, own, _ = collate([items[j] for j in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net.generate(src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev),
                               None if own is None else own.to(dev), beams)
        out = out[:, 1:].tolist()
        j = 0
        for i in b:
            n = len(items[i][0])
            preds[i] = " ".join(repair(dec_bytes(o)) for o in out[j:j + n])
            j += n
    return preds


def report(tag, held, hp, seen):
    sc = common.score_rows(hp, held)
    hit = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}
    for r, p in zip(held, hp):
        for w, g, q in zip(r["words"], r["groups"], p.split(" ")):
            k, gf, qf = w in seen, g.split(";"), q.split(";")
            hit[k][0] += gf[0] == qf[0]
            hit[k][1] += gf[1] == qf[1]
            hit[k][2] += gf == qf
            hit[k][3] += 1
    a, u = hit[True], hit[False]
    return sc, (f" | {tag} s {sc['s']:.4f} T {sc['T']:.3f} L {sc['L']:.3f} P {sc['P']:.3f} G {sc['G']:.3f}"
                f" | seen tl {a[0] / max(1, a[3]):.3f} lem {a[1] / max(1, a[3]):.3f} all {a[2] / max(1, a[3]):.3f}"
                f" | unseen tl {u[0] / max(1, u[3]):.3f} lem {u[1] / max(1, u[3]):.3f} all {u[2] / max(1, u[3]):.3f}")


def main():
    train = common.load("train")
    test = common.load("test")
    hm = common.held_mask(train, HOLD)
    fit = [r for r, h in zip(train, hm) if not h]
    held = [r for r, h in zip(train, hm) if h]
    print(f"model {MODEL} hold {HOLD} fit {len(fit)} held {len(held)} epochs {EPOCHS} lr {LR} bs {BS} "
          f"beams {BEAMS} ctx {CTX} strip {STRIP} dch {DCH} noise {NOISE} opt {OPT} seed {SEED} lemma_first {LEMMA_FIRST} del_only {DEL_ONLY} ls {LS} dropout {DROPOUT}", flush=True)
    fit_items, held_items, test_items = prep(fit), prep(held), prep(test)
    pool = sorted({tuple(s) for it in fit_items for w in it[0] for s in w})
    pool = [list(s) for s in pool]

    net = Net().to(dev)
    if OPT == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(net.parameters(), lr=LR, relative_step=False, scale_parameter=False, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    rng = random.Random(SEED)
    nrng = random.Random(SEED + 1) if NOISE else None
    plan = [batches(fit_items, BS, True, rng) for _ in range(EPOCHS)]
    steps = sum(len(p) for p in plan)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    print(f"steps {steps} ({len(plan[0])} per epoch)", flush=True)
    t_start = time.time()
    seen = {w for r in fit for w in r["words"]}
    sc = None
    for epoch in range(EPOCHS):
        t0 = time.time()
        net.train()
        tot = 0.0
        for b in plan[epoch]:
            src, idx_b, idx_t, mmask, own, labels = collate([fit_items[j] for j in b], nrng, pool)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev),
                           None if own is None else own.to(dev), labels.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"epoch {epoch + 1}: loss {tot / len(plan[epoch]):.4f} train {time.time() - t0:.0f}s"
        if held and ((epoch + 1) % EVAL_EVERY == 0 or epoch + 1 == EPOCHS):
            t1 = time.time()
            hp = predict(net, held_items)
            sc, txt = report("greedy", held, hp, seen)
            msg += txt + f" | decode {time.time() - t1:.0f}s"
        print(msg, flush=True)
    print(f"train total {(time.time() - t_start) / 60:.1f} min | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    if held:
        common.save_preds(OUT, "held", [r["id"] for r in held], hp, sc)
        if BEAMS > 1:
            t1 = time.time()
            hb = predict(net, held_items, BEAMS)
            scb, txt = report(f"beam{BEAMS}", held, hb, seen)
            print(txt + f" | decode {time.time() - t1:.0f}s", flush=True)
            common.save_preds(OUT, f"held_beam{BEAMS}", [r["id"] for r in held], hb, scb)
        with open(f"{OUT}/score.txt", "w") as f:
            f.write(f"{sc['s']:.5f}\n")
    t1 = time.time()
    tp = predict(net, test_items)
    common.save_preds(OUT, "test", [r["id"] for r in test], tp)
    print(f"test decode {time.time() - t1:.0f}s | total {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
