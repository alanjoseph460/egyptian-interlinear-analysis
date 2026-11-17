"""Several per-word analysers trained one after another in one job, scored alone and as an ensemble
that decodes greedily on the sum of the members' log-probabilities.

MEMBERS: comma list of model:lr:epochs:seed:budget. Other settings as in e02_word.py (HOLD, STRIP,
NOISE, LEMMA_FIRST ...), shared by all members."""
import os
import random
import sys
import time

import torch
from transformers import T5Config
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, ".")
sys.path.insert(0, "exp")
import common
import e02_word as E

MEMBERS = os.environ.get("MEMBERS", "google/byt5-base:3e-4:10:42:45000,google/byt5-small:5e-4:12:42:40000")
OUT = os.environ.get("ERIS_OUT", "cache/e03")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"


def fit_member(spec, fit_items, pool):
    model, lr, epochs, seed, budget = spec.split(":")
    lr, epochs, seed, budget = float(lr), int(epochs), int(seed), int(budget)
    E.MODEL, E.cfg = model, T5Config.from_pretrained(model)
    E.N_ENC, E.N_DEC = E.cfg.num_layers, E.cfg.num_decoder_layers
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = E.Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=E.WD)
    rng = random.Random(seed)
    nrng = random.Random(seed + 1) if E.NOISE else None
    plan = [E.batches(fit_items, budget, True, rng) for _ in range(epochs)]
    steps = sum(len(p) for p in plan)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        tot = 0.0
        for b in plan[epoch]:
            src, idx_b, idx_t, mmask, own, labels = E.collate([fit_items[j] for j in b], nrng, pool)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev),
                           None if own is None else own.to(dev), labels.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"{spec} epoch {epoch + 1}: loss {tot / len(plan[epoch]):.4f}", flush=True)
    print(f"{spec} trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    del opt
    torch.cuda.empty_cache()
    return net.eval()


@torch.no_grad()
def joint_predict(nets, items):
    """Greedy decoding on the summed log-probabilities of the members, with their caches."""
    preds = [None] * len(items)
    for b in E.batches(items, 512, False, None):
        src, idx_b, idx_t, mmask, own, _ = E.collate([items[j] for j in b])
        src, idx_b, idx_t, mmask = src.to(dev), idx_b.to(dev), idx_t.to(dev), mmask.to(dev)
        own = None if own is None else own.to(dev)
        W = idx_b.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mems = [BaseModelOutput(last_hidden_state=n.memory(src, idx_b, idx_t, own)) for n in nets]
            past = [None] * len(nets)
            last = torch.zeros(W, 1, dtype=torch.long, device=dev)
            done = torch.zeros(W, dtype=torch.bool, device=dev)
            outs = []
            for _ in range(E.MAX_TGT):
                lp = 0
                for k, n in enumerate(nets):
                    o = n.t5(encoder_outputs=mems[k], attention_mask=mmask.long(), decoder_input_ids=last,
                             past_key_values=past[k], use_cache=True)
                    past[k] = o.past_key_values
                    lp = lp + torch.log_softmax(o.logits[:, -1].float(), -1)
                nxt = lp.argmax(-1)
                nxt[done] = E.EOS
                outs.append(nxt)
                done |= nxt == E.EOS
                last = nxt.unsqueeze(1)
                if bool(done.all()):
                    break
        out = torch.stack(outs, 1).tolist()
        j = 0
        for i in b:
            n = len(items[i][0])
            preds[i] = " ".join(E.repair(E.dec_bytes(o)) for o in out[j:j + n])
            j += n
    return preds


def main():
    train = common.load("train")
    test = common.load("test")
    hm = common.held_mask(train, E.HOLD)
    fit = [r for r, h in zip(train, hm) if not h]
    held = [r for r, h in zip(train, hm) if h]
    cap = int(os.environ.get("CAP", 0))  # smoke runs only
    if cap:
        fit, held, test = fit[:cap], held[:cap], test[:cap]
    print(f"members {MEMBERS} hold {E.HOLD} fit {len(fit)} held {len(held)} strip {E.STRIP} noise {E.NOISE} "
          f"lemma_first {E.LEMMA_FIRST}", flush=True)
    fit_items, held_items, test_items = E.prep(fit), E.prep(held), E.prep(test)
    pool = [list(s) for s in sorted({tuple(s) for it in fit_items for w in it[0] for s in w})]
    seen = {w for r in fit for w in r["words"]}
    t_start = time.time()
    nets = []
    for spec in MEMBERS.split(","):
        nets.append(fit_member(spec, fit_items, pool))
        if held:
            t1 = time.time()
            hp = joint_predict(nets[-1:], held_items)
            sc, txt = E.report(f"single {spec}", held, hp, seen)
            print(txt + f" | decode {time.time() - t1:.0f}s", flush=True)
            common.save_preds(OUT, f"held_m{len(nets)}", [r["id"] for r in held], hp, sc)
            if len(nets) == 1:
                # the cached loop must agree with the library's own greedy generate
                ref = E.predict(nets[0], held_items)
                same = sum(x == y for x, y in zip(ref, hp))
                print(f"joint decode equals generate on {same} of {len(hp)} sentences", flush=True)
            if len(nets) > 1:
                t1 = time.time()
                hp = joint_predict(nets, held_items)
                sc, txt = E.report(f"ensemble of {len(nets)}", held, hp, seen)
                print(txt + f" | decode {time.time() - t1:.0f}s", flush=True)
                common.save_preds(OUT, f"held_ens{len(nets)}", [r["id"] for r in held], hp, sc)
    t1 = time.time()
    tp = joint_predict(nets, test_items)
    common.save_preds(OUT, "test", [r["id"] for r in test], tp)
    print(f"test decode {time.time() - t1:.0f}s | total {(time.time() - t_start) / 60:.1f} min | "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
