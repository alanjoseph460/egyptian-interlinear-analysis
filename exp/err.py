"""Error analysis of a held-out prediction file: python exp/err.py cloud/<job>/out/held.json [hold]"""
import collections
import json
import random
import sys

sys.path.insert(0, ".")
import common
import metric

sys.stdout.reconfigure(encoding="utf-8")
path, hold = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "page")
train = common.load("train")
hm = common.held_mask(train, hold)
fit = [r for r, h in zip(train, hm) if not h]
held = {r["id"]: r for r, h in zip(train, hm) if h}
d = json.load(open(path, encoding="utf-8"))
seen = {w for r in fit for w in r["words"]}
seen_lemma = {g.split(";")[1] for r in fit for g in r["groups"]}

by_len = collections.defaultdict(list)
cat = collections.Counter()
tot = collections.Counter()
examples = []
lost = collections.Counter()
for i, p in zip(d["ids"], d["pred"]):
    r = held[i]
    s, T, L, P, G = metric.score_row(p, r["prediction"])
    n = len(r["words"])
    by_len["1" if n == 1 else "2-3" if n <= 3 else "4-8" if n <= 8 else "9-19" if n <= 19 else "20+"].append((s, T, L, P, G))
    lost["T"] += 0.4 * (1 - T)
    lost["L"] += 0.2 * (1 - L)
    lost["P"] += 0.2 * (1 - P)
    lost["G"] += 0.2 * (1 - G)
    for w, g, q in zip(r["words"], r["groups"], p.split(" ")):
        gf, qf = g.split(";"), q.split(";")
        k = "seen" if w in seen else ("unseen/known lemma" if gf[1] in seen_lemma else "unseen/new lemma")
        tot[k] += 1
        for name, a, b in zip("TLPG", gf, qf):
            cat[k, name] += a == b
        if gf[0] != qf[0]:
            examples.append((k, w, g, q))
n = len(d["ids"])
print("lost points per layer:", {k: round(v / n, 4) for k, v in lost.items()})
for k, v in sorted(by_len.items()):
    m = [sum(x[j] for x in v) / len(v) for j in range(5)]
    print(f"n_words {k:5s} share {len(v) / n:.3f} s {m[0]:.3f} T {m[1]:.3f} L {m[2]:.3f} P {m[3]:.3f} G {m[4]:.3f}")
for k in tot:
    print(f"{k:20s} words {tot[k]:6d} share {tot[k] / sum(tot.values()):.3f} exact T {cat[k, 'T'] / tot[k]:.3f} "
          f"L {cat[k, 'L'] / tot[k]:.3f} P {cat[k, 'P'] / tot[k]:.3f} G {cat[k, 'G'] / tot[k]:.3f}")
random.seed(0)
for k, w, g, q in random.sample(examples, 45):
    print(f"{k[:12]:12s} {w}  true {g}  pred {q}")
