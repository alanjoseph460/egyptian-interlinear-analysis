"""Second error view: what kind of transliteration mistakes, and which gloss confusions.
python exp/err2.py cloud/<job>/out/held.json [hold]"""
import collections
import json
import re
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
fit_tl = {g.split(";")[0] for r in fit for g in r["groups"]}
fit_lem = {g.split(";")[1] for r in fit for g in r["groups"]}
skel = lambda s: re.sub(r"[().=\-]|PL|DU", "", s)

kind = collections.Counter()
gl = collections.Counter()
pos = collections.Counter()
n_err = collections.Counter()
sent_err = collections.Counter()
ed = collections.defaultdict(list)
for i, p in zip(d["ids"], d["pred"]):
    r = held[i]
    wrong = 0
    for w, g, q in zip(r["words"], r["groups"], p.split(" ")):
        gf, qf = g.split(";"), q.split(";")
        k = "seen" if w in seen else "unseen"
        if gf[0] != qf[0]:
            wrong += 1
            e = metric.levenshtein(gf[0], qf[0])
            ed[k].append(e / max(1, len(gf[0])))
            if skel(gf[0]) == skel(qf[0]):
                kind[k, "marks only (brackets, dots, PL)"] += 1
            elif gf[0].lower() == qf[0].lower():
                kind[k, "case only"] += 1
            elif e <= 1:
                kind[k, "one character"] += 1
            elif qf[0] in fit_tl:
                kind[k, "another known transliteration"] += 1
            else:
                kind[k, "new wrong string"] += 1
            kind[k, "true translit exists in fit: " + str(gf[0] in fit_tl)] += 1
        n_err[k] += 1
        if gf[3] != qf[3] and k == "seen":
            gl[gf[3], qf[3]] += 1
        if gf[2] != qf[2] and k == "seen":
            pos[gf[2], qf[2]] += 1
    sent_err[min(wrong, 5)] += 1
print("words", dict(n_err))
for k, v in sorted(kind.items()):
    print(f"{k[0]:7s} {k[1]:45s} {v:5d}")
for k, v in ed.items():
    v.sort()
    print(k, "relative edit distance of wrong transliterations: median %.2f mean %.2f" % (v[len(v) // 2], sum(v) / len(v)))
print("sentences by number of wrong transliterations:", sorted(sent_err.items()))
print("top gloss confusions on seen spellings (true, predicted):")
for (a, b), v in gl.most_common(25):
    print(f"  {v:4d}  {a}  ->  {b}")
print("top UPOS confusions on seen spellings:")
for (a, b), v in pos.most_common(10):
    print(f"  {v:4d}  {a} -> {b}")
