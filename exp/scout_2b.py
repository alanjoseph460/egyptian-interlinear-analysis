import sys, csv, re, collections
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
D = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
with open(D + "train.csv", encoding="utf-8", newline="") as f:
    tr = list(csv.DictReader(f))
with open(D + "test.csv", encoding="utf-8", newline="") as f:
    te = list(csv.DictReader(f))
C = collections.Counter
cn_row = np.array([int(re.match(r"\d+", r["latest_century"]).group()) for r in tr])
w, lem, upos, gloss, cn, row = [], [], [], [], [], []
for i, r in enumerate(tr):
    for a, g in zip(r["hieroglyphs"].split(" "), r["prediction"].split(" ")):
        f4 = g.split(";")
        w.append(a); lem.append(f4[1]); upos.append(f4[2]); gloss.append(f4[3]); cn.append(cn_row[i]); row.append(i)
w = np.array(w, dtype=object); lem = np.array(lem, dtype=object); cn = np.array(cn); row = np.array(row)
strs = np.array([r["hieroglyphs"] for r in tr], dtype=object)
tew = [a for r in te for a in r["hieroglyphs"].split(" ")]


def rates(held, rest, name):
    ws, ls = set(w[rest]), set(lem[rest])
    a = np.mean([x not in ws for x in w[held]])
    b = np.mean([x not in ls for x in lem[held]])
    hr = np.unique(row[held]); rr = np.unique(row[rest])
    rs = set(strs[rr])
    d = np.mean([s in rs for s in strs[hr]])
    print(f"  {name}: held rows {len(hr)} words {int(held.sum())} (mean {held.sum()/len(hr):.2f}) | train rows {len(rr)} words {int(rest.sum())} | "
          f"unseen spelling {a:.4f} unseen lemma {b:.4f} | held rows with string in train part {d:.4f}")


print("candidate slices, continued (published test: spelling 0.335, lemma 0.097)")
rates(cn <= 19, cn >= 20, "<=19th held, >=20th train (page validation)")
rates(cn <= 18, cn >= 19, "18th+16th held, all other train")
rates(cn == 22, cn > 22, "22nd held, earlier-only")
rates(cn == 22, cn != 22, "22nd held, all other")
rates(cn == 23, cn > 23, "23rd held, earlier-only")
rates(cn == 23, cn != 23, "23rd held, all other")
rates(cn == 24, cn > 24, "24th held, earlier-only")
rates(cn == 24, cn != 24, "24th held, all other")
rates((cn >= 23) & (cn <= 24), (cn < 23) | (cn > 24), "23-24th held, all other")
rates(cn >= 25, cn < 25, ">=25th held, all later train")
rng = np.random.RandomState(0)
rid = rng.rand(len(tr)) < 0.2
rates(rid[row], ~rid[row], "random 20% rows")
for seed in range(3):
    rng = np.random.RandomState(seed)
    half = (cn_row <= 19) & (rng.rand(len(tr)) < 0.5)
    rates(half[row], ~half[row], f"random half of <=19th held, rest incl. other half train (seed {seed})")
    half = (cn_row <= 18) & (rng.rand(len(tr)) < 0.5)
    rates(half[row], ~half[row], f"random half of 18th+16th held, rest incl. other half train (seed {seed})")

print("\nper-century unseen spelling vs all other centuries / vs strictly earlier")
for c in sorted(set(cn.tolist())):
    h = w[cn == c]
    o = set(w[cn != c]); e = set(w[cn > c])
    print(f"  {c}: words {len(h)} vs-other {np.mean([x not in o for x in h]):.3f} vs-earlier {np.mean([x not in e for x in h]):.3f}")

early, late, c18, c19 = set(w[cn >= 20]), set(w[cn <= 19]), set(w[cn == 18]), set(w[cn == 19])
allw = set(w)
print("\ntest words unseen in: all train", round(float(np.mean([x not in allw for x in tew])), 4),
      "| train >=20th only", round(float(np.mean([x not in early for x in tew])), 4),
      "| train <=19th only", round(float(np.mean([x not in late for x in tew])), 4),
      "| 18th only", round(float(np.mean([x not in c18 for x in tew])), 4),
      "| 19th only", round(float(np.mean([x not in c19 for x in tew])), 4))
print("test words seen ONLY thanks to the <=19th train rows", round(float(np.mean([(x in late) and (x not in early) for x in tew])), 4))

upos = np.array(upos, dtype=object); gloss = np.array(gloss, dtype=object)
for nm, m in (("<=19th", cn <= 19), (">=20th", cn >= 20)):
    cu = C(upos[m]); n = m.sum()
    print(nm, "UPOS share", {k: round(v / n, 3) for k, v in cu.most_common()})
    cg = C(gloss[m])
    print(nm, "top gloss", {k: round(v / n, 3) for k, v in cg.most_common(12)})
g_l, g_e = set(gloss[cn <= 19]), set(gloss[cn >= 20])
print("glosses in <=19th not in >=20th: distinct", len(g_l - g_e), "tokens share", round(float(np.mean([x not in g_e for x in gloss[cn <= 19]])), 4))
print("1-word sentences share: train", round(float(np.mean([r['n_words'] == '1' for r in tr])), 4), "train <=19th",
      round(float(np.mean([tr[i]['n_words'] == '1' for i in np.where(cn_row <= 19)[0]])), 4), "test", round(float(np.mean([r['n_words'] == '1' for r in te])), 4))
for nm, idx in (("<=19th", np.where(cn_row <= 19)[0]), (">=20th", np.where(cn_row >= 20)[0])):
    nw = np.array([int(tr[i]["n_words"]) for i in idx])
    print(nm, "n_words mean", round(float(nw.mean()), 2), "p50", np.percentile(nw, 50), "p90", np.percentile(nw, 90), "p99", np.percentile(nw, 99), "max", nw.max())
