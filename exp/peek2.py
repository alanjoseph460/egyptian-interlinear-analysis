import csv, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
d = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
tr = list(csv.DictReader(open(d + "train.csv", encoding="utf-8")))
te = list(csv.DictReader(open(d + "test.csv", encoding="utf-8")))
for name, rows in (("train", tr), ("test", te)):
    c = collections.Counter(r["latest_century"] for r in rows)
    w = collections.Counter()
    for r in rows: w[r["latest_century"]] += int(r["n_words"])
    print(name, sorted(((k, v, w[k]) for k, v in c.items()), key=lambda x: -int(x[0].split("t")[0].split("n")[0].split("r")[0].split("s")[0])))
import statistics as st
wl = [len(h.encode("utf-8")) for r in tr for h in r["hieroglyphs"].split(" ")]
al = [len(a.encode("utf-8")) for r in tr for a in r["prediction"].split(" ")]
sl = [len(r["hieroglyphs"].encode("utf-8")) for r in tr]
tl = [len(r["hieroglyphs"].encode("utf-8")) for r in te]
q = lambda x, p: sorted(x)[int(len(x) * p) - 1]
for n, x in (("word bytes", wl), ("analysis bytes", al), ("sent bytes train", sl), ("sent bytes test", tl)):
    print(n, len(x), "mean %.1f" % st.mean(x), "p50", q(x, .5), "p95", q(x, .95), "p99", q(x, .99), "p999", q(x, .999), "max", max(x))
bad = sum(1 for r in tr if len(r["hieroglyphs"].split(" ")) != len(r["prediction"].split(" ")) or len(r["hieroglyphs"].split(" ")) != int(r["n_words"]))
print("misaligned rows", bad, "bad groups", sum(1 for r in tr for a in r["prediction"].split(" ") if a.count(";") != 3))
