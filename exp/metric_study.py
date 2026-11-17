"""Metric study: baselines, oracles, readings and T sensitivity on the latest-period holdout.

Run from the repo root. Also writes working/toy_submission.csv (lookup baseline over test.csv).
"""
import csv
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import metric as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PUB = os.path.join(ROOT, "dataset", "public")


def read(name):
    with open(os.path.join(PUB, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def century(s):
    m = re.match(r"\s*(\d+)\D+century\s+(BCE|CE)", s)
    if not m:
        raise ValueError(s)
    return -int(m.group(1)) if m.group(2) == "BCE" else int(m.group(1))


def words(row):
    return [tuple(w.split(";")) for w in row["prediction"].split(" ")]


def build_lookup(rows):
    by_sp = defaultdict(Counter)
    allc = Counter()
    for r in rows:
        for h, w in zip(r["hieroglyphs"].split(" "), r["prediction"].split(" ")):
            by_sp[h][w] += 1
            allc[w] += 1
    table = {h: tuple(c.most_common(1)[0][0].split(";")) for h, c in by_sp.items()}
    return table, tuple(allc.most_common(1)[0][0].split(";"))


def join(ws):
    return " ".join(";".join(w) for w in ws)


def show(name, parts, extra=""):
    print(f"{name:<58} s {parts['s']:.4f}  T {parts['T']:.3f}  L {parts['L']:.3f}  "
          f"P {parts['P']:.3f}  G {parts['G']:.3f} {extra}")


def main():
    train, test = read("train.csv"), read("test.csv")

    # self checks of the metric
    rng = random.Random(0)
    for _ in range(3000):
        a = "".join(rng.choice("abcꜣḥ. ") for _ in range(rng.randint(0, 70)))
        b = "".join(rng.choice("abcꜣḥ. ") for _ in range(rng.randint(0, 70)))
        assert M.levenshtein(a, b) == M.levenshtein_dp(a, b), (a, b)
    for ms in (False, True):
        assert abs(M.gloss_f1("-3sg.m", "-3sg.f", ms) - 0.5) < 1e-12
        assert abs(M.gloss_f1("V\\tam.act", "V\\tam.act:stpr", ms) - 6 / 7) < 1e-12
    assert M.features("V\\tam.act:stpr") == ["V", "tam", "act", "stpr"]
    print("self checks ok: levenshtein == dp on 3000 pairs; gloss anchors 0.5 and 0.857 under "
          "set and multiset")

    cent = [century(r["latest_century"]) for r in train]
    hold = [r for r, c in zip(train, cent) if c >= -19]
    rest = [r for r, c in zip(train, cent) if c < -19]
    print(f"holdout (19th century BCE or later) {len(hold)} sentences, lookup source {len(rest)}")
    print("holdout centuries", Counter(r["latest_century"] for r in hold).most_common())

    table, default = build_lookup(rest)
    rest_lemmas = {w[1] for r in rest for w in words(r)}
    true_w = [words(r) for r in hold]
    hier = [r["hieroglyphs"].split(" ") for r in hold]
    seen = [[h in table for h in hs] for hs in hier]
    look_w = [[table.get(h, default) for h in hs] for hs in hier]
    y_true = [r["prediction"] for r in hold]
    nw = sum(len(x) for x in true_w)
    n_unseen = sum(not f for fs in seen for f in fs)
    n_newlem = sum(w[1] not in rest_lemmas for ws in true_w for w in ws)
    print(f"holdout words {nw}, unseen spelling {n_unseen / nw:.3f} (test page 0.335), "
          f"unseen lemma {n_newlem / nw:.3f} (test page 0.097), default {';'.join(default)}")

    def mix(fn):
        return [join([fn(t, l, s) for t, l, s in zip(tw, lw, ss)])
                for tw, lw, ss in zip(true_w, look_w, seen)]

    def sc(pred, **kw):
        return M.score_parts(y_true, pred, **kw)

    print("\n== baselines (published test anchors in brackets) ==")
    const = mix(lambda t, l, s: default)
    show("constant most frequent analysis", sc(const), "[0.0903: 0.021 0.029 0.200 0.179]")
    look = mix(lambda t, l, s: l)
    t0 = time.perf_counter()
    lp = sc(look)
    dt = time.perf_counter() - t0
    show("word lookup", lp, "[0.4238: 0.191 0.579 0.574 0.583]")
    print(f"   scoring {len(hold)} sentences took {dt:.2f} s")
    rows = [M.score_row(t, t) for t in y_true]
    assert all(r == (1.0, 1.0, 1.0, 1.0, 1.0) for r in rows), "perfect is not exactly 1.0"
    show("perfect (asserted 1.0 on every row)", sc(y_true))

    print("\n== oracles ==")
    show("perfect T, other layers empty", sc(mix(lambda t, l, s: (t[0], "", "", ""))), "[0.4]")
    show("perfect T, lookup L P G", sc(mix(lambda t, l, s: (t[0],) + l[1:])), "[0.747]")
    show("lookup T, perfect L", sc(mix(lambda t, l, s: (l[0], t[1], l[2], l[3]))))
    show("lookup T, perfect P", sc(mix(lambda t, l, s: (l[0], l[1], t[2], l[3]))))
    show("lookup T, perfect G", sc(mix(lambda t, l, s: (l[0], l[1], l[2], t[3]))))
    show("lookup T, perfect L P G", sc(mix(lambda t, l, s: (l[0],) + t[1:])))
    show("lookup on seen, perfect on unseen spellings", sc(mix(lambda t, l, s: l if s else t)))
    show("perfect on seen, lookup on unseen spellings", sc(mix(lambda t, l, s: t if s else l)))
    show("lookup, perfect transliteration on unseen only",
         sc(mix(lambda t, l, s: l if s else (t[0],) + l[1:])))
    show("lookup, perfect transliteration on seen only",
         sc(mix(lambda t, l, s: (t[0],) + l[1:] if s else l)))
    show("lookup, perfect L P G on unseen only",
         sc(mix(lambda t, l, s: l if s else (l[0],) + t[1:])))
    show("lookup, perfect L P G on seen only",
         sc(mix(lambda t, l, s: (l[0],) + t[1:] if s else l)))

    # word accuracy of the lookup per layer, by seen or unseen spelling
    acc = {True: [0, 0, 0, 0, 0], False: [0, 0, 0, 0, 0]}
    for tw, lw, ss in zip(true_w, look_w, seen):
        for t, l, s in zip(tw, lw, ss):
            acc[s][4] += 1
            for k in range(4):
                acc[s][k] += t[k] == l[k]
    for s in (True, False):
        a = acc[s]
        print(f"   lookup word accuracy on {'seen' if s else 'unseen'} spellings ({a[4]} words): "
              f"translit {a[0] / a[4]:.3f} lemma {a[1] / a[4]:.3f} upos {a[2] / a[4]:.3f} "
              f"gloss {a[3] / a[4]:.3f}")

    print("\n== readings: s of each reading on test-like inputs ==")
    dup = sum(len(set(M.features(w[3]))) != len(M.features(w[3])) for r in train for w in words(r))
    nof = sum(not M.features(w[3]) for r in train for w in words(r))
    notl = sum(not " ".join(w[0] for w in words(r)) for r in train)
    emp = [sum(w[k] == "" for r in train for w in words(r)) for k in range(4)]
    print(f"train words with a repeated gloss feature {dup}, with no gloss feature {nof}, "
          f"sentences with an empty transliteration string {notl}, empty fields per layer {emp}")
    # a messy set: lookup output with a malformed word, a doubled space, a tab, a trailing space
    messy = []
    for i, p in enumerate(look):
        ws = p.split(" ")
        k = len(ws) // 2
        if i % 4 == 0:
            ws[k] = ws[k].rsplit(";", 1)[0]
            p = " ".join(ws)
        elif i % 4 == 1:
            p = " ".join(ws[:k]) + "  " + " ".join(ws[k:]) if k else p + " "
        elif i % 4 == 2:
            p = " ".join(ws[:k]) + "\t" + " ".join(ws[k:]) if k else p + "\n"
        messy.append(p)
    variants = [("default", {}), ("multiset gloss", {"multiset": True}),
                ("both gloss sets empty scores 0", {"empty_f1": 0.0}),
                ("empty true transliteration: exact only", {"empty_true": "exact"}),
                ("malformed word dropped from the join", {"keep_malformed": False}),
                ("split on any whitespace", {"any_ws": True})]
    empty_layers = mix(lambda t, l, s: (t[0], "", "", ""))
    for name, kw in variants:
        print(f"   {name:<40} lookup {sc(look, **kw)['s']:.5f}  constant {sc(const, **kw)['s']:.5f}"
              f"  perfect {sc(y_true, **kw)['s']:.5f}  perfectT+empty {sc(empty_layers, **kw)['s']:.5f}"
              f"  messy lookup {sc(messy, **kw)['s']:.5f}")

    print("\n== sensitivity of T (truth corrupted, L P G kept perfect unless stated) ==")
    rng = random.Random(1)

    def tl_edit(fn):
        out = []
        for tw in true_w:
            tl = fn([w[0] for w in tw])
            out.append(join([(a,) + w[1:] for a, w in zip(tl, tw)]))
        return out

    def one_sub(tl):
        tl = list(tl)
        k = rng.randrange(len(tl))
        w = tl[k]
        j = rng.randrange(len(w)) if w else 0
        tl[k] = w[:j] + "#" + w[j + 1:]
        return tl

    def one_del(tl):
        tl = list(tl)
        k = rng.randrange(len(tl))
        w = tl[k]
        j = rng.randrange(len(w)) if w else 0
        tl[k] = w[:j] + w[j + 1:]
        return tl

    def per_word(fn):
        return lambda tl: [fn(w) for w in tl]

    base = sc(y_true)["s"]
    tests = [
        ("one substituted character per sentence", one_sub),
        ("one deleted character per sentence", one_del),
        ("every '.' ending mark dropped", per_word(lambda w: w.replace(".", ""))),
        ("brackets dropped, letters kept: (x) -> x", per_word(lambda w: w.replace("(", "").replace(")", ""))),
        ("bracketed letters dropped: (x) -> ''", per_word(lambda w: re.sub(r"\([^)]*\)", "", w))),
        ("'=' dropped", per_word(lambda w: w.replace("=", ""))),
        ("'-' dropped", per_word(lambda w: w.replace("-", ""))),
        ("combining marks dropped (i with breve below -> i)",
         per_word(lambda w: "".join(c for c in w if not unicodedata.combining(c)))),
        ("lower-cased", per_word(lambda w: w.lower())),
        ("last character of every word dropped", per_word(lambda w: w[:-1])),
    ]
    for name, fn in tests:
        p = sc(tl_edit(fn))
        print(f"   {name:<52} T {p['T']:.3f}  s lost {base - p['s']:.4f}")
    p = sc(mix(lambda t, l, s: (t[1],) + t[1:]))
    print(f"   {'transliteration := true lemma':<52} T {p['T']:.3f}  s lost {base - p['s']:.4f}")
    same = sum(w[0] == w[1] for tw in true_w for w in tw) / nw
    print(f"   share of words whose transliteration equals the lemma {same:.3f}")

    print("\n== word count errors (truth, one word touched) ==")

    def drop_mid(ws):
        k = len(ws) // 2
        return ws[:k] + ws[k + 1:] if len(ws) > 1 else ws

    def blank_mid(ws):
        k = len(ws) // 2
        return ws[:k] + [("", "", "", "")] + ws[k + 1:]

    def dup_mid(ws):
        k = len(ws) // 2
        return ws[:k] + [ws[k]] + ws[k:]

    multi = [i for i, tw in enumerate(true_w) if len(tw) > 1]
    yt_m = [y_true[i] for i in multi]
    for name, fn in [("middle word missing, later words shift left", drop_mid),
                     ("middle word left as an empty slot (;;;)", blank_mid),
                     ("middle word emitted twice, later words shift right", dup_mid),
                     ("middle word twice, then cut to n_words", lambda ws: dup_mid(ws)[:len(ws)]),
                     ("last word missing", lambda ws: ws[:-1]),
                     ("one extra copy of the last word", lambda ws: ws + ws[-1:])]:
        p = M.score_parts(yt_m, [join(fn(true_w[i])) for i in multi])
        show("   " + name, p)
    print(f"   (on the {len(multi)} holdout sentences with 2 or more words)")

    print("\n== unsure words: what to write as the transliteration of an unseen spelling ==")
    med = sorted(len(w[0]) for r in rest for w in words(r))[sum(len(words(r)) for r in rest) // 2]
    opts = [("lookup default " + default[0], lambda t, l, s: l),
            ("empty", lambda t, l, s: l if s else ("",) + l[1:]),
            ("one wrong character", lambda t, l, s: l if s else ("#",) + l[1:]),
            (f"{med} wrong characters (train median word length)",
             lambda t, l, s: l if s else ("#" * med,) + l[1:]),
            ("wrong characters, true length", lambda t, l, s: l if s else ("#" * len(t[0]),) + l[1:]),
            ("first half of the true word", lambda t, l, s: l if s else (t[0][:(len(t[0]) + 1) // 2],) + l[1:]),
            ("true word plus 2 wrong characters", lambda t, l, s: l if s else (t[0] + "##",) + l[1:]),
            ("true word", lambda t, l, s: l if s else (t[0],) + l[1:])]
    for name, fn in opts:
        p = sc(mix(fn))
        print(f"   {name:<52} T {p['T']:.3f}  s {p['s']:.4f}")

    print("\n== by sentence length (holdout) ==")
    lens = [len(tw) for tw in true_w]
    chars = [len(" ".join(w[0] for w in tw)) for tw in true_w]
    lrows = [M.score_row(p, t) for p, t in zip(look, y_true)]
    subs = tl_edit(one_sub)
    srows = [M.score_row(p, t) for p, t in zip(subs, y_true)]
    tot_words = sum(lens)
    for name, lo, hi in [("1", 1, 1), ("2 to 3", 2, 3), ("4 to 8", 4, 8), ("9+", 9, 10 ** 6)]:
        ix = [i for i, n in enumerate(lens) if lo <= n <= hi]
        if not ix:
            continue
        m = [sum(lrows[i][k] for i in ix) / len(ix) for k in range(5)]
        exact = sum(lrows[i][1] == 1.0 for i in ix) / len(ix)
        print(f"   n_words {name:<7} sentences {len(ix) / len(lens):.3f}  words "
              f"{sum(lens[i] for i in ix) / tot_words:.3f}  mean chars "
              f"{sum(chars[i] for i in ix) / len(ix):5.1f}  lookup s {m[0]:.3f} T {m[1]:.3f} "
              f"L {m[2]:.3f} P {m[3]:.3f} G {m[4]:.3f}  lookup T exact {exact:.3f}  "
              f"T after one wrong char {sum(srows[i][1] for i in ix) / len(ix):.3f}")
    tvals = [r[1] for r in lrows]
    print("   lookup T distribution: exact %.3f, T<0.05 %.3f, between %.3f" % (
        sum(t == 1.0 for t in tvals) / len(tvals), sum(t < 0.05 for t in tvals) / len(tvals),
        sum(0.05 <= t < 1.0 for t in tvals) / len(tvals)))
    cers = [M.cer(" ".join(w[0] for w in lw), " ".join(w[0] for w in tw))
            for lw, tw in zip(look_w, true_w)]
    mc = sum(cers) / len(cers)
    import math
    print(f"   lookup mean CER {mc:.3f}: exp(-mean CER/0.2) {math.exp(-mc / 0.2):.3f} against "
          f"mean T {sum(tvals) / len(tvals):.3f}")
    tn = Counter()
    for r in test:
        n = int(r["n_words"])
        tn["1" if n == 1 else "2 to 3" if n <= 3 else "4 to 8" if n <= 8 else "9+"] += 1
    print("   test.csv sentences by n_words bucket:",
          {k: round(v / len(test), 3) for k, v in sorted(tn.items())})

    # toy submission: lookup from all of train over test.csv
    table_all, default_all = build_lookup(train)
    out = os.path.join(ROOT, "working", "toy_submission.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "prediction"])
        for r in test:
            hs = r["hieroglyphs"].split(" ")
            assert len(hs) == int(r["n_words"])
            w.writerow([r["id"], join([table_all.get(h, default_all) for h in hs])])
    print(f"\nwrote {out}: {len(test)} rows, default {';'.join(default_all)}")
    samp = read("sample_submission.csv")
    assert [r["id"] for r in samp] == [r["id"] for r in test]
    assert all(r["prediction"] == " ".join([";".join(default_all)] * int(t["n_words"]))
               for r, t in zip(samp, test)), "sample submission is not the constant default"
    print("sample_submission.csv equals the constant default once per word: confirmed")


if __name__ == "__main__":
    main()
