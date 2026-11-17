"""Local copy of the challenge scoring rule (REQUIREMENTS.md, section Evaluation).

s = 0.4 T + 0.2 L + 0.2 P + 0.2 G per sentence, mean over sentences.
The keyword arguments select between readings the text leaves open; the defaults are the
readings we take as the grader's.
"""
import re
import unicodedata

import numpy as np

MAX_CHARS = 6000
TAU = 0.2
EMPTY = ("", "", "", "")
_FEAT = re.compile(r"[\\.:\-=]")


def parse(pred, any_ws=False):
    """Words of an analysis string as (transliteration, lemma, upos, gloss) tuples."""
    words = pred.split() if any_ws else pred.split(" ")
    out = []
    for w in words:
        f = w.split(";")
        out.append(tuple(f) if len(f) == 4 else EMPTY)
    return out


def _malformed(pred, any_ws=False):
    words = pred.split() if any_ws else pred.split(" ")
    return [w.count(";") != 3 for w in words]


def levenshtein(a, b):
    """Edit distance, bit-parallel (Hyyro 2003) on Python integers."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    m = len(a)
    mask = (1 << m) - 1
    last = 1 << (m - 1)
    vp, vn, dist = mask, 0, m
    get = peq.get
    for ch in b:
        x = get(ch, 0) | vn
        d0 = ((((x & vp) + vp) ^ vp) | x) & mask
        hp = (vn | ~(d0 | vp)) & mask
        hn = d0 & vp
        if hp & last:
            dist += 1
        elif hn & last:
            dist -= 1
        hp = ((hp << 1) | 1) & mask
        hn = (hn << 1) & mask
        vp = (hn | ~(d0 | hp)) & mask
        vn = hp & d0
    return dist


def levenshtein_dp(a, b):
    """Plain row-buffer version, kept as the check for levenshtein()."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def features(gloss):
    return [x for x in _FEAT.split(gloss) if x]


def gloss_f1(pred_gloss, true_gloss, multiset=False, empty_f1=1.0):
    p, t = features(pred_gloss), features(true_gloss)
    if not multiset:
        p, t = set(p), set(t)
    if not p and not t:
        return empty_f1
    if not p or not t:
        return 0.0
    if multiset:
        left = list(t)
        hit = 0
        for x in p:
            if x in left:
                left.remove(x)
                hit += 1
    else:
        hit = len(p & t)
    if hit == 0:
        return 0.0
    return 2.0 * hit / (len(p) + len(t))


def cer(pred_tl, true_tl, empty_true="len"):
    if not true_tl:
        if empty_true == "len":
            return float(len(pred_tl))
        return 0.0 if not pred_tl else float("inf")
    return levenshtein(pred_tl, true_tl) / len(true_tl)


def score_row(pred, true, *, multiset=False, empty_f1=1.0, empty_true="len",
              keep_malformed=True, any_ws=False):
    """(s, T, L, P, G) of one sentence.

    multiset: gloss features counted with repeats (False: as a set).
    empty_f1: F1 of a word whose true and predicted gloss both have no feature.
    empty_true: CER when the true transliteration is empty; "len" divides by 1,
        "exact" gives T = 1 only for an empty prediction.
    keep_malformed: a malformed predicted word leaves an empty slot in the joined
        transliteration (False: it is dropped from the join).
    any_ws: split words on any run of whitespace (False: on single spaces).
    """
    if not isinstance(pred, str) or len(pred) > MAX_CHARS:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    pred = unicodedata.normalize("NFC", pred)
    true = unicodedata.normalize("NFC", true)
    pw, tw = parse(pred, any_ws), parse(true, any_ws)
    n = max(len(pw), len(tw))
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if keep_malformed:
        ptl = " ".join(w[0] for w in pw)
    else:
        ptl = " ".join(w[0] for w, bad in zip(pw, _malformed(pred, any_ws)) if not bad)
    ttl = " ".join(w[0] for w in tw)
    c = cer(ptl, ttl, empty_true)
    T = float(np.exp(-c / TAU)) if c != float("inf") else 0.0
    nl = npos = 0
    g = 0.0
    for p, t in zip(pw, tw):
        nl += p[1] == t[1]
        npos += p[2] == t[2]
        g += gloss_f1(p[3], t[3], multiset, empty_f1)
    L, P, G = nl / n, npos / n, g / n
    return (0.4 * T + 0.2 * L + 0.2 * P + 0.2 * G, T, L, P, G)


def score_parts(y_true, y_pred, **kw):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    rows = np.array([score_row(p, t, **kw) for t, p in zip(y_true, y_pred)], dtype=np.float64)
    m = rows.mean(axis=0) if len(rows) else np.zeros(5)
    return {k: float(v) for k, v in zip(("s", "T", "L", "P", "G"), m)}


def score(y_true, y_pred, **kw):
    return score_parts(y_true, y_pred, **kw)["s"]
