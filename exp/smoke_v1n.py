"""Interlinear analysis of hieroglyphic sentences with two fine-tuned ByT5 models, decoded word by word.

google/byt5-base and google/byt5-small (pinned revisions) are each fine-tuned on train.csv only.
The encoder reads the whole sentence as bytes, so every hieroglyph and every <g> code is read as it
is and no vocabulary is built from any file. Every sign of the Egyptian Hieroglyphs block starts
with the same two UTF-8 bytes; they carry no information and are dropped, which halves the input.
Word boundaries are given, so the decoder is run once per word: it cross-attends to the encoder
states of that word's bytes, which already carry the sentence context, and writes
`transliteration;lemma;UPOS;gloss` for it. All four layers of a word are generated together, and
the output has exactly one analysis per hieroglyphic word by construction.

At prediction time the two models decode together: at every output byte their log-probabilities
are added and the best byte is taken. Test sentences are analysed independently of each other;
batching is only for speed.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import csv
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

SEED = 42
DEVICE = torch.device("cuda")
# name, pinned revision, encoder and decoder layers, learning rate, epochs, batch budget, seed.
# The budget is padded positions x layers per training batch, sized for a 24 GB card.
MEMBERS = [
    ("google/byt5-base", "92d8c008d55cf7c254915bac165171dfe6c20c44", 18, 6, 3e-4, 1, 45000, 42),
    ("google/byt5-small", "68377bdc18a2ffec8a0533fef03b1c513a4dd49d", 12, 4, 5e-4, 1, 40000, 43),
]
WARMUP = 0.05
DECODER_CHUNKS = 3  # the decoder runs on length-sorted thirds of a batch to avoid padding
PRED_WORDS = 512  # words per decoding batch
MAX_TGT = 128  # bytes of one word analysis; the longest in train is 161, 99.9% are under 94
MAX_PRED_CHARS = 5900  # the grader zeroes a row over 6,000 characters

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PAD, EOS, SPACE = 0, 1, 32 + 3  # ByT5 ids: byte value + 3
SIGN = re.compile(r"<g>.*?</g>|.", re.S)
LEAD = b"\xf0\x93"


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_ids(text):
    return [b + 3 for b in text.encode("utf-8")]


def sign_ids(sign):
    b = sign.encode("utf-8")
    if len(b) == 4 and b[:2] == LEAD:
        b = b[2:]
    return [x + 3 for x in b]


def to_text(ids):
    out = bytearray()
    for i in ids:
        if i == EOS:
            break
        if 3 <= i < 259:
            out.append(i - 3)
    return out.decode("utf-8", errors="ignore")


def prepare(rows, with_target):
    """Per sentence: source ids, the byte span of each word, and one target per word."""
    items = []
    for r in rows:
        src, spans = [], []
        for k, word in enumerate(r["hieroglyphs"].split(" ")):
            if k:
                src.append(SPACE)
            start = len(src)
            for sign in SIGN.findall(word):
                src += sign_ids(sign)
            spans.append((start, len(src)))
        src.append(EOS)
        targets = None
        if with_target:
            groups = r["prediction"].split(" ")
            assert len(groups) == len(spans)
            targets = [to_ids(g)[:MAX_TGT - 1] + [EOS] for g in groups]
        items.append((src, spans, targets))
    return items


def collate(batch):
    """Source tensor, the index that gathers each word's encoder states, and the labels."""
    longest = max(len(b[0]) for b in batch)
    src = torch.full((len(batch), longest), PAD, dtype=torch.long)
    row, start, end = [], [], []
    for i, b in enumerate(batch):
        src[i, :len(b[0])] = torch.tensor(b[0])
        for a, e in b[1]:
            row.append(i), start.append(a), end.append(e)
    n_words = len(row)
    width = max(e - a for a, e in zip(start, end))
    pos = torch.arange(width).unsqueeze(0).expand(n_words, width) + torch.tensor(start).unsqueeze(1)
    word_mask = pos < torch.tensor(end).unsqueeze(1)
    pos = torch.where(word_mask, pos, torch.zeros_like(pos))
    row = torch.tensor(row).unsqueeze(1).expand(n_words, width)
    labels = None
    if batch[0][2] is not None:
        targets = [t for b in batch for t in b[2]]
        labels = torch.full((n_words, max(len(t) for t in targets)), -100, dtype=torch.long)
        for j, t in enumerate(targets):
            labels[j, :len(t)] = torch.tensor(t)
    return src, row, pos, word_mask, labels


def train_batches(items, rng, n_enc, n_dec, budget):
    """Shuffled batches of sentences of similar length under a memory budget. Memory follows
    sentences x longest source x encoder layers + target bytes x 1.3 x decoder layers."""
    order = list(range(len(items)))
    rng.shuffle(order)
    chunks = [order[i:i + 1600] for i in range(0, len(order), 1600)]
    order = [j for c in chunks for j in sorted(c, key=lambda j: len(items[j][0]))]
    out, cur, max_src, tgt_bytes = [], [], 0, 0
    for j in order:
        ls = len(items[j][0]) + 1
        lt = sum(len(t) for t in items[j][2])
        new_src = max(max_src, ls)
        cost = (len(cur) + 1) * new_src * n_enc + (tgt_bytes + lt) * 1.3 * n_dec
        if cur and cost > budget:
            out.append(cur)
            cur, tgt_bytes, new_src = [], 0, ls
        cur.append(j)
        tgt_bytes, max_src = tgt_bytes + lt, new_src
    out.append(cur)
    rng.shuffle(out)
    return out


def predict_batches(items):
    order = sorted(range(len(items)), key=lambda j: len(items[j][0]))
    out, cur, words = [], [], 0
    for j in order:
        n = len(items[j][1])
        if cur and words + n > PRED_WORDS:
            out.append(cur)
            cur, words = [], 0
        cur.append(j)
        words += n
    out.append(cur)
    return out


class WordAnalyser(torch.nn.Module):
    def __init__(self, name, revision):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(name, revision=revision)
        # T5's own tanh GELU is written with several float32 temporaries; torch's is the same
        # function in one operation and needs far less memory.
        for m in self.t5.modules():
            if hasattr(m, "wi_0") and hasattr(m, "act"):
                m.act = torch.nn.GELU(approximate="tanh")

    def word_memory(self, src, row, pos):
        states = self.t5.encoder(input_ids=src, attention_mask=(src != PAD).long()).last_hidden_state
        return states[row, pos]

    def forward(self, src, row, pos, word_mask, labels):
        memory = self.word_memory(src, row, pos)
        lengths = (labels != -100).sum(1)
        total, count = 0.0, 0
        for part in torch.tensor_split(torch.argsort(lengths), DECODER_CHUNKS):
            if len(part) == 0:
                continue
            lt, lw = int(lengths[part].max()), int(word_mask[part].sum(1).max())
            loss = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=memory[part, :lw]),
                           attention_mask=word_mask[part, :lw].long(), labels=labels[part, :lt]).loss
            n = int(lengths[part].sum())
            total, count = total + loss * n, count + n
        return total / count


def fine_tune(member, train_items):
    name, revision, n_enc, n_dec, lr, n_epochs, budget, seed = member
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = WordAnalyser(name, revision).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    epochs = [train_batches(train_items, rng, n_enc, n_dec, budget) for _ in range(n_epochs)]
    total = sum(len(e) for e in epochs)
    warm = int(WARMUP * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    for epoch, batch_list in enumerate(epochs):
        model.train()
        loss_sum = 0.0
        for b in batch_list:
            src, row, pos, word_mask, labels = collate([train_items[j] for j in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(src.to(DEVICE), row.to(DEVICE), pos.to(DEVICE), word_mask.to(DEVICE),
                             labels.to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        print(f"{name} epoch {epoch + 1}/{n_epochs}: mean training loss {loss_sum / len(batch_list):.4f}",
              flush=True)
    return model.eval()


@torch.no_grad()
def analyse(models, src, row, pos, word_mask):
    """Greedy decoding of every word on the summed log-probabilities of the models."""
    n_words = row.shape[0]
    memories = [BaseModelOutput(last_hidden_state=m.word_memory(src, row, pos)) for m in models]
    caches = [None] * len(models)
    last = torch.full((n_words, 1), PAD, dtype=torch.long, device=DEVICE)  # T5 starts from the pad id
    done = torch.zeros(n_words, dtype=torch.bool, device=DEVICE)
    steps = []
    for _ in range(MAX_TGT):
        log_prob = 0
        for k, m in enumerate(models):
            out = m.t5(encoder_outputs=memories[k], attention_mask=word_mask.long(),
                       decoder_input_ids=last, past_key_values=caches[k], use_cache=True)
            caches[k] = out.past_key_values
            log_prob = log_prob + torch.log_softmax(out.logits[:, -1].float(), dim=-1)
        token = log_prob.argmax(-1)
        token[done] = EOS
        steps.append(token)
        done |= token == EOS
        last = token.unsqueeze(1)
        if bool(done.all()):
            break
    return torch.stack(steps, 1).tolist()


def clean_group(text):
    """The grader needs exactly four ;-separated fields and no space inside a word's analysis."""
    fields = text.replace(" ", "").split(";")
    return ";".join((fields + ["", "", "", ""])[:4])


def main():
    public_dir, submission_out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(submission_out)), exist_ok=True)
    train = read_rows(os.path.join(public_dir, "train.csv"))[:500]
    test = read_rows(os.path.join(public_dir, "test.csv"))[:200]
    train_items = prepare(train, True)
    test_items = prepare(test, False)
    print(f"train sentences {len(train)}, test sentences {len(test)}", flush=True)

    models = [fine_tune(member, train_items) for member in MEMBERS]

    predictions = [None] * len(test)
    for b in predict_batches(test_items):
        src, row, pos, word_mask, _ = collate([test_items[j] for j in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = analyse(models, src.to(DEVICE), row.to(DEVICE), pos.to(DEVICE), word_mask.to(DEVICE))
        k = 0
        for j in b:
            n = len(test_items[j][1])
            groups = [clean_group(to_text(o)) for o in out[k:k + n]]
            k += n
            # keep a very long sentence under the grader's length cap by blanking its longest groups
            while len(" ".join(groups)) > MAX_PRED_CHARS:
                groups[max(range(n), key=lambda i: len(groups[i]))] = ";;;"
            predictions[j] = " ".join(groups)

    assert len(predictions) == len(test) and all(isinstance(p, str) and p for p in predictions)
    assert len({r["id"] for r in test}) == len(test)
    for r, p in zip(test, predictions):
        assert len(p.split(" ")) == len(r["hieroglyphs"].split(" "))
    with open(submission_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "prediction"])
        for r, p in zip(test, predictions):
            writer.writerow([r["id"], p])
    print(f"wrote {len(predictions)} rows to {submission_out}", flush=True)


if __name__ == "__main__":
    main()
