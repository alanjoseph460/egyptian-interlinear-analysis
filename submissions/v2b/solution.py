"""Interlinear analysis of hieroglyphic sentences with a fine-tuned ByT5, decoded word by word.

google/byt5-small (pinned revision) is fine-tuned on train.csv only. The encoder reads the whole
sentence as raw UTF-8 bytes, so every hieroglyph and every <g> code is read as it is and no
vocabulary is built from any file. Word boundaries are given, so the decoder is run once per word:
it cross-attends to the encoder states of that word's bytes, which already carry the sentence
context, and writes `transliteration;lemma;UPOS;gloss` for it. All four layers of a word are
generated together by the same model, and the output has exactly one analysis per hieroglyphic
word by construction. Test sentences are analysed one at a time; batching is only for speed.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import csv
import os
import random
import sys

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

SEED = 42
DEVICE = torch.device("cuda")
MODEL = "google/byt5-small"
REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
EPOCHS = 20
LR = 1e-3
LABEL_SMOOTHING = 0.1
WARMUP = 0.05
N_ENC, N_DEC = 12, 4  # layers of byt5-small, used to size batches
BATCH_POSITIONS = 40000  # fixed batch size, counted in positions x layers; about 390 updates per epoch
DECODER_CHUNKS = 3  # the decoder runs on length-sorted thirds of a batch, so short targets are not padded
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


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_ids(text):
    return [b + 3 for b in text.encode("utf-8")]


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
            src += to_ids(word)
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


def train_batches(items, rng):
    """Shuffled batches of sentences of similar length and of one fixed size, counted as
    sentences x longest source x encoder layers + target bytes x 1.3 x decoder layers."""
    order = list(range(len(items)))
    rng.shuffle(order)
    chunks = [order[i:i + 1600] for i in range(0, len(order), 1600)]
    order = [j for c in chunks for j in sorted(c, key=lambda j: len(items[j][0]))]
    out, cur, max_src, tgt_bytes = [], [], 0, 0
    for j in order:
        ls = len(items[j][0]) + 2
        lt = sum(len(t) for t in items[j][2])
        new_src = max(max_src, ls)
        cost = (len(cur) + 1) * new_src * N_ENC + (tgt_bytes + lt) * 1.3 * N_DEC
        if cur and cost > BATCH_POSITIONS:
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
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION)
        # torch's tanh GELU is the same function as T5's own, computed in one operation
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
            logits = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=memory[part, :lw]),
                             attention_mask=word_mask[part, :lw].long(), labels=labels[part, :lt]).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1), labels[part, :lt].flatten(), ignore_index=-100,
                label_smoothing=LABEL_SMOOTHING)
            n = int(lengths[part].sum())
            total, count = total + loss * n, count + n
        return total / count

    @torch.no_grad()
    def analyse(self, src, row, pos, word_mask):
        memory = BaseModelOutput(last_hidden_state=self.word_memory(src, row, pos))
        return self.t5.generate(encoder_outputs=memory, attention_mask=word_mask.long(),
                                max_new_tokens=MAX_TGT, num_beams=1, do_sample=False)


def clean_group(text):
    """The grader needs exactly four ;-separated fields and no space inside a word's analysis."""
    fields = text.replace(" ", "").split(";")
    return ";".join((fields + ["", "", "", ""])[:4])


def main():
    public_dir, submission_out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(submission_out)), exist_ok=True)
    train = read_rows(os.path.join(public_dir, "train.csv"))
    test = read_rows(os.path.join(public_dir, "test.csv"))
    train_items = prepare(train, True)
    test_items = prepare(test, False)
    print(f"train sentences {len(train)}, test sentences {len(test)}", flush=True)

    model = WordAnalyser().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    epochs = [train_batches(train_items, rng) for _ in range(EPOCHS)]
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
        print(f"epoch {epoch + 1}/{EPOCHS}: mean training loss {loss_sum / len(batch_list):.4f}", flush=True)

    model.eval()
    predictions = [None] * len(test)
    for b in predict_batches(test_items):
        src, row, pos, word_mask, _ = collate([test_items[j] for j in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.analyse(src.to(DEVICE), row.to(DEVICE), pos.to(DEVICE), word_mask.to(DEVICE))
        out = out.tolist()
        k = 0
        for j in b:
            n = len(test_items[j][1])
            groups = [clean_group(to_text(o[1:])) for o in out[k:k + n]]  # o[0] is the start token
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
