# Egyptian interlinear analysis: notes

## Summary (closed by the user 2026-10-07 08:39)

- Uploads: one. v1 (byt5-small, per-word decoding, full bytes, 20 epochs, lr 1e-3) scored 0.7199 on
  the platform, 2nd public at 07:36 (1st 0.7253, AI 0.6606, reference 0.6746). The lockdown was
  running with the freeze due about 08:22; final rank not reported.
- v2 was not uploaded: the user closed the problem while its clean runs were in flight. Two
  validated candidates existed (label smoothing 0.1, or lr 2e-3), each 0.595 on the page holdout
  against 0.581 for v1, expected about 0.73 on the platform. Scripts are in `submissions/v2/` (lr
  2e-3 + smoothing) and `submissions/v2b/` (lr 1e-3 + smoothing); neither has a finished clean run.
- Money: $2.83 (limit raised to $4 with the user's yes at 08:21). Time: 4h15m.
- Worked: per-word decoding on the encoder states of each word (0.571 at 12 epochs against the
  reference's published 0.545, in 19 minutes); longer training with a higher learning rate
  (+0.010); label smoothing 0.1 at 20 epochs (+0.014); lr 2e-3 (+0.014, does not stack with
  smoothing).
- Did not work: dropping the constant UTF-8 lead bytes of each sign (-0.015), byt5-base (+0.00 to
  +0.02 at 1.7 times the time, 0.012 run noise), a base plus small ensemble (0.5711 at 50 minutes),
  two 12-epoch seeds (0.589, below one 20-epoch model), smaller batches, random sign edits, beam
  search, neighbour words for the decoder, lemma written first, dropout 0.2, smoothing 0.2.
- Validation: the page's own split (19th century BCE or later held out) read 0.139 below the
  platform for v1 (0.581 to 0.7199); the reference's pair is 0.545 to 0.675.

## Approach brief (04:10)

1. **What is scored.** Per sentence, s = 0.4 T + 0.2 L + 0.2 P + 0.2 G, mean over sentences. T is
   exp(-CER/0.2) on the joined transliteration: one wrong character per sentence already costs
   0.085 of s, and T is 0.36 for a 1-word sentence with one wrong character. Whole-sentence
   exactness pays. Word position matters: a dropped middle word scores 0.465 where an empty slot
   scores 0.702, so output must stay aligned to the given words.
2. **Where the points are.** On the page holdout (19th century or later, lookup from the rest):
   lookup 0.315; perfect T with lookup L/P/G 0.672; perfect on unseen spellings only 0.892. Unseen
   spellings carry 0.58 of the 0.685 lost. Reference on test: T 0.510, L 0.734, P 0.841, G 0.777, so
   0.196 of its 0.325 loss is transliteration. Reading unseen spellings is the game.
3. **How the data was made.** Sentences of a corpus with expert analyses, sorted by id (no signal in
   id or order). 51,291 train words, 11,404 spellings (7,987 once), 222 glosses, 10 UPOS. Lemma
   equals the transliteration in 60% of words; PROPN lemmas are capitalised. 2.9% of train strings
   repeat with conflicting analyses.
4. **Test against train.** Test is 17th and 16th century (17th never in train, 16th 9 rows); 33.5%
   unseen spellings. Test sentences are longer (8.0 words against 6.0). The late train rows (18th
   and 19th century, 1,809 rows) alone cover 15.9% of test words, so the final fit uses all rows.
   Validation: `page` (19th or later held out, 46% unseen, harder than test, comparable with the
   reference's published 0.545) is the main one; `near` (18th or later) and `f2` (20th to 22nd) for
   close calls. Absolute level is not the test level (reference 0.545 here, 0.675 on test).
5. **Model family.** Fine-Tuning Domain: a pretrained backbone fine-tuned. ByT5 (byte level) reads
   hieroglyphs natively and needs no vocabulary fitted on anything. Ladder: small 300M, base 582M,
   large 1.23B.
6. **Design.** Per-word decoding: the encoder reads the sentence once; for each word the decoder
   cross-attends only to that word's (contextual) encoder states and writes its four fields.
   Outputs are about 26 bytes, parallel over words, and aligned by construction. This removes the
   long-target cost that limits the reference (159 bytes mean, up to 1,344) and makes beams cheap.
7. **Board.** AI 0.6606, reference 0.6746, top 0.6825 (1 of 10 over the AI at 03:47).

**v1**: per-word ByT5-small, fixed epochs, greedy or small beam, fit on all of train. Run time
target under 25 minutes. Deadline 05:18, so the fresh run starts by 04:35.

**v2 queue** (gain per minute first)
1. Size ladder: base, then large (memory: AdamW state of large is 19 GB, needs Adafactor).
2. Sign noise: mask or drop 10 to 15% of input signs, target unchanged (aims at unseen spellings).
3. Context window for the decoder (neighbour words) and a per-word beam with sentence-level choice.
4. More epochs, two seeds with summed log-probs.
5. Type-balanced sampling of spelling and analysis pairs.
6. Sentence concatenation and crops (test sentences are longer).
Left out: train-lexicon hints in the input (grey, strip-the-model test), `latest_century` as an
input (17th century never in train).

## Fairness notes

- No lookup table anywhere; every field of every word is written by the fine-tuned model.
- `repair` only fixes the shape of a group (four fields, no spaces).

## Results

| run | hold | s | T | L | P | G | translit acc seen / unseen | minutes |
|---|---|---|---|---|---|---|---|---|
| lookup baseline | page | 0.315 | 0.107 | 0.444 | 0.450 | 0.468 | 0.89 / 0 | |
| reference (published) | page | 0.545 | | | | | | 28 |
| e01 small, 12 ep, 240 steps/ep, greedy | page | 0.5714 | 0.380 | 0.645 | 0.756 | 0.696 | 0.894 / 0.362 | 17.4 |
| e01 at epoch 3 / 6 / 9 | page | 0.494 / 0.545 / 0.556 | | | | | | |
| e02 small, 117 steps/ep (big batches), epoch 6 | page | 0.488 | | | | | | killed: fewer updates lose |
| e02 small strip, 233 steps/ep, seed 42 | page | 0.5591 | 0.370 | 0.625 | 0.744 | 0.686 | 0.889 / 0.324 | 15.0 |
| same, seed 7 | page | 0.5537 | 0.359 | 0.627 | 0.740 | 0.683 | 0.889 / 0.339 | 13 |
| same, beam 4 instead of greedy | page | 0.5560 | | | | | | beam gives nothing |
| e02 small full bytes (no strip) | page | 0.5721 | 0.380 | 0.650 | 0.756 | 0.695 | 0.895 / 0.353 | 18.4 |
| e02 small strip + sign noise 0.15 | page | 0.5532 | 0.365 | 0.618 | 0.738 | 0.680 | 0.872 / 0.334 | 12.3 |
| e02 base strip, 10 ep, lr 3e-4 | page | 0.5753 | 0.377 | 0.645 | 0.771 | 0.706 | 0.893 / 0.351 | 21.7 (3090 Ti) |
| e02 small strip 24 ep, at epoch 8 / 12 / 16 / 20 | page | 0.559 / 0.540 / 0.558 / 0.570 | | | | | | 78 s per epoch |

| e03 small strip (0.5538) + base strip (0.5637) ensemble | page | 0.5711 | 0.384 | 0.642 | 0.751 | 0.696 | 0.885 / 0.370 | 27.4; about 50 on an A10 |
| e02 small full bytes, lr 1e-3 | page | 0.5774 | 0.376 | 0.655 | 0.774 | 0.705 | 0.893 / 0.363 | 21.2 |
| e02 small full bytes, 611 updates/ep (budget 20000) | page | 0.5710 | 0.371 | 0.648 | 0.766 | 0.698 | 0.890 / 0.362 | 19.9 |
| e02 small strip 24 ep final | page | 0.5729 | 0.373 | 0.644 | 0.772 | 0.703 | 0.888 / 0.349 | 34.4 |

| e02 small full bytes, lr 1e-3, 20 ep (the uploaded v1 recipe) | page | 0.5807 | 0.390 | 0.654 | 0.767 | 0.701 | 0.900 / 0.370 | 21.5 |
| same at 12 ep with CTX=1 (decoder sees neighbour words) | page | 0.5699 | 0.372 | 0.648 | 0.759 | 0.699 | 0.899 / 0.359 | 15.1; no gain against 0.5774 |

A10 timing (vast.ai A10, full train): base strip at budget 30000 is 195 s per epoch, small strip at
40000 is 83 s, joint decode of the test 50 s. The base model at budget 45000 printed allocator
out-of-memory retries on 24 GB cards; 30000 is the safe size.

Decisions at 05:50: the base plus small ensemble is dropped (0.5711 at 50 minutes against 0.572 for
the v1 recipe at 19). The same base recipe read 0.5753 and 0.5637 on two hosts, so base runs carry
0.012 of run noise; the small full-bytes recipe read 0.5714, 0.5721 and 0.5710 in three runs.
Candidate for a stronger v1: small, full bytes, lr 1e-3, 20 epochs.

Read: seed noise about 0.005 (two seeds of small strip). Dropping the two constant lead bytes of
each sign costs 0.015 (0.5714 and 0.5721 with them, 0.5591 and 0.5537 without) and saves 30% time.
Sign noise: no gain. Beam: no gain. Base over small at equal input: +0.02. One evaluation inside a
run swings by 0.02 between epochs at high learning rate, so only end-of-schedule numbers count.

Preflight v1: READY. Two WARNs kept: argv is the page's run command; hub weights are allowed in
Fine-Tuning. v1 fresh run 1,128 s on an RTX 3090.

## v2 screening (all small, full bytes, 20 epochs, page holdout; v1 recipe = lr 1e-3)

| change | s | note |
|---|---|---|
| v1 recipe, seed 42 / seed 7 | 0.5807 / 0.5813 | run noise of this recipe is about 0.001 |
| lemma written first, seed 42 / seed 7 | 0.5783 / 0.5856 | mean 0.582: no gain, dropped |
| decoder sees neighbour words (12 ep) | 0.5699 | against 0.5774: dropped |
| dropout 0.2 | 0.566 at epoch 15 | behind 0.577, killed |
| learning rate 2e-3 | 0.5952 | +0.014. It read 0.569 at epoch 15 and jumped as the rate decayed |
| label smoothing 0.1 | 0.5972 | +0.016 |

v1 platform 0.7199, rank 2; 1st 0.7253 (07:39). Lockdown running, roster freeze about 08:22.
The three idle machines the user pointed at belong to the speed-lab chat; the tool refuses another
problem's machines, so they were left alone.

## Error analysis (e01, page holdout)

- Lost points: T 0.248, L 0.071, G 0.061, P 0.049.
- Seen spellings (54% of words): T 0.894, L 0.894, P 0.897, G 0.797 exact.
- Unseen spelling, known lemma (32%): T 0.386, L 0.488. Unseen, new lemma (14%): T 0.307, L 0.221.
- Many unseen errors are late-period sign habits absent from the early fit (the coil sign for w in
  pw, iw, tw). The final fit includes the 18th and 19th century rows, so `page` overstates them;
  `near` (19th in the fit, 18th held out) is the better guide for those.
- Sentence length barely matters (s 0.52 at 2 to 3 words, 0.60 at 20+).

## Calibration

| version | page holdout | near | CSV check | platform |
|---|---|---|---|---|
| v1_old (small, 12 ep, lr 5e-4), not uploaded | 0.571 | | | |
| v1 (small, 20 ep, lr 1e-3), ready 06:43 | 0.581 | | not reported | 0.7199 (07:28) |

Gap: page holdout 0.581 reads 0.72 outside, +0.139. The reference's published pair is 0.545 and 0.675, +0.130.
