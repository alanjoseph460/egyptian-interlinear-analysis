# Egyptian Interlinear Analysis

My solution to the **Egyptian Interlinear Analysis** challenge on Shipd Project Eris, a Kaggle-style ML
competition with auto-graded solutions.

Task: Give the transliteration, lemma, part of speech and gloss of every word in a hieroglyphic sentence.

## Result

- Public score **0.7199**, **rank 2** when last checked (1st 0.7253, AI baseline 0.6606,
  challenge reference 0.6746).
- Final rank was not reported.

## Approach

- google/byt5-small is fine-tuned inside the script as a word-by-word analyser.
- The encoder reads the whole sentence as raw UTF-8 bytes, so no tokenizer or vocabulary is fitted.
- The word boundaries are given, so the decoder runs once per word, attending only to that
  word's encoder states, and writes transliteration;lemma;UPOS;gloss.
- 20 epochs of AdamW at a peak learning rate of 1e-3, greedy decoding.

## Files

- `submissions/v1/solution.py`: the uploaded solution (43-minute run).
- `submissions/v2/`, `submissions/v2b/`: validated improvements (label smoothing, higher
  learning rate) without a finished clean run; not uploaded.
- `submissions/v1_old/`, `submissions/ens_dropped/`: earlier and dropped versions.
- `common.py`, `metric.py`: shared experiment code.
- `NOTES.md`: the full work log (what worked, what did not, measured numbers).
- `REQUIREMENTS.md`: the challenge statement as saved when the problem was picked.
- `results.tsv`: every scored experiment.
- `exp/`: experiment scripts and logs.

## Running

The solution is one script that trains and predicts from the raw files on every run. It reads
`./dataset/public/` and writes `./working/submission.csv`. The challenge data is not included.
