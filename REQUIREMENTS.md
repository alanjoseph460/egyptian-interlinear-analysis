# Interlinear Analysis of Egyptian Hieroglyphs from a Later Period: Joint Transliteration, Lemma, Part of Speech and Morphology

Saved 2026-10-07 03:47 from the user's paste and three screenshots.

## Page facts

- Domain: **Fine-Tuning**. Difficulty: Hard. Tags: text. Status: Accepted.
- Compute: **A10G**.
- Scoring: higher is better, range 0 to 1.
- Creator shivam_s, approver poggerman. Created Sep 27, 2026.
- Board at 03:47: 1 submission, 1/10 solvers beat AI. Top score 0.6825 (shadxzz, 41 min ago).
  AI baseline 0.6606. "4 more distinct solvers needed to activate the $650 prize pool and start the
  closing countdown. At 10, up to 10 solvers will be selected to continue."
- Prizes: top 3 $250 / $150 / $100, others share $150 by merit.
- Submission credits 6/6.
- Runtime command: `python3 solution.py <public_dir> <submission_out>`
- Workspace: input `dataset/public/` (sample_submission.csv 231 KB, test.csv 187 KB, train.csv
  2.1 MB), `solution.py` or `.ipynb`, output `working/submission.csv`. "Your script receives the
  exact output path as its second argument. Upload a sample output here."
- Upload form: 1 solution.py (.py or .ipynb), 2 submission.csv (predictions output), 3 Time Spent
  (hours, "Total time spent, including iterations, training, and research").
- Pre-submission checks: CSV Score Validation, Prompt Compliance, Held-out Answer Ingestion,
  Deterministic Execution.
- Dataset source is visible after the challenge closes.

## Problem Description (verbatim)

### Overview

Egyptologists analyse a hieroglyphic text word by word. For each word they write:

its transliteration into a Latin-based alphabet (the consonants the signs stand for, the endings and the suffix pronouns);

its lemma, the dictionary entry it belongs to;

its part of speech;

its morphological gloss (gender, number, person, verb form).

This interlinear analysis is the first step of every edition, dictionary entry and translation. It cannot be done by substituting sign for sound. Hieroglyphic writing mixes:

signs for sounds (one, two or three consonants);

signs for whole words;

determinatives: silent signs that mark a word's meaning.

Scribes also left letters unwritten, abbreviated common words, and spelled the same word in different ways across the centuries.

Each row of this challenge is one Ancient Egyptian sentence in Unicode hieroglyphs, with the word boundaries given. Produce its full interlinear analysis in the conventions used in train.csv: for every word, the transliteration, the lemma, the Universal Dependencies part-of-speech tag and the morphological gloss.

The training sentences date from the Old Kingdom to the 12th Dynasty. Every test sentence comes from a later period, the late Middle Kingdom and Second Intermediate Period (about 1800-1550 BCE), with its own spellings, vocabulary and kinds of text.

This is a structured sequence-to-sequence task. The expected approach is to fine-tune a pretrained model on a GPU so that it produces all four layers together. Hieroglyphs are Unicode characters outside every common tokenizer's vocabulary, so byte-level or character-level pretrained models are the natural starting point.

### Why it matters

Analysing newly recorded texts is slow expert work, and a large share of excavated and museum material is still unpublished. A model that proposes a word-by-word analysis from the signs could speed up editing and help check existing annotations.

The four layers depend on each other. The ending read in the transliteration decides the gloss, and the lemma decides the part of speech. Scoring them jointly rewards a reading that is linguistically consistent, not only a string that looks close.

The data covers a limited set of complete sentences, so this is a benchmark for analysing a logo-phonetic script under a change of period, not a tool for publishing editions.

### Data

Files in public/:

train.csv - 8,505 sentences whose latest possible date is before about 1800 BCE, with their analysis.

id - string. Opaque sentence identifier.

latest_century - string. Century of the latest possible date of the text witness, for example 19th century BCE.

n_words - integer. Number of words in the sentence.

hieroglyphs - string. The sentence in Unicode Egyptian Hieroglyphs, one space between words. Signs that Unicode does not encode are written as Gardiner-style codes in `<g>...</g>`, for example `<g>M12B</g>`.

prediction - string. The interlinear analysis: one analysis per hieroglyphic word, in the same order, separated by single spaces. Each word's analysis is four fields joined by ;: transliteration;lemma;UPOS;gloss.

transliteration: Leiden Unified Transliteration (for example ꜣ, ꞽ, ꜥ, ḥ, ḫ, ẖ, š, ḏ, ṯ), with these marks: . before endings, = before suffix pronouns, and (...) for letters the scribe did not write;

lemma: the citation form of the word's dictionary entry (for example ssn for ssn.n);

UPOS: one of NOUN, PRON, VERB, ADP, PROPN, ADJ, PART, ADV, INTJ, NUM;

gloss: the morphological gloss, for example N.m (masculine noun), -3sg.m (suffix pronoun, 3rd person singular masculine), V\tam.act:stpr, PREP, TITL (title).

test.csv - 1,519 sentences from about 1800-1550 BCE. Same columns without prediction.

sample_submission.csv - a valid submission that writes the most frequent training word analysis once per word of the sentence.

id - string. Sentence identifier from test.csv.

prediction - string. An interlinear analysis.

Example (train), a sentence of three words:

𓂜 𓋴𓋴𓈖𓂉𓈖 𓆑
n;n;PART;PTCL ssn.n;ssn;VERB;V\tam.act-ant:stpr =f;=f;PRON;-3sg.m

### The train-to-test shift

Training sentences are from the Old Kingdom to the 12th Dynasty. Every test sentence is from the late Middle Kingdom or Second Intermediate Period, a period absent from training.

Spelling conventions changed over time, new words entered the language, and the kinds of text differ. The early material is dominated by religious and funerary formulae. The later material has more literary, administrative and medical writing.

33.5% of the test words have a hieroglyphic spelling that never occurs in training, so memorising words is not enough: the model has to read the signs.

9.7% of the test words belong to a lemma never seen in training.

We measured the shift with a word-lookup baseline (each hieroglyphic spelling mapped to its most frequent training analysis) and the real grader. We scored half of the test period twice at equal training size, averaged over 5 random halvings:

trained without the test period: 0.424

trained with the other half of the test period swapped in: 0.564

Leaving the period out costs 0.140 (standard deviation over halvings 0.004).

A random split of train.csv over-reads, because recurring formulae land on both sides. The closest local imitation is to validate on the latest training sentences (latest_century of 19th century BCE or later). The reference recipe scores 0.545 on that validation and 0.675 on the test. The absolute levels differ because the kinds of text differ. Use the validation to compare methods, not to predict the test score.

### What makes this hard

The script is logo-phonetic. One sign can be a sound, a word or a silent determinative, and the same word can be spelled several ways.

Unwritten letters, abbreviations and grammatical endings have to be inferred and written in the notation of train.csv.

Lemma, part of speech and gloss depend on reading the word correctly and on its role in the sentence. For example, the same spelling can be a noun or a verb form.

Unseen spellings and vocabulary in a later period (above).

Hieroglyphs lie outside the vocabulary of standard tokenizers, so the model and its tokenizer have to handle raw Unicode (U+13000 and up), plus `<g>` codes for unencoded signs.

Sentences range from 1 to 53 words, and a full analysis is about three times as long as the transliteration alone.

The GPU budget is one hour, including decoding.

### Evaluation

Each sentence gets a score s from 0 to 1, made of four parts. Both strings are first Unicode NFC-normalised. Words are compared position by position, and n is the larger of the true and predicted word counts.

Transliteration T = exp(-CER / 0.2). Here CER = Levenshtein(predicted, true) / length(true), computed on the transliteration fields joined by single spaces.

Lemma L = number of words with exactly the right lemma, divided by n.

Part of speech P = number of words with exactly the right UPOS tag, divided by n.

Gloss G = sum over words of the F1 of their gloss features, divided by n.

A gloss is split into features at \, ., :, - and =. For example V\tam.act:stpr has the features V, tam, act, stpr.

A nearly right gloss therefore earns partial credit. For example, -3sg.m against the true -3sg.f earns 0.5, and V\tam.act against the true V\tam.act:stpr earns 0.86.

s = 0.4 T + 0.2 L + 0.2 P + 0.2 G.

The score is the mean of s over all test sentences. Higher is better; minimum 0, maximum 1.

Reading guide for the transliteration part:

an exact transliteration earns T = 1;

5% of the characters wrong earns 0.78;

10% earns 0.61;

20% earns 0.37;

50% earns 0.08.

A perfect transliteration with the other three layers left empty scores 0.4. A perfect transliteration with the other layers taken from a training lookup scores 0.747.

Rules applied by the grader:

The submission must have exactly the columns id,prediction and exactly one row per test id. Missing, duplicated or unknown ids, or any other column, give 0 for the whole submission.

A predicted word that does not have exactly four ;-separated fields counts as a word with four empty fields.

Extra words count against the lemma, UPOS and gloss parts through n.

A prediction that is not a string (for example NaN), or is longer than 6,000 characters, scores 0 for its row. It is never treated as a default answer.

Measured baselines on the test set (the real grader; T, L, P, G in brackets):

sample_submission.csv unchanged (the most frequent training word analysis, once per word): 0.0903 (0.021, 0.029, 0.200, 0.179)

word lookup: each hieroglyphic word spelling replaced by its most frequent analysis in train, unseen spellings by the most frequent analysis: 0.4238 (0.191, 0.579, 0.574, 0.583)

reference solution: 0.6746 (0.510, 0.734, 0.841, 0.777).

It fine-tunes google/byt5-small (pinned revision) for 10 epochs to produce all four layers at once, decodes with beam search, and repairs outputs to one analysis per word.

It runs in about 28 minutes on a 24 GB Ampere GPU.

Its transliteration word accuracy is 84% on word spellings seen in training and 44% on unseen ones.

perfect: 1.0000

### Submission format

A UTF-8 CSV file submission.csv with a header row and exactly 1,519 rows, one per test sentence:

id - the sentence id from test.csv.

prediction - the interlinear analysis, in the format of train.csv: one transliteration;lemma;UPOS;gloss group per hieroglyphic word, groups separated by single spaces.

Example row:

S1a2b3c4d5e,n;n;PART;PTCL ssn.n;ssn;VERB;V\tam.act-ant:stpr =f;=f;PRON;-3sg.m

### What is rewarded

Reading new spellings sign by sign, not only recalling whole words from training.

A consistent analysis across the four layers: the ending in the transliteration, the lemma, the tag and the gloss should agree.

Correct endings, suffix marks and omitted-letter brackets in the notation of train.csv.

Use of the given word boundaries (one analysis per hieroglyphic word).

Validation on the latest training period, with separate figures per layer and for seen and unseen word spellings.

A fine-tuning recipe that fits one GPU and one hour.

### What not to use

No searching for or looking up the sentences or their analyses in any external source: web sites, APIs, online databases, dataset or model hubs, or any printed or digital edition.

No external dictionaries, lemma lists, translations, or hand analysis of test sentences.

No external Egyptian text corpora or sign lists with readings. Pretrained general-purpose models (for example byte-level or multilingual language models) used as a starting point for fine-tuning are allowed.

No use of the test set beyond per-sentence inference. That excludes pseudo-labelling, fitting vocabularies or tokenizers on test sentences, and tuning to statistics of the test period.

No hard-coding of anything that can only come from the answers, and no degenerate outputs such as the same text for every sentence.

### Expected Output

Your script receives the public dataset directory and exact submission CSV path as two positional arguments.

## Checklist (hard requirements)

Format
- [ ] One file `solution.py`, run as `python3 solution.py <public_dir> <submission_out>`. Read
      `sys.argv[1]` and `sys.argv[2]` directly, no defaults (playbook: fixed paths and argv
      defaults both failed checks before).
- [ ] Reads `train.csv` and `test.csv` from `<public_dir>`, opened as UTF-8.
- [ ] Writes a UTF-8 CSV with exactly the columns `id,prediction`, exactly 1,519 rows, one per
      test id, no duplicates, no unknown ids, no extra column. Any slip gives 0 for everything.
- [ ] Every prediction is a non-empty string under 6,000 characters. Never NaN.
- [ ] Each sentence has exactly `n_words` groups separated by single spaces, each group exactly
      four `;`-separated fields. A repair step guarantees this.
- [ ] Output NFC-safe; the grader NFC-normalises both sides.

Runtime
- [ ] **One hour on the A10G, decoding included** (stated on the page, so it binds over the
      guide's 1.5 hours). Plan 50 minutes or less on an RTX 3090. Preflight `--limit-min 60`.

Model (Domain Fine-Tuning)
- [ ] A pretrained general-purpose backbone is really fine-tuned inside the script and writes
      every prediction. Byte-level or character-level models are the natural start.
- [ ] Weights from the Hugging Face hub at run time, revision pinned. No weights of ours.
- [ ] Kaggle image libraries only, no pip, nothing from GitHub.

Bans in the text
- [ ] No lookup of the sentences or analyses in any outside source.
- [ ] No external dictionaries, lemma lists, translations, sign lists with readings, Egyptian
      corpora. Nothing typed in from Egyptological knowledge: no sign-to-sound table.
- [ ] No use of the test set beyond per-sentence inference: no pseudo-labels, no vocabulary or
      tokenizer fitted on test sentences, no tuning to test-period statistics. Any vocabulary is
      built from train.csv only; an unseen test sign maps to an unknown token or raw bytes.
- [ ] No hard-coded answers, no degenerate constant output.
- [ ] `latest_century` of a test row is an input column of that row: usable per row. Not used to
      pool statistics across test rows.

Determinism
- [ ] Guide seed template, `DEVICE = torch.device("cuda")`, fixed epochs and steps, no clock
      reads, no machine-dependent branches, seeded loaders.

Grey zone here (keep minor, note in NOTES.md)
- A lookup of train word analyses doing the model's work. The page itself measures a "word lookup"
  baseline, but under Fine-Tuning the fine-tuned model must carry the prediction. A train lexicon
  may at most be an input hint or a repair for malformed groups, never the main predictor.

Leftover wording
- None found. The one-hour budget is a restriction, not a leftover, and is obeyed.
