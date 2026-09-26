# SuTRA — Structurally-Unified Tokenization with Root Awareness

[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-1f6feb.svg)](https://www.interspeech2026.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.18087-b31b1b.svg)](https://arxiv.org/abs/2608.18087)
[![Project Page](https://img.shields.io/badge/Project-Page-brightgreen.svg)](https://mo-vaibhavr-43300.github.io/SuTRA/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **Accepted at INTERSPEECH 2026.**

Reference implementation of the SuTRA tokenizer training algorithm from
[**SuTRA: Structurally-Unified Tokenization with Root Awareness**](https://arxiv.org/abs/2608.18087) (Interspeech 2026).

📄 **Paper:** [arXiv:2608.18087](https://arxiv.org/abs/2608.18087) &nbsp;|&nbsp;
🌐 **Project page:** [mo-vaibhavr-43300.github.io/SuTRA](https://mo-vaibhavr-43300.github.io/SuTRA/)

This repo contains the full pipeline: training the ByT5 morpheme segmenter, akshara-aware pre-tokenization, morphological boundary probing, and score-based BPE merging with rigidity annealing.

---

## Algorithm

**Phase 0 — Morpheme segmenter (one-time setup)**

Fine-tune ByT5 on the gold morphological lexicon so that boundaries can be inferred for out-of-lexicon (OOV) words.

**Phase 1 — Pre-tokenization**

1. Group Indic text into akshara-like units (base consonant + matras/halant).
2. Mark forbidden merge boundaries at morpheme edges using the gold lexicon, with ByT5 inference for OOV words.

**Phase 2 — Morphology-aware merging**

Merge pairs by score rather than raw frequency:

$$
S(a, b) = f(a, b) \cdot \Psi(a, b)^{\gamma_t}, \qquad \Psi(a, b) = 1 - \frac{\chi(a, b)}{f(a, b)}
$$

where $f$ is pair frequency, $\chi$ counts boundary violations, and $\gamma_t$ anneals linearly from $\gamma_{\text{start}}$ to $\gamma_{\text{end}}$.

---

## Files

| File | Role |
|------|------|
| `train_byt5.py` | Fine-tunes `google/byt5-small` as a morpheme segmenter (Phase 0) |
| `phase1.py` | `SUTRA_Phase1` — akshara grouping, gold-lexicon lookup, ByT5 boundary inference |
| `phase2.py` | `SUTRATrainer` — score-based BPE with $\Psi$ penalty and $\gamma$ annealing |
| `pilot.py` | End-to-end runner; writes `vocab_16k.json` and `merges_16k.txt` |

---

## Requirements

```bash
pip install torch transformers datasets pandas
```

---

## Data

### Gold morphological lexicon

Download the gold-standard lexicon here: **https://mo-vaibhavr-43300.github.io/SuTRA/**

The lexicon was built with [SampoNLP](https://github.com/AragonerUA/SampoNLP), adapted for Indic scripts as described in the paper (§3, Supp. Mat. §10).

It is used in two formats:

**1. ByT5 training format** (`Word`, `Segmentation` with `+` separators), used by `train_byt5.py`:

```csv
Word,Segmentation
असुविधाजनक,अ+सुविधा+जनक
```

**2. Gold lookup format** (headerless; word followed by one morpheme per column), used by `pilot.py`:

```csv
असुविधाजनक,अ,सुविधा,जनक
```

To convert format 1 into format 2:

```python
import pandas as pd, csv
df = pd.read_csv("data/morph_segmentation.csv").dropna()
with open("data/morph_splits.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    for word, seg in zip(df["Word"], df["Segmentation"]):
        w.writerow([word] + str(seg).split("+"))
```

> Morphemes must concatenate exactly back to the word. Rows that don't reconstruct are ignored as constraints.

### Training corpus

Plain UTF-8 text in the target language.

---

## Usage

### Step 1 — Train the ByT5 segmenter

```bash
python train_byt5.py \
    --csv data/morph_segmentation.csv \
    --output_dir ./byt5_model
```

Optional flags: `--model_name` (default `google/byt5-small`), `--separator` (default `+`), `--epochs` (3), `--batch_size` (32), `--lr` (5e-4), `--max_length` (128), `--test_size` (0.01).

The saved model outputs space-delimited morphemes, which is what `phase1.py` expects.

### Step 2 — Configure the pipeline

Edit paths in `pilot.py`:

```python
CORPUS_PATH   = "data/corpus.txt"
CSV_PATH      = "data/morph_splits.csv"
MODEL_PATH    = "./byt5_model"
TARGET_VOCAB  = 16000
GAMMA_START   = 4.0
GAMMA_END     = 0.0
```

### Step 3 — Run SuTRA

Full pipeline:

```bash
python pilot.py
```

Or run phases individually:

```bash
python phase1.py   # boundary probing smoke test (set MODEL_PATH inside)
python phase2.py   # merge loop smoke test on mock data
```

### Suggested layout

```
SuTRA/
├── data/
│   ├── corpus.txt
│   ├── morph_segmentation.csv   # ByT5 training format
│   └── morph_splits.csv         # gold lookup format
├── byt5_model/                  # output of train_byt5.py
├── train_byt5.py
├── phase1.py
├── phase2.py
└── pilot.py
```

---

## Pipeline

```
Gold lexicon (Word, Segmentation)
    │
    └─► train_byt5.py  →  ./byt5_model

Corpus (UTF-8)
    │
    ├─► Extract words (Devanagari regex)
    │
    ├─► Phase 1: apply_phonetic_grouping  →  "कि ता ब </w>"
    │         get_boundaries (CSV → cache → ByT5)  →  boundary_map
    │
    └─► Phase 2: SUTRATrainer.train
              loop: score pairs with Ψ^γ, merge best
              │
              └─► merges_16k.txt + vocab_16k.json
```

> **Note on scripts:** `pilot.py` extracts words with the Devanagari range (`\u0900-\u097F`) and `phase1.py` groups Devanagari combining marks. For other Indic scripts (e.g. Gujarati, `\u0A80-\u0AFF`), update both regexes to the corresponding Unicode block.

---

## Citation

If you use SuTRA in your work, please cite:

```bibtex
@misc{rathore2026sutrastructurallyunifiedtokenization,
      title={SuTRA : Structurally-Unified Tokenization with Root Awareness},
      author={Vaibhav Rathore and Siddhant Gole and Dadhichi Telwadkar and Rooshil Bhatia and Maulik Ruparel and Siddharth Surekha and Neha Bhargava},
      year={2026},
      eprint={2608.18087},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.18087},
}
```

---

## Related work

**SampoNLP** — morphological lexicon toolkit used in the paper's dataset pipeline:

- Code: https://github.com/AragonerUA/SampoNLP
- Paper: Chelombitko, I., Chelombitko, E., & Komissarov, A. (2025). *SampoNLP: A Self-Referential Toolkit for Morphological Analysis of Subword Tokenizers.* Proceedings of the 10th International Workshop on Computational Linguistics for Uralic Languages (IWCLUL 2025), pp. 57–67. https://aclanthology.org/2025.iwclul-1.8/

---

## Authors

Vaibhav Rathore, Siddhant Gole, Dadhichi Telwadkar, Rooshil Bhatia, Maulik Ruparel, Siddharth Surekha, Neha Bhargava

Motilal Oswal Financial Services Ltd. & IIT Bombay.
