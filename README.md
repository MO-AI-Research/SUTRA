# SuTRA — Training Code

Reference implementation of the SuTRA tokenizer training algorithm from [*SuTRA: Structurally-Unified Tokenization with Root Awareness*](SuTRA_Interspeech2026.pdf) (Interspeech 2026).

Project page: https://mo-vaibhavr-43300.github.io/SuTRA/

This repo contains the core two-phase training pipeline: akshara-aware pre-tokenization, morphological boundary probing, and score-based BPE merging with rigidity annealing.

---

## Algorithm

**Phase 1 — Pre-tokenization**

1. Group Indic text into akshara-like units (base consonant + matras/halant).
2. Mark forbidden merge boundaries at morpheme edges using a gold lexicon, with ByT5 inference for OOV words.

**Phase 2 — Morphology-aware merging**

Merge pairs by score rather than raw frequency:

\[
S(a, b) = f(a, b) \cdot \Psi(a, b)^{\gamma_t}, \quad \Psi(a, b) = 1 - \frac{\chi(a, b)}{f(a, b)}
\]

where \(f\) is pair frequency, \(\chi\) counts boundary violations, and \(\gamma_t\) anneals linearly from \(\gamma_{\text{start}}\) to \(\gamma_{\text{end}}\).

---

## Files

| File | Role |
|------|------|
| `phase1.py` | `SUTRA_Phase1` — akshara grouping, gold-lexicon lookup, ByT5 boundary inference |
| `phase2.py` | `SUTRATrainer` — score-based BPE with \(\Psi\) penalty and \(\gamma\) annealing |
| `pilot.py` | End-to-end runner; writes `vocab_16k.json` and `merges_16k.txt` |

---

## Requirements

```bash
pip install torch transformers
```

External inputs:

- Training corpus (plain UTF-8 text)
- Gold morphological splits CSV (`word, morpheme1, morpheme2, ...`)
- Fine-tuned ByT5 seq2seq model (space-delimited morpheme output)

The morphological gold lexicon used with SuTRA can be built with [SampoNLP](https://github.com/AragonerUA/SampoNLP), adapted for Indic scripts as described in the paper (§3, Supp. Mat. §10).

---

## Usage

Edit paths in `pilot.py`:

```python
CORPUS_PATH = "corpus_small.txt"
CSV_PATH = "hindi_morph_splits.csv"
MODEL_PATH = "./byt5_model"
TARGET_VOCAB = 16000
GAMMA_START = 4.0
GAMMA_END = 0.0
```

Run the full pipeline:

```bash
python pilot.py
```

Or run phases individually:

```bash
python phase1.py   # boundary probing smoke test
python phase2.py   # merge loop smoke test on mock data
```

---

## Pipeline

```
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

---

## Related work

**SampoNLP** — morphological lexicon toolkit used in the paper's dataset pipeline:

- Code: https://github.com/AragonerUA/SampoNLP
- Paper: Chelombitko, I., Chelombitko, E., & Komissarov, A. (2025). *SampoNLP: A Self-Referential Toolkit for Morphological Analysis of Subword Tokenizers.* Proceedings of the 10th International Workshop on Computational Linguistics for Uralic Languages (IWCLUL 2025), pp. 57–67. https://aclanthology.org/2025.iwclul-1.8/

---

## Authors

Vaibhav Rathore, Siddhant Gole, Dadhichi Telwadkar, Rooshil Bhatia, Maulik Ruparel, Siddharth Sureka, Neha Bhargava — Motilal Oswal Financial Services Ltd. & IIT Bombay.
