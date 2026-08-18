import re
import csv
import json
from collections import Counter

from phase1 import SUTRA_Phase1
from phase2 import SUTRATrainer

CORPUS_PATH = "corpus_small.txt"
CSV_PATH = "/home/sagemaker-user/InterSpeech/hindi_morph_splits.csv"
MODEL_PATH = "/home/sagemaker-user/InterSpeech/byt5_model"
TARGET_VOCAB = 16000
GAMMA_START = 4.0
GAMMA_END = 0.0

def load_csv_dict(path):
    """CSV rows: word, split1, split2, ..."""
    print(f" Loading CSV from {path}...")
    gold_data = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                word = row[0].strip()
                splits = [s.strip() for s in row[1:] if s.strip()]
                gold_data[word] = splits
        print(f"   ✅ Loaded {len(gold_data)} words from CSV.")
        return gold_data
    except Exception as e:
        print(f"   ⚠️ Error loading CSV: {e}")
        return {}


def save_artifacts(final_vocab_freqs, merges_list):
    """Write merges.txt and vocab.json in HuggingFace BPE format."""
    print("\n💾 Generating Artifacts...")

    with open("merges_16k.txt", "w", encoding="utf-8") as f:
        for pair in merges_list:
            f.write(f"{pair[0]} {pair[1]}\n")
    print("   -> Saved merges.txt")

    unique_tokens = set()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for t in special_tokens:
        unique_tokens.add(t)

    # Every merge operand and result must appear in vocab or decode will fail.
    for p1, p2 in merges_list:
        unique_tokens.add(p1)
        unique_tokens.add(p2)
        unique_tokens.add(p1 + p2)

    for word_seq in final_vocab_freqs.keys():
        for t in word_seq.split():
            unique_tokens.add(t)

    sorted_tokens = sorted(list(unique_tokens))

    vocab_map = {}
    next_id = 0

    for t in special_tokens:
        vocab_map[t] = next_id
        next_id += 1

    for t in sorted_tokens:
        if t not in vocab_map:
            vocab_map[t] = next_id
            next_id += 1

    with open("vocab_16k.json", "w", encoding="utf-8") as f:
        json.dump(vocab_map, f, ensure_ascii=False, indent=2)

    print(f"   -> Saved vocab.json (Size: {len(vocab_map)})")

    return vocab_map


def main():
    print(" STARTING SuTRA PILOT EXPERIMENT")

    print("\n--- Phase 1: Setup & Probing ---")
    try:
        tokenizer_p1 = SUTRA_Phase1(model_folder_path=MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load ByT5: {e}")
        return

    csv_data = load_csv_dict(CSV_PATH)
    tokenizer_p1.load_gold_csv(csv_data)

    print(f"📖 Reading corpus: {CORPUS_PATH}...")
    try:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("❌ Corpus file not found.")
        return

    print("   Applying Phonetic Grouping (Regex)...")
    raw_words = re.findall(r'[\u0900-\u097F]+', text)

    vocab = Counter()
    unique_words_to_probe = set()

    print(f"   Processing {len(raw_words)} raw words...")

    for w in raw_words:
        unique_words_to_probe.add(w)

        units = tokenizer_p1.apply_phonetic_grouping(w)
        vocab_key = " ".join(units) + " </w>"
        vocab[vocab_key] += 1

    print(f"   Found {len(vocab)} unique initial token sequences.")
    print(f"   Found {len(unique_words_to_probe)} unique words to probe.")

    print("   Running Morphological Probe...")
    tokenizer_p1.prepare_corpus_constraints(list(unique_words_to_probe))

    boundary_map = tokenizer_p1.boundary_map
    print(f"   Boundary Map Ready: {len(boundary_map)} constraints.")

    print("\n--- Phase 2: Training Loop (Score-Based) ---")
    trainer = SUTRATrainer(
        target_vocab_size=TARGET_VOCAB,
        gamma_start=GAMMA_START,
        gamma_end=GAMMA_END
    )

    final_vocab_freqs, merges_dict = trainer.train(vocab, boundary_map)
    merges_list = list(merges_dict.keys())

    save_artifacts(final_vocab_freqs, merges_list)

    print("\n EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    main()
