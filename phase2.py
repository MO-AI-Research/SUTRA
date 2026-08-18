import re
from typing import Dict, Tuple, Set

class SUTRATrainer:
    def __init__(self, target_vocab_size: int, gamma_start: float = 3.0, gamma_end: float = 0.0):
        self.target_size = target_vocab_size
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.merges = {}

    def _get_dynamic_gamma(self, current_size: int, start_size: int) -> float:
        """Linear anneal from gamma_start to gamma_end over training."""
        if self.target_size == start_size:
            return self.gamma_end

        progress = (current_size - start_size) / (self.target_size - start_size)
        current_gamma = self.gamma_start - (progress * (self.gamma_start - self.gamma_end))
        return max(current_gamma, 0.0)

    def _get_stats(self, vocab: Dict[str, int], boundary_map: Dict[str, Set[int]]) -> Tuple[Dict, Dict]:
        """Return (pair_freqs, pair_conflicts)."""
        pairs = {}
        conflicts = {}

        for word_str, freq in vocab.items():
            symbols = word_str.split()

            # "u n a v a i l" -> "unavail"
            original_word = "".join(symbols).replace("</w>", "")
            forbidden_indices = boundary_map.get(original_word, set())

            cursor = 0
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])

                pairs[pair] = pairs.get(pair, 0) + freq

                if forbidden_indices:
                    split_point = cursor + len(symbols[i])
                    if split_point in forbidden_indices:
                        conflicts[pair] = conflicts.get(pair, 0) + freq

                cursor += len(symbols[i])

        return pairs, conflicts

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)

        new_vocab = {}
        for word, freq in vocab.items():
            w_out = p.sub(replacement, word)
            new_vocab[w_out] = freq

        return new_vocab

    def train(self, initial_vocab: Dict[str, int], boundary_map: Dict[str, Set[int]]):
        vocab = initial_vocab.copy()

        unique_tokens = set()
        for w in vocab.keys():
            unique_tokens.update(w.split())
        start_vocab_size = len(unique_tokens)

        curr_vocab_size = start_vocab_size
        print(f"--- Starting Training (V_start: {start_vocab_size}, Target: {self.target_size}) ---")

        while curr_vocab_size < self.target_size:
            gamma = self._get_dynamic_gamma(curr_vocab_size, start_vocab_size)
            pair_freqs, pair_conflicts = self._get_stats(vocab, boundary_map)

            if not pair_freqs:
                print("No more pairs to merge. Stopping early.")
                break

            best_pair = None
            max_score = -1.0

            for pair, freq in pair_freqs.items():
                conflict_count = pair_conflicts.get(pair, 0)
                psi = 1.0 - (conflict_count / freq)  # boundary validity
                score = freq * (psi ** gamma)

                if score > max_score:
                    max_score = score
                    best_pair = pair

            vocab = self._merge_vocab(best_pair, vocab)

            self.merges[best_pair] = len(self.merges) + 1
            curr_vocab_size += 1

            if curr_vocab_size % 100 == 0:
                print(f"Iter: {len(self.merges)} | Vocab: {curr_vocab_size} | Gamma: {gamma:.4f} | Merged: {best_pair} (Psi: {1.0 - (pair_conflicts.get(best_pair,0)/pair_freqs[best_pair]):.2f})")

        print(f"✅ Training Complete. Final Vocab Size: {curr_vocab_size}")
        return vocab, self.merges

if __name__ == "__main__":
    mock_vocab = {
        "u n a v a i l a b i l i t y </w>": 100,
        "d i s a b i l i t y </w>": 50,
        "a b i l i t y </w>": 200
    }

    # split indices: un|avail|abil|ity, dis|abil|ity, abil|ity
    mock_boundaries = {
        "unavailability": {2, 7, 11},
        "disability": {3, 7},
        "ability": {4}
    }

    trainer = SUTRATrainer(target_vocab_size=15, gamma_start=3.0, gamma_end=0.0)
    final_vocab, merge_rules = trainer.train(mock_vocab, mock_boundaries)

    print("\n--- Final Vocab Sample ---")
    for w, f in final_vocab.items():
        print(f"{w} : {f}")
