import re
import torch
from typing import List, Set, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SUTRA_Phase1:
    def __init__(self, model_folder_path: str):
        print(f"--- Initializing SuTRA Phase 1 ---")

        # Matras, vowel signs, nukta, halant, etc.
        self.DEVANAGARI_PATTERN = re.compile(
            r"[\u093E-\u094C\u0962-\u0963\u0901-\u0903\u093C\u094D\u0951-\u0954]+"
        )

        self.gold_data: Dict[str, List[str]] = {}
        self.boundary_map: Dict[str, Set[int]] = {}

        print(f"   Loading ByT5 model from: {model_folder_path} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_folder_path)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"   ✅ Model loaded on {self.device.upper()}")

        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise e

    def apply_phonetic_grouping(self, text: str) -> List[str]:
        """Group each base grapheme with its following combining marks."""
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            char = text[i]
            token = char
            i += 1
            while i < n:
                if self.DEVANAGARI_PATTERN.match(text[i]):
                    token += text[i]
                    i += 1
                else:
                    break
            tokens.append(token)
        return tokens

    def load_gold_csv(self, data: Dict[str, List[str]]):
        self.gold_data = data
        print(f"   ✅ Loaded {len(data)} words from Gold Standard.")

    def _predict_with_byt5(self, word: str) -> List[str]:
        """Predict morpheme splits. Model output must be space-delimited."""
        inputs = self.tokenizer(word, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=3,
                early_stopping=True
            )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        splits = decoded_output.split(" ")
        return [s for s in splits if s]

    def get_boundaries(self, word: str) -> Set[int]:
        if word in self.gold_data:
            splits = self.gold_data[word]
        elif word in self.boundary_map:
            return self.boundary_map[word]
        else:
            splits = self._predict_with_byt5(word)

        reconstructed = "".join(splits)
        if reconstructed != word:
            # Bad prediction — don't constrain this word.
            self.boundary_map[word] = set()
            return set()

        boundaries = set()
        current_idx = 0
        for part in splits[:-1]:
            current_idx += len(part)
            boundaries.add(current_idx)

        self.boundary_map[word] = boundaries
        return boundaries

    def prepare_corpus_constraints(self, unique_words: List[str]):
        """Precompute boundary_map for all corpus words before BPE training."""
        print(f"   --- Probing {len(unique_words)} unique words ---")

        for i, word in enumerate(unique_words):
            self.get_boundaries(word)

            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1} words...", end="\r")

        print("\n   ✅ Morphological Probing Complete.")

if __name__ == "__main__":
    MODEL_PATH = "/home/sagemaker-user/InterSpeech/byt5_model"

    try:
        tokenizer_phase1 = SUTRA_Phase1(model_folder_path=MODEL_PATH)

        test_word = "असुविधाजनक"
        print(f"\nTesting Inference on: {test_word}")

        boundaries = tokenizer_phase1.get_boundaries(test_word)
        print(f"Boundaries Found: {boundaries}")

        visual = list(test_word)
        for b in sorted(list(boundaries), reverse=True):
            visual.insert(b, " | ")
        print(f"Visual Split: {''.join(visual)}")

    except Exception as e:
        print("\n⚠️ Could not load model. Make sure MODEL_PATH is correct.")
        print(f"Error: {e}")
