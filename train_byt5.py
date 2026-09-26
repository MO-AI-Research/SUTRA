"""
Fine-tune ByT5 as a morpheme segmenter for SuTRA Phase 1.

Input CSV must have two columns:
    Word          - the surface word
    Segmentation  - morphemes joined by a separator (default '+'), e.g. असु+विधा+जनक

The trained model outputs space-delimited morphemes, which is the format
expected by SUTRA_Phase1._predict_with_byt5.

Usage:
    python train_byt5.py --csv data/morph_segmentation.csv --output_dir ./byt5_model
"""

import argparse

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Defaults (override via CLI)
CSV_FILE = "data/morph_segmentation.csv"
MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = "./byt5_model"


def parse_args():
    p = argparse.ArgumentParser(description="Train a ByT5 morpheme segmenter for SuTRA.")
    p.add_argument("--csv", default=CSV_FILE, help="CSV with 'Word' and 'Segmentation' columns")
    p.add_argument("--model_name", default=MODEL_NAME, help="Base HF model")
    p.add_argument("--output_dir", default=OUTPUT_DIR, help="Where to save the fine-tuned model")
    p.add_argument("--separator", default="+", help="Morpheme separator used in the Segmentation column")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--test_size", type=float, default=0.01)
    return p.parse_args()


def make_preprocess(tokenizer, separator, max_length):
    def prepare_data(batch):
        input_texts = batch["Word"]
        target_texts = [seg.replace(separator, " ") for seg in batch["Segmentation"]]

        model_inputs = tokenizer(
            input_texts, max_length=max_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            target_texts, max_length=max_length, truncation=True, padding="max_length"
        )

        # Ignore padding positions in the loss
        pad_id = tokenizer.pad_token_id
        model_inputs["labels"] = [
            [(t if t != pad_id else -100) for t in seq] for seq in labels["input_ids"]
        ]
        return model_inputs

    return prepare_data


def main():
    args = parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.csv)
    print(f"Original size: {len(df)}")
    df = df.dropna(subset=["Word", "Segmentation"])
    df["Word"] = df["Word"].astype(str)
    df["Segmentation"] = df["Segmentation"].astype(str)
    print(f"Cleaned size: {len(df)}")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.train_test_split(test_size=args.test_size)

    print("Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized = dataset.map(
        make_preprocess(tokenizer, args.separator, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,  # ByT5 is unstable in fp16
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"ByT5 model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
