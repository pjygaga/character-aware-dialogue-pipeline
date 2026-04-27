"""
Re-train the v2 emotion classifier from scratch.

Reproduces the final v2 run from `notebooks/02_hermione_emotion_classifier.ipynb`:
    - Backbone: bert-base-uncased
    - Training data: 478 original Hermione movie lines + 100 LoRA-generated
      samples (re-labeled by GPT-4o-mini), total 578 examples
    - Class-weighted cross-entropy (weights inversely proportional to
      train-set class frequency)
    - lr=2e-5, batch=16, weight_decay=0.01, warmup=10%, dropout=0.1
    - Up to 8 epochs, early stopping on val macro F1, patience=2
    - Saves checkpoint to this directory (overwrites the .gitignored
      model.safetensors)

Expected runtime: ~20 minutes on a T4 GPU, ~5 minutes on RTX 4090.
Final test macro F1 in the original run: 0.453.

Usage:
    python models/emotion_classifier_v2/train_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"
AUG_PATH = REPO_ROOT / "outputs" / "classifier" / "lora_generated_100_labeled.json"
OUTPUT_DIR = Path(__file__).resolve().parent

EMOTION_LABELS = ["determined", "worried", "angry", "happy", "sad", "neutral"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
BACKBONE = "bert-base-uncased"
MAX_LEN = 64


def load_split(name):
    with open(DATA_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_augmentation():
    if not AUG_PATH.exists():
        raise FileNotFoundError(
            f"Augmentation file missing: {AUG_PATH}. "
            "It is produced by notebook 01 (Phase 3.5) + GPT-4o-mini relabeling."
        )
    with open(AUG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def to_hf(records, tokenizer):
    texts = [r["text"] for r in records]
    labels = [LABEL2ID[r["label"]] for r in records]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def compute_class_weights(train_records):
    counts = np.zeros(len(EMOTION_LABELS), dtype=np.float64)
    for r in train_records:
        counts[LABEL2ID[r["label"]]] += 1
    total = counts.sum()
    weights = total / (len(EMOTION_LABELS) * (counts + 1e-6))
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer that applies class-weighted cross-entropy."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


def metrics_fn(eval_pred):
    preds = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def main():
    train = load_split("train") + load_augmentation()
    val = load_split("val")
    test = load_split("test")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    train_ds = to_hf(train, tokenizer)
    val_ds = to_hf(val, tokenizer)
    test_ds = to_hf(test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE,
        num_labels=len(EMOTION_LABELS),
        id2label={i: l for l, i in LABEL2ID.items()},
        label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_train_tmp"),
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    class_weights = compute_class_weights(train)
    print(f"Class weights: {dict(zip(EMOTION_LABELS, class_weights.tolist()))}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    test_results = trainer.evaluate(test_ds)
    print("Test results:", test_results)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
