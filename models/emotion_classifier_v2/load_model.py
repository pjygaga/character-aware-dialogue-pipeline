"""
Load the fine-tuned Hermione emotion classifier (BERT base, 6 classes).

The 440 MB weights file `model.safetensors` is excluded from git. Two ways to
obtain it:
  (1) Re-train via `python models/emotion_classifier_v2/train_classifier.py`
      (~20 minutes on a T4 GPU, ~5 minutes on a 4090).
  (2) Drop an existing checkpoint at the path printed in the error message.

Usage:
    from models.emotion_classifier_v2.load_model import load_classifier, EMOTION_LABELS
    model, tokenizer = load_classifier()
    # model.eval(); inputs = tokenizer(["Honestly, Ronald."], return_tensors="pt", ...)
"""

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = MODEL_DIR / "model.safetensors"

EMOTION_LABELS = ["determined", "worried", "angry", "happy", "sad", "neutral"]


def load_classifier():
    """Return (model, tokenizer) loaded from this directory."""
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing weights at {WEIGHTS_FILE}.\n"
            f"This file (~440 MB) is not committed to git. To obtain it:\n"
            f"  (1) Run: python {MODEL_DIR / 'train_classifier.py'}\n"
            f"      (~20 min on T4, ~5 min on RTX 4090)\n"
            f"  (2) Or place an existing checkpoint at the path above."
        )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_classifier()
    print(f"Loaded classifier with {model.num_labels} labels: {EMOTION_LABELS}")
    print(f"Architecture: {type(model).__name__}")
    print(f"Tokenizer: {type(tokenizer).__name__}")
