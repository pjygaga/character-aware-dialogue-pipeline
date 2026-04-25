"""
Rule-based classifier baseline: always predict 'determined'
(the most frequent emotion in Hermione's lines). Evaluates on val.json.

This is a dumb baseline — the trained BERT classifier in later stages must beat it.
"""

import json
from collections import Counter
from pathlib import Path

# Resolve paths relative to the repo root (this script lives in src/classifier/)
REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_PATH = REPO_ROOT / "data" / "processed" / "val.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "classifier" / "rule_baseline_results.txt"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load validation set
with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# Always predict 'determined' — count how many are correct
RULE_PREDICTION = "determined"
correct = 0
for item in val_data:
    if item["label"] == RULE_PREDICTION:
        correct += 1

accuracy = correct / len(val_data)
print(f"Rule-based baseline accuracy: {accuracy:.4f}")
print(f"Correct: {correct}/{len(val_data)}")

# Save the result
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Rule-based baseline (always predict '{RULE_PREDICTION}'): {accuracy:.4f}\n")
    f.write(f"Correct: {correct}/{len(val_data)}\n")

print(f"\nSaved to {OUTPUT_PATH}")

# Also show the full label distribution (useful for the Stage 4 report)
label_dist = Counter(item["label"] for item in val_data)
print("\nLabel distribution in val set:")
for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count} ({count / len(val_data):.1%})")
