import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Fix the random seed so the split is reproducible
random.seed(42)

# Resolve paths relative to the repo root (this script lives in src/data/)
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "extracted" / "hermione_labeled_v2.json"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "hermione_dataset_lora.json"
CLASSIFIER_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "hermione_dataset_classifier.json"
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.json"
VAL_PATH = REPO_ROOT / "data" / "processed" / "val.json"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.json"
LORA_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "lora_train.json"
LORA_VAL_PATH = REPO_ROOT / "data" / "processed" / "lora_val.json"

# Load data
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Total entries: {len(raw_data)}")

# Count entries per emotion
emotion_counts = Counter(item["emotion"] for item in raw_data)
print(emotion_counts)

# Convert to LoRA format
lora_data = []
for item in raw_data:
    lora_entry = {
        "instruction": "Respond as Hermione Granger in her characteristic voice and style.",
        "input": f"[Emotional context: {item['emotion']}]",
        "output": item["text"],
        "emotion": item["emotion"],
    }
    lora_data.append(lora_entry)

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(lora_data, f, indent=2, ensure_ascii=False)

print(f"LoRA dataset generated: {len(lora_data)} entries")
print("First entry:")
print(json.dumps(lora_data[0], indent=2, ensure_ascii=False))

# Convert to classifier format
classifier_data = []
for item in raw_data:
    classifier_entry = {
        "text": item["text"],
        "label": item["emotion"],
    }
    classifier_data.append(classifier_entry)

# Save
with open(CLASSIFIER_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(classifier_data, f, indent=2, ensure_ascii=False)

print(f"Classifier dataset generated: {len(classifier_data)} entries")
print("First entry:")
print(json.dumps(classifier_data[0], indent=2, ensure_ascii=False))


def stratified_split(data, label_key="label", train_ratio=0.8, val_ratio=0.1):
    # 1. Group items by their label
    groups = defaultdict(list)
    for item in data:
        groups[item[label_key]].append(item)

    train, val, test = [], [], []

    # 2. Split each group separately so every label keeps the same ratio
    for label, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # 3. Shuffle each set so labels are mixed together
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# Split the classifier dataset
train_set, val_set, test_set = stratified_split(classifier_data, label_key="label")

print(f"Train: {len(train_set)} entries")
print(f"Val:   {len(val_set)} entries")
print(f"Test:  {len(test_set)} entries")

# Save the three splits
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)
with open(VAL_PATH, "w", encoding="utf-8") as f:
    json.dump(val_set, f, indent=2, ensure_ascii=False)
with open(TEST_PATH, "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2, ensure_ascii=False)


def print_distribution(name, dataset, label_key="label"):
    # Count how many items per label, and show the percentage
    counts = Counter(item[label_key] for item in dataset)
    total = len(dataset)
    print(f"\n{name} ({total} entries):")
    for emotion, cnt in sorted(counts.items()):
        print(f"  {emotion}: {cnt} ({cnt / total * 100:.1f}%)")


# Show the label distribution of each split
print_distribution("Train", train_set)
print_distribution("Val", val_set)
print_distribution("Test", test_set)

# Split the LoRA dataset 90/10 (no test set needed, evaluation uses hand-written questions)
# Note: LoRA entries use 'emotion' as the label key, not 'label'
lora_train, lora_val, _ = stratified_split(
    lora_data,
    label_key="emotion",
    train_ratio=0.9,
    val_ratio=0.1,
)

print(f"\nLoRA Train: {len(lora_train)} entries")
print(f"LoRA Val:   {len(lora_val)} entries")

# Save
with open(LORA_TRAIN_PATH, "w", encoding="utf-8") as f:
    json.dump(lora_train, f, indent=2, ensure_ascii=False)
with open(LORA_VAL_PATH, "w", encoding="utf-8") as f:
    json.dump(lora_val, f, indent=2, ensure_ascii=False)

# Check the label distribution
print_distribution("LoRA Train", lora_train, label_key="emotion")
print_distribution("LoRA Val", lora_val, label_key="emotion")
