"""
Batch emotion labeling for Hermione dialogue lines using GPT-4o-mini.

Input:  data/raw/hermione_lines_movies.json
Output: data/extracted/hermione_labeled.json
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = OpenAI()  # Automatically reads OPENAI_API_KEY from env

# Load the annotation guidelines from the docs folder
# We read this file and include it in every API call as the system prompt
GUIDELINES_PATH = Path("docs/annotation_guidelines.md")
with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
    GUIDELINES = f.read()

# The valid labels (must match what you defined in the guidelines)
VALID_LABELS = {"determined", "worried", "angry", "happy", "sad", "neutral"}

# Paths
INPUT_PATH = Path("data/raw/hermione_lines_movies.json")
OUTPUT_PATH = Path("data/extracted/hermione_labeled.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # Create folder if needed


def label_one_line(line_text: str, max_retries: int = 3) -> str:
    """Send one line to the API and return the emotion label.
    Retries up to 3 times if the API fails or returns an invalid label."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": GUIDELINES},
                    {"role": "user", "content": f"Line: {line_text}\n\nLabel:"}
                ],
                temperature=0,   # Deterministic: same input -> same output
                max_tokens=10    # We only need one word
            )
            
            # Extract the label and clean it
            label = response.choices[0].message.content.strip().lower()
            # Sometimes the model adds punctuation or quotes, strip those
            label = label.strip('.,!?"\' ')
            
            if label in VALID_LABELS:
                return label
            else:
                # Model returned something weird, retry
                print(f"  Attempt {attempt + 1}: got invalid label '{label}', retrying...")
        
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retry
    
    # All retries failed, return neutral as a safe default
    return "neutral"


def main():
    # Load the input data
    print(f"Loading lines from {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = json.load(f)
    print(f"Loaded {len(lines)} lines.")
    
    # Check if we have partial output to resume from
    labeled = []
    start_idx = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            labeled = json.load(f)
        start_idx = len(labeled)
        print(f"Found existing output with {start_idx} entries. Resuming from index {start_idx}.")
    
    # Label each line
    print(f"Labeling lines {start_idx} to {len(lines)}...")
    for i in tqdm(range(start_idx, len(lines)), desc="Labeling"):
        item = lines[i].copy()  # Don't modify the original
        label = label_one_line(item["text"])
        item["emotion"] = label
        labeled.append(item)
        
        # Save progress every 50 items in case something crashes
        if (i + 1) % 50 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(labeled, f, ensure_ascii=False, indent=2)
    
    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    
    # Print label distribution (good sanity check!)
    print("\n=== Label distribution ===")
    from collections import Counter
    counts = Counter(item["emotion"] for item in labeled)
    for label in VALID_LABELS:
        count = counts.get(label, 0)
        pct = count / len(labeled) * 100
        print(f"  {label:12s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\nTotal: {len(labeled)} lines labeled.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()