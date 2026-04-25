"""
Run the three baseline prompt variants (A/B/C) against all test questions.
Produces one CSV row per (prompt, question) pair: 3 * 30 = 90 rows.

Inputs:
  src/baseline/prompt_A.txt
  src/baseline/prompt_B.txt
  src/baseline/prompt_C.txt
  src/baseline/test_questions.json

Output:
  outputs/baseline/baseline_responses.csv
"""

import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load OPENAI_API_KEY from the .env file at repo root
load_dotenv()
client = OpenAI()

# Resolve paths relative to the repo root (this script lives in src/baseline/)
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "src" / "baseline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "baseline" / "baseline_responses.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Model settings
MODEL = "gpt-4o-mini"
MAX_TOKENS = 200
TEMPERATURE = 0.7

# Load test questions
with open(BASELINE_DIR / "test_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# Load the three prompt variants
prompts = {}
for version in ["A", "B", "C"]:
    with open(BASELINE_DIR / f"prompt_{version}.txt", "r", encoding="utf-8") as f:
        prompts[version] = f.read()

# Run every (prompt, question) pair: 3 prompts * 30 questions = 90 API calls
results = []
total = len(prompts) * len(questions)
with tqdm(total=total, desc="Generating responses") as bar:
    for version, system_prompt in prompts.items():
        for q in questions:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q["question"]},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content.strip()
            results.append({
                "prompt_version": version,
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "target_emotion": q["target_emotion"],
                "answer": answer,
            })
            bar.update(1)

# Save CSV
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results)} responses to {OUTPUT_PATH}")
