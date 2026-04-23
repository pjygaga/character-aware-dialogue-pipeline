import pandas as pd
import json

# Load both CSV files
characters = pd.read_csv("Characters.csv", encoding="latin-1")
dialogue = pd.read_csv("Dialogue.csv", encoding="latin-1")

# Step 1: Find Hermione's Character ID
# Boolean indexing: the expression inside the brackets returns a True/False mask for each row,
# and only rows with True are kept. Then we grab the "Character ID" column's first value.
hermione_id = characters[characters["Character Name"] == "Hermione Granger"]["Character ID"].values[0]
print(f"Hermione's Character ID: {hermione_id}")

# Step 2: Filter all of Hermione's lines
hermione_lines = dialogue[dialogue["Character ID"] == hermione_id].copy()
print(f"Before cleaning, Hermione has: {len(hermione_lines)} lines")

# Step 3: Data cleaning
# 3.1 Drop rows where the dialogue text is missing (NaN)
hermione_lines = hermione_lines.dropna(subset=["Dialogue"])

# 3.2 Strip leading/trailing whitespace
hermione_lines["Dialogue"] = hermione_lines["Dialogue"].str.strip()

# 3.3 Count words in each line
# split() breaks the string on whitespace; len() gives the number of tokens
hermione_lines["word_count"] = hermione_lines["Dialogue"].apply(lambda x: len(x.split()))

# 3.4 Remove very short lines (fewer than 4 words usually carry no useful info,
# e.g. "Yes", "Harry!", "Oh no")
hermione_lines = hermione_lines[hermione_lines["word_count"] >= 4]

# 3.5 Deduplicate (keep only one copy of identical lines)
hermione_lines = hermione_lines.drop_duplicates(subset=["Dialogue"])

print(f"After cleaning, Hermione has: {len(hermione_lines)} lines")

# Step 4: Preview a few random lines to sanity-check the data
print("\nRandom 5 lines preview:")
for i, line in enumerate(hermione_lines["Dialogue"].sample(5, random_state=42).values, 1):
    print(f"{i}. {line}")

# Step 5: Quick stats on line length
print(f"\nLine length statistics:")
print(f"  Shortest: {hermione_lines['word_count'].min()} words")
print(f"  Longest:  {hermione_lines['word_count'].max()} words")
print(f"  Average:  {hermione_lines['word_count'].mean():.1f} words")

# Step 6: Save in two formats for downstream use

# 6.1 Save as CSV for easy manual inspection in Excel
hermione_lines.to_csv("hermione_lines_movies.csv", index=False, encoding="utf-8")
print(f"\nSaved: hermione_lines_movies.csv")

# 6.2 Save as JSON for emotion labeling and model training later
# Keep only the fields we need downstream
output_list = []
for idx, row in hermione_lines.iterrows():
    output_list.append({
        "id": int(row["Dialogue ID"]),
        "text": row["Dialogue"],
        "source": "movie",
        "word_count": int(row["word_count"])
    })

with open("hermione_lines_movies.json", "w", encoding="utf-8") as f:
    json.dump(output_list, f, ensure_ascii=False, indent=2)
print(f"Saved: hermione_lines_movies.json")

print("\nDone!")