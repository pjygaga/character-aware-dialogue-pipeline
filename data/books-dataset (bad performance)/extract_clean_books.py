import pandas as pd
import re
import json
import random

# Load the clean dataset
book = pd.read_csv("harry_potter_books_clean.csv", encoding="utf-8")
print(f"Total rows: {len(book)}")
print(f"Unique books: {book['book'].unique().tolist()}")

# Step 1: Concatenate all fragments within the same chapter
# Group by (book, chapter), then join the text pieces with a space
print("\nConcatenating fragments into full chapter text...")
chapters = book.groupby(["book", "chapter"], sort=False)["text"].apply(
    lambda fragments: " ".join(str(f) for f in fragments)
).reset_index()
chapters.columns = ["book", "chapter", "full_text"]
print(f"Got {len(chapters)} full chapters")

# Quick sanity check: print a snippet from chapter 6 of book 1
sample_chap = chapters[chapters["chapter"] == "chap-6"].iloc[0]
print(f"\nSample from {sample_chap['book']} {sample_chap['chapter']}:")
snippet = sample_chap["full_text"][:500]
print(snippet)

# Step 2: Build regex patterns using backtick as the quote character
# In this dataset, dialogue is wrapped in backticks: `text`
# Note: backtick is a single character, so we use `([^`]+?)` to match minimally
#
# Speaker verbs (common dialogue verbs in HP books)
speech_verbs = r"(?:said|cried|whispered|shouted|gasped|replied|asked|answered|muttered|moaned|sighed|snapped|hissed|breathed|groaned|squeaked|wailed|shrieked|yelled|called|murmured|sobbed|stammered|beamed|snorted|began|added|agreed|continued|admitted|declared|protested|insisted|announced|explained|repeated|exclaimed|urged|scolded|pleaded|retorted|interrupted|corrected|supplied|told|remarked|observed|noted|responded|countered|complained|grumbled|moaned|panted)"

# Optional adverb: "said Hermione happily" / "said Hermione angrily"
# \s+\w+ly is an optional adverb right after speaker name
adverb = r"(?:\s+\w+ly)?"

# Pattern A: `quote,` said Hermione  (quote before speaker tag)
pattern_a = rf"`([^`]+?)[,.!?]*`\s+{speech_verbs}{adverb}\s+Hermione"

# Pattern B: Hermione said, `quote`  (speaker tag before quote)
pattern_b = rf"Hermione{adverb}\s+{speech_verbs}[,\s]+`([^`]+?)`"

# Pattern C: `quote,` Hermione said  (speaker name before verb, quote first)
pattern_c = rf"`([^`]+?)[,.!?]*`\s+Hermione\s+{speech_verbs}"

all_lines = []

for idx, row in chapters.iterrows():
    text = row["full_text"]
    book_name = row["book"]
    chapter_name = row["chapter"]
    
    matches_a = re.findall(pattern_a, text)
    matches_b = re.findall(pattern_b, text)
    matches_c = re.findall(pattern_c, text)
    
    for m in matches_a + matches_b + matches_c:
        line = m.strip()
        # Remove stray quotes, extra whitespace, etc.
        line = re.sub(r"\s+", " ", line)
        
        word_count = len(line.split())
        # Filter: too short (no info) or too long (probably a regex mismatch)
        if word_count < 4 or word_count > 150:
            continue
        
        all_lines.append({
            "text": line,
            "source": "book",
            "book": book_name,
            "chapter": chapter_name,
            "word_count": word_count
        })

print(f"\nTotal raw matches: {len(all_lines)}")

# Dedup by text
seen = set()
unique_lines = []
for item in all_lines:
    if item["text"] not in seen:
        seen.add(item["text"])
        unique_lines.append(item)

print(f"After dedup: {len(unique_lines)}")

# Per-book breakdown
print("\nPer-book breakdown:")
book_counts = {}
for item in unique_lines:
    book_counts[item["book"]] = book_counts.get(item["book"], 0) + 1
for b, c in sorted(book_counts.items()):
    print(f"  {b}: {c} lines")

# Assign IDs
for i, item in enumerate(unique_lines):
    item["id"] = 10000 + i

# Preview
print("\n15 random samples:")
random.seed(42)
for i, item in enumerate(random.sample(unique_lines, min(15, len(unique_lines))), 1):
    print(f"{i}. [{item['book'][:15]}] {item['text']}")

# Length stats
word_counts = [item["word_count"] for item in unique_lines]
print(f"\nLength stats:")
print(f"  Shortest: {min(word_counts)} words")
print(f"  Longest:  {max(word_counts)} words")
print(f"  Average:  {sum(word_counts)/len(word_counts):.1f} words")

# Save
with open("hermione_lines_books.json", "w", encoding="utf-8") as f:
    json.dump(unique_lines, f, ensure_ascii=False, indent=2)

df_out = pd.DataFrame(unique_lines)
df_out.to_csv("hermione_lines_books.csv", index=False, encoding="utf-8")

print(f"\nSaved: hermione_lines_books.json")
print(f"Saved: hermione_lines_books.csv")
print("Done!")