import pandas as pd
import re

book = pd.read_csv("harry_potter_books_clean.csv", encoding="utf-8")

# Concatenate all fragments per chapter
chapters = book.groupby(["book", "chapter"], sort=False)["text"].apply(
    lambda fragments: " ".join(str(f) for f in fragments)
).reset_index()
chapters.columns = ["book", "chapter", "full_text"]

# --- Check 1: Is Order of the Phoenix even in the data? ---
print("=== Check 1: Book 5 content ===")
book5 = chapters[chapters["book"].str.contains("Phoenix")]
print(f"Chapters in Book 5: {len(book5)}")
if len(book5) > 0:
    sample_text = book5.iloc[0]["full_text"]
    print(f"First chapter length: {len(sample_text)} chars")
    print(f"Mentions of 'Hermione' in Book 5: {sum(ch['full_text'].count('Hermione') for _, ch in book5.iterrows())}")
    print(f"Sample snippet (first 500 chars):")
    print(sample_text[:500])
    
    # Check: does Book 5 use backticks for dialogue?
    backtick_count = sum(ch["full_text"].count("`") for _, ch in book5.iterrows())
    doublequote_count = sum(ch["full_text"].count('"') for _, ch in book5.iterrows())
    print(f"\nBacktick count in Book 5: {backtick_count}")
    print(f"Double quote count in Book 5: {doublequote_count}")

# --- Check 2: Why are our matches so short? Let's see what a long Hermione paragraph looks like ---
print("\n\n=== Check 2: What does a Hermione paragraph look like in Book 4? ===")
book4_text = chapters[chapters["book"].str.contains("Goblet")].iloc[0]["full_text"]

# Find first occurrence of "said Hermione" in book 4
idx = book4_text.find("said Hermione")
if idx > 0:
    # Show 400 chars before and 100 chars after
    start = max(0, idx - 400)
    end = min(len(book4_text), idx + 100)
    print(f"Raw text around 'said Hermione':")
    print(repr(book4_text[start:end]))

# --- Check 3: Backtick pattern might have issue with nested quotes ---
print("\n\n=== Check 3: Find a complex Hermione quote manually ===")
# Search for "honestly" which is Hermione's signature word
for _, row in chapters.iterrows():
    text = row["full_text"]
    matches = re.finditer(r"[Hh]onestly", text)
    for m in matches:
        start = max(0, m.start() - 150)
        end = min(len(text), m.end() + 150)
        snippet = text[start:end]
        if "Hermione" in snippet:
            print(f"\nFound in {row['book']} {row['chapter']}:")
            print(repr(snippet[:400]))
            break
    else:
        continue
    break