import pandas as pd

# Try UTF-8 first since the source says it's UTF-8
try:
    book = pd.read_csv("harry_potter_books_clean.csv", encoding="utf-8")
    print("Loaded with UTF-8 encoding")
except UnicodeDecodeError:
    book = pd.read_csv("harry_potter_books_clean.csv", encoding="latin-1")
    print("Fallback: loaded with latin-1 encoding")

print(f"\nTotal rows: {len(book)}")
print(f"Columns: {book.columns.tolist()}")

print("\nFirst 10 rows:")
print(book.head(10))

print("\nLast 5 rows:")
print(book.tail(5))

# Check what the text column looks like
text_col = None
for candidate in ["text", "Text", "content", "sentence"]:
    if candidate in book.columns:
        text_col = candidate
        break
if text_col is None:
    text_col = book.columns[-1]  # last column is usually the content

print(f"\nUsing text column: {text_col}")

# Show some samples that mention Hermione
hermione_mentions = book[book[text_col].astype(str).str.contains("Hermione", na=False)]
print(f"\nRows mentioning Hermione: {len(hermione_mentions)}")
print("\nFirst 5 rows mentioning Hermione:")
for i, row in hermione_mentions.head(5).iterrows():
    print(f"\n--- Row {i} ---")
    for col in book.columns:
        val = str(row[col])
        print(f"{col}: {val[:250]}")

# Check quote characters
all_text = " ".join(book[text_col].astype(str).head(100).values)
print("\n\nQuote characters in first 100 rows:")
for char in ['"', '\u201c', '\u201d', "'", '\u2018', '\u2019']:
    count = all_text.count(char)
    if count > 0:
        print(f"  {repr(char)} appears {count} times")