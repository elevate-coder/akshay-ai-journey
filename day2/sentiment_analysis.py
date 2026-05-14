import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

print("=" * 55)
print("SENTIMENT ANALYSIS — HuggingFace")
print("=" * 55)

# ── LOAD THE MODEL ──
# First run: downloads distilbert model (~67MB) — takes ~30 seconds
# Every run after: loads from cache — instant
print("\nLoading sentiment model...")
sentiment = pipeline("sentiment-analysis")
print("Model ready.\n")

# ── TEST SENTENCES ──
# Mix of positive, negative, and neutral/ambiguous
texts = [
    "This AI solution has transformed our operations completely.",
    "The implementation was a disaster — costs doubled overnight.",
    "The product is okay, nothing special but gets the job done.",
    "Our team is cautiously optimistic about the new platform.",
    "Worst onboarding experience I have ever had in 20 years.",
]

# ── RUN AND DISPLAY ──
print(f"{'TEXT':<52} {'SENTIMENT':<12} {'CONFIDENCE'}")
print("-" * 75)

for text in texts:
    result = sentiment(text)[0]
    label  = result["label"]
    score  = result["score"]
    emoji  = "✅" if label == "POSITIVE" else "❌"
    # Truncate long text for display
    short  = text[:48] + ".." if len(text) > 48 else text
    print(f"{short:<52} {emoji} {label:<10} {score:.2%}")

print("-" * 75)
print("\n✅ Done. Model: distilbert-base-uncased-finetuned-sst-2-english")