import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

print("=" * 55)
print("ZERO-SHOT CLASSIFIER — HuggingFace")
print("=" * 55)

# ── LOAD THE MODEL ──
# facebook/bart-large-mnli — understands relationships
# between text and labels without being trained on them
print("\nLoading zero-shot model...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("Model ready.\n")

# ── YOUR AI USE CASE TAXONOMY ──
# Your exact 8 solutioning categories from the taxonomy exercise
# The model has never seen these — that is what makes it zero-shot
AI_TAXONOMY = [
    "Classify",
    "Generate",
    "Retrieve RAG",
    "Predict",
    "Recommend",
    "Detect",
    "Automate",
    "Personalise",
]

# ── CLIENT PAIN POINTS TO CLASSIFY ──
# Real enterprise pain points — the kind you see at NTT DATA
pain_points = [
    "Our fraud team manually reviews 2,000 transactions per day and misses 40% of suspicious activity",
    "Customers keep churning after 3 months and we don't know why until it's too late",
    "HR agents spend 3 hours daily answering the same policy questions from employees",
    "We want every customer to see a homepage tailored specifically to their preferences",
    "The marketing team produces 200 product descriptions per week — each takes 45 minutes",
    "Our quality control team inspects 500 units per hour but has a 3% defect escape rate",
]

# ── RUN CLASSIFIER ON EACH PAIN POINT ──
for i, pain_point in enumerate(pain_points, 1):
    print(f"Pain Point {i}:")
    print(f"  \"{pain_point[:80]}...\"" if len(pain_point) > 80 else f"  \"{pain_point}\"")

    result = classifier(pain_point, candidate_labels=AI_TAXONOMY)

    # Top 3 matches only — keeps output clean
    print(f"\n  {'CATEGORY':<20} {'SCORE':<10} {'BAR'}")
    print(f"  {'-'*45}")
    for label, score in zip(result["labels"][:3], result["scores"][:3]):
        bar   = "█" * int(score * 30)
        arrow = " ← BEST MATCH" if label == result["labels"][0] else ""
        print(f"  {label:<20} {score:.1%}     {bar}{arrow}")
    print()

print("=" * 55)
print("✅ Done. Model: facebook/bart-large-mnli")
print("Taxonomy: Your 8-category AI Use Case Framework")
print("=" * 55)