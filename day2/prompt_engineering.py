import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Starting...")

# ── SYSTEM PROMPTS ──
def ask_with_system(system_prompt, user_message):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

formal_system = """You are a senior banking advisor at a private bank
in Singapore. Speak formally and use financial terminology."""

casual_system = """You are a friendly assistant who explains finance
in simple everyday language. Short sentences. No jargon."""

question = "Should I put my savings in a fixed deposit or stocks?"

print("=" * 50)
print("FORMAL ADVISOR:")
print(ask_with_system(formal_system, question))

print("=" * 50)
print("CASUAL ASSISTANT:")
print(ask_with_system(casual_system, question))

print("DONE")

# ── FEW-SHOT PROMPTING ──
# Few-shot = giving the AI examples of what you want
# before asking your real question.
# The AI learns the pattern from your examples.
# ── FEW-SHOT PROMPTING ──
# Few-shot = giving the AI examples of what you want
# before asking your real question.
# The AI learns the pattern from your examples.

print("\n" + "=" * 50)
print("FEW-SHOT PROMPTING:")

def few_shot_classifier(customer_message):
    """Classify customer sentiment using examples."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[
            # Example 1 — show the AI what you want
            {"role": "user", "content": "My order arrived broken and customer service ignored me."},
            {"role": "assistant", "content": "SENTIMENT: Negative | URGENCY: High | ACTION: Escalate to manager"},

            # Example 2
            {"role": "user", "content": "Package came on time, everything perfect, love this service!"},
            {"role": "assistant", "content": "SENTIMENT: Positive | URGENCY: Low | ACTION: Send thank you"},

            # Example 3
            {"role": "user", "content": "Still waiting for my refund after 2 weeks, very frustrated."},
            {"role": "assistant", "content": "SENTIMENT: Negative | URGENCY: High | ACTION: Process refund immediately"},

            # Now the real question — AI follows the pattern
            {"role": "user", "content": customer_message}
        ]
    )
    return response.content[0].text

    # Test it on new messages it has never seen
test_messages = [
    "I have been waiting 3 days and my package hasn't arrived.",
    "Just received my order, absolutely fantastic quality!",
    "The app keeps crashing every time I try to checkout.",
]

for msg in test_messages:
    print(f"\nCustomer: {msg}")
    print(f"AI:       {few_shot_classifier(msg)}")