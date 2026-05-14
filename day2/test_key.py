import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

load_dotenv()

QUESTION = "In one sentence, what is the biggest AI opportunity for banks?"

print("\n" + "="*50)

# ── ANTHROPIC (Claude) ──
try:
    ant = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    r = ant.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}]
    )
    print("✅ CLAUDE:", r.content[0].text)
except Exception as e:
    print("❌ CLAUDE ERROR:", e)

print("="*50)

# ── GROQ (Llama) ──
try:
    groq = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    r = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",   # ← updated model name
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}]
    )
    print("✅ GROQ (Llama):", r.choices[0].message.content)
except Exception as e:
    print("❌ GROQ ERROR:", e)

print("="*50)


# ── OPENROUTER (free access to 10+ models) ──
try:
    openrouter = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    r = openrouter.chat.completions.create(
        #model="google/gemini-2.0-flash-exp:free",
        #model="meta-llama/llama-3.3-70b-instruct:free",
        #model="deepseek/deepseek-r1:free",
        model="openrouter/free",
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}]
    )
    print("✅ OPENROUTER (Deepseek free):", r.choices[0].message.content)
except Exception as e:
    print("❌ OPENROUTER ERROR:", e)

# ── MISTRAL ──
try:
    mistral = OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )
    r = mistral.chat.completions.create(
        model="mistral-small-latest",
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}]
    )
    print("✅ MISTRAL:", r.choices[0].message.content)
except Exception as e:
    print("❌ MISTRAL ERROR:", e)

print("="*50)
print("\nDONE — all models tested.")