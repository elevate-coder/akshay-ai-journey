import os
from dotenv import load_dotenv
import openai
import anthropic


#load your api keys from .env file
load_dotenv()

#the same question sent to both models
PROMPT = "In one sentence what is the biggest opportunity for banks in 2026"

# ── Claude ──
ant = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
claude_resp = ant.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": PROMPT}]
)


print("\n🟣 Claude says:")
print(claude_resp.content[0].text)

print("\n✅ Your first LLM call is working!")

