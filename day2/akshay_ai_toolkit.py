import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from transformers import pipeline

import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)

load_dotenv()

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Akshay's AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

# ── SECURITY CONFIG ──
APP_PASSWORD = "akshay2026"
MAX_PROMPTS  = 5

# ── SESSION STATE ──
if "authenticated"  not in st.session_state:
    st.session_state.authenticated = False
if "prompt_count"   not in st.session_state:
    st.session_state.prompt_count = 0
if "locked"         not in st.session_state:
    st.session_state.locked = False
if "sentiment_model" not in st.session_state:
    st.session_state.sentiment_model = None
if "classifier_model" not in st.session_state:
    st.session_state.classifier_model = None

# ── PASSWORD SCREEN ──
if not st.session_state.authenticated:
    st.title("🔐 Akshay's AI Toolkit")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Enter password to access")
        st.caption("3 AI tools — Multi-LLM Comparator · Sentiment Analyser · AI Use Case Classifier")
        password = st.text_input("Password", type="password", placeholder="Enter access password")
        if st.button("Login", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    st.stop()

# ── LOCKED SCREEN ──
if st.session_state.locked:
    st.title("🔒 Session Limit Reached")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.warning(
            f"You have used all {MAX_PROMPTS} queries for this session.\n\n"
            "Contact Akshay Sharma for access."
        )
        st.markdown("📧 akshaysharma2009@gmail.com")
        st.markdown("💼 linkedin.com/in/akshaysharma21")
        st.markdown("💻 github.com/elevate-coder")
    st.stop()

# ── HEADER ──
col_title, col_counter = st.columns([4, 1])
with col_title:
    st.title("🤖 Akshay's AI Toolkit")
    st.caption("Multi-LLM Comparator  ·  Sentiment Analyser  ·  AI Use Case Classifier")
with col_counter:
    remaining = MAX_PROMPTS - st.session_state.prompt_count
    color = "green" if remaining >= 3 else "orange" if remaining == 2 else "red"
    icon  = "🟢" if remaining >= 3 else "🟡" if remaining == 2 else "🔴"
    st.markdown(
        f"<div style='text-align:right;padding:10px;background:#f0f2f6;"
        f"border-radius:8px;margin-top:10px;'>"
        f"<span style='font-size:13px;color:{color};'>{icon} "
        f"<b>{remaining} queries left</b></span></div>",
        unsafe_allow_html=True
    )

st.divider()

# ── TABS ──
tab1, tab2, tab3 = st.tabs([
    "⚡ Multi-LLM Comparator",
    "💬 Sentiment Analyser",
    "🎯 AI Use Case Classifier"
])

# ══════════════════════════════════════════
# TAB 1 — MULTI-LLM COMPARATOR
# ══════════════════════════════════════════
def call_claude(prompt):
    start = time.time()
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        return {"text": r.content[0].text, "time": elapsed,
                "tokens": r.usage.input_tokens + r.usage.output_tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_groq(prompt):
    start = time.time()
    try:
        client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                        base_url="https://api.groq.com/openai/v1")
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        return {"text": r.choices[0].message.content, "time": elapsed,
                "tokens": r.usage.total_tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_mistral(prompt):
    start = time.time()
    try:
        client = OpenAI(api_key=os.getenv("MISTRAL_API_KEY"),
                        base_url="https://api.mistral.ai/v1")
        r = client.chat.completions.create(
            model="mistral-small-latest", max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        return {"text": r.choices[0].message.content, "time": elapsed,
                "tokens": r.usage.total_tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_openrouter(prompt):
    start = time.time()
    try:
        client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                        base_url="https://openrouter.ai/api/v1")
        r = client.chat.completions.create(
            model="openrouter/auto", max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        return {"text": r.choices[0].message.content, "time": elapsed,
                "tokens": r.usage.total_tokens if r.usage else 0, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def show_result(col, model_name, emoji, result, color):
    with col:
        st.markdown(f"### {emoji} {model_name}")
        if result["error"]:
            st.error(f"❌ {result['error'][:200]}")
        else:
            m1, m2 = st.columns(2)
            m1.metric("⏱ Time", f"{result['time']}s")
            m2.metric("🔢 Tokens", result["tokens"])
            st.markdown(
                f"<div style='background:{color};padding:16px;border-radius:8px;"
                f"margin-top:8px;min-height:180px;font-size:14px;line-height:1.6;"
                f"color:#ffffff;'>{result['text'].replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )

with tab1:
    st.markdown("### Ask the same question to 4 AI models simultaneously")
    query = st.text_area("Enter your question", height=100,
                         placeholder="e.g. What is the biggest AI opportunity for banks in 2026?",
                         key="llm_query")
    run_llm = st.button("⚡ Compare All Models", type="primary",
                        use_container_width=True, key="run_llm")

    if run_llm and query.strip():
        st.session_state.prompt_count += 1
        if st.session_state.prompt_count >= MAX_PROMPTS:
            st.session_state.locked = True

        st.divider()
        st.markdown(f"### 📊 Results — Query {st.session_state.prompt_count} of {MAX_PROMPTS}")

        with st.spinner("Asking all 4 models..."):
            results = {
                "Claude":       call_claude(query),
                "Groq (Llama)": call_groq(query),
                "Mistral":      call_mistral(query),
                "OpenRouter":   call_openrouter(query),
            }

        col1, col2, col3, col4 = st.columns(4)
        show_result(col1, "Claude",       "🟣", results["Claude"],       "#1a1a2e")
        show_result(col2, "Groq (Llama)", "🟠", results["Groq (Llama)"], "#1a2e1a")
        show_result(col3, "Mistral",      "🔵", results["Mistral"],      "#1a1a3a")
        show_result(col4, "OpenRouter",   "🟢", results["OpenRouter"],   "#2e1a1a")

        st.divider()
        st.markdown("### ⚡ Speed & Token Summary")
        summary = [
            {"Model": n, "Time (s)": r["time"],
             "Tokens": r["tokens"],
             "Words": len(r["text"].split()) if r["text"] else 0}
            for n, r in results.items() if not r["error"]
        ]
        if summary:
            st.dataframe(summary, use_container_width=True)

        if st.session_state.locked:
            st.error("🔒 Session limit reached. Contact akshaysharma2009@gmail.com")

    elif run_llm and not query.strip():
        st.warning("⚠️ Please enter a question first.")

# ══════════════════════════════════════════
# TAB 2 — SENTIMENT ANALYSER
# ══════════════════════════════════════════
with tab2:
    st.markdown("### Analyse text sentiment — Positive or Negative with confidence score")
    st.caption("Powered by DistilBERT — HuggingFace open-source model")

    # Load model once and cache in session state
    if st.session_state.sentiment_model is None:
        with st.spinner("Loading sentiment model — first time takes 30 seconds..."):
            st.session_state.sentiment_model = pipeline("sentiment-analysis")

    sentiment_input = st.text_area(
        "Enter text to analyse",
        height=120,
        placeholder="e.g. This AI solution has transformed our operations completely.",
        key="sentiment_input"
    )

    # Allow multiple lines — one per row
    run_sentiment = st.button("💬 Analyse Sentiment", type="primary",
                              use_container_width=True, key="run_sentiment")

    if run_sentiment and sentiment_input.strip():
        st.session_state.prompt_count += 1
        if st.session_state.prompt_count >= MAX_PROMPTS:
            st.session_state.locked = True

        lines = [l.strip() for l in sentiment_input.strip().split("\n") if l.strip()]

        st.divider()
        st.markdown(f"### 📊 Results — {len(lines)} text(s) analysed")

        results = []
        for line in lines:
            result = st.session_state.sentiment_model(line)[0]
            results.append({
                "Text": line[:80] + "..." if len(line) > 80 else line,
                "Sentiment": result["label"],
                "Confidence": f"{result['score']:.2%}",
                "Signal": "✅ Positive" if result["label"] == "POSITIVE" else "❌ Negative"
            })

        st.dataframe(results, use_container_width=True)

        # Visual breakdown
        pos = sum(1 for r in results if r["Sentiment"] == "POSITIVE")
        neg = len(results) - pos
        c1, c2, c3 = st.columns(3)
        c1.metric("Total analysed", len(results))
        c2.metric("✅ Positive", pos)
        c3.metric("❌ Negative", neg)

        if st.session_state.locked:
            st.error("🔒 Session limit reached. Contact akshaysharma2009@gmail.com")

    elif run_sentiment and not sentiment_input.strip():
        st.warning("⚠️ Please enter some text first.")

    st.info("💡 Tip: Enter multiple texts on separate lines to analyse them all at once.")

# ══════════════════════════════════════════
# TAB 3 — AI USE CASE CLASSIFIER
# ══════════════════════════════════════════
with tab3:
    st.markdown("### Classify any business pain point into the AI Use Case Taxonomy")
    st.caption("Powered by Facebook BART — Zero-shot classification, no training needed")

    # Load model once and cache in session state
    if st.session_state.classifier_model is None:
        with st.spinner("Loading classifier model — first time takes 60 seconds..."):
            st.session_state.classifier_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )

    # Taxonomy categories
    AI_TAXONOMY = [
        "Classify", "Generate", "Retrieve RAG",
        "Predict", "Recommend", "Detect",
        "Automate", "Personalise"
    ]

    pain_point = st.text_area(
        "Enter a business pain point",
        height=120,
        placeholder="e.g. Our fraud team manually reviews 2,000 transactions per day and misses 40% of suspicious activity",
        key="classifier_input"
    )

    run_classifier = st.button("🎯 Classify Use Case", type="primary",
                               use_container_width=True, key="run_classifier")

    if run_classifier and pain_point.strip():
        st.session_state.prompt_count += 1
        if st.session_state.prompt_count >= MAX_PROMPTS:
            st.session_state.locked = True

        with st.spinner("Classifying against your 8-category AI taxonomy..."):
            result = st.session_state.classifier_model(
                pain_point, candidate_labels=AI_TAXONOMY
            )

        st.divider()
        st.markdown("### 📊 Classification Results")

        # Best match highlighted
        best = result["labels"][0]
        best_score = result["scores"][0]
        st.success(f"**Primary AI Approach: {best}** — {best_score:.1%} confidence")

        # All scores as a table
        scores_data = [
            {
                "AI Category": label,
                "Confidence": f"{score:.1%}",
                "Match": "⭐ Best match" if label == best else
                         "Strong" if score > 0.15 else
                         "Weak" if score > 0.05 else "—"
            }
            for label, score in zip(result["labels"], result["scores"])
        ]
        st.dataframe(scores_data, use_container_width=True)

        # Bar chart
        st.markdown("### 📈 Confidence Scores")
        chart_data = {
            label: score
            for label, score in zip(result["labels"], result["scores"])
        }
        st.bar_chart(chart_data)

        if st.session_state.locked:
            st.error("🔒 Session limit reached. Contact akshaysharma2009@gmail.com")

    elif run_classifier and not pain_point.strip():
        st.warning("⚠️ Please enter a pain point first.")

    st.info("💡 Tip: Try entering real client pain points from your NTT DATA engagements.")

# ── FOOTER ──
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px;'>"
    "Akshay Sharma  ·  AI Builder Intensive 2026  ·  "
    "github.com/elevate-coder  ·  linkedin.com/in/akshaysharma21"
    "</div>",
    unsafe_allow_html=True
)