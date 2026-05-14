import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

load_dotenv()

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Multi-LLM Comparator",
    page_icon="🤖",
    layout="wide"
)

# ── SECURITY CONFIG ──
APP_PASSWORD = "akshay2026"     # change this to whatever you want
MAX_PROMPTS  = 5                # number of queries allowed per session

# ── SESSION STATE INIT ──
# Session state persists across reruns within the same browser session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0
if "locked" not in st.session_state:
    st.session_state.locked = False

# ── PASSWORD SCREEN ──
if not st.session_state.authenticated:
    st.title("🔐 Multi-LLM Comparator")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Enter password to access")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter access password"
        )
        if st.button("Login", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
    st.stop()  # stop here if not authenticated

# ── LOCKED SCREEN ──
if st.session_state.locked:
    st.title("🔒 Session Limit Reached")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.warning(
            f"You have used all {MAX_PROMPTS} of your allowed queries for this session.\n\n"
            "Please contact Akshay Sharma to request access."
        )
        st.markdown("📧 akshaysharma2009@gmail.com")
        st.markdown("💼 linkedin.com/in/akshaysharma21")
    st.stop()  # stop here if locked

# ── HEADER ──
st.title("🤖 Multi-LLM Comparator")
st.caption("Ask the same question to 4 AI models simultaneously and compare results")

# ── PROMPT COUNTER ──
remaining = MAX_PROMPTS - st.session_state.prompt_count
col_title, col_counter = st.columns([4, 1])
with col_counter:
    color = "green" if remaining >= 3 else "orange" if remaining == 2 else "red"
    st.markdown(
        f"<div style='text-align:right; padding:8px; "
        f"background:#f0f2f6; border-radius:8px;'>"
        f"<span style='font-size:12px; color:{color};'>"
        f"{'🟢' if remaining >= 3 else '🟡' if remaining == 2 else '🔴'} "
        f"<b>{remaining} queries remaining</b></span></div>",
        unsafe_allow_html=True
    )

st.divider()

# ── QUERY INPUT ──
query = st.text_area(
    "Enter your question",
    placeholder="e.g. What is the biggest AI opportunity for banks in 2026?",
    height=100,
    disabled=st.session_state.locked
)

run = st.button(
    "⚡ Compare All Models",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.locked
)

# ── MODEL CALLER FUNCTIONS ──
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
        text = r.content[0].text
        tokens = r.usage.input_tokens + r.usage.output_tokens
        return {"text": text, "time": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_groq(prompt):
    start = time.time()
    try:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        text = r.choices[0].message.content
        tokens = r.usage.total_tokens
        return {"text": text, "time": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_mistral(prompt):
    start = time.time()
    try:
        client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1"
        )
        r = client.chat.completions.create(
            model="mistral-small-latest",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        text = r.choices[0].message.content
        tokens = r.usage.total_tokens
        return {"text": text, "time": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

def call_openrouter(prompt):
    start = time.time()
    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        r = client.chat.completions.create(
            model="openrouter/free",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = round(time.time() - start, 2)
        text = r.choices[0].message.content
        tokens = r.usage.total_tokens if r.usage else 0
        return {"text": text, "time": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        return {"text": None, "time": None, "tokens": None, "error": str(e)}

# ── RESULT CARD ──
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
                f"""<div style='background-color:{color};
                            padding:16px;
                            border-radius:8px;
                            margin-top:8px;
                            min-height:200px;
                            font-size:14px;
                            line-height:1.6;
                            color:#ffffff;'>
                {result["text"].replace(chr(10), "<br>")}
                </div>""",
                unsafe_allow_html=True
            )

# ── RUN COMPARISON ──
if run and query.strip():

    # Increment counter
    st.session_state.prompt_count += 1

    # Check if this was the last allowed prompt
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
    show_result(col1, "Claude",        "🟣", results["Claude"],        "#1a1a2e")
    show_result(col2, "Groq (Llama)",  "🟠", results["Groq (Llama)"],  "#1a2e1a")
    show_result(col3, "Mistral",       "🔵", results["Mistral"],       "#1a1a3a")
    show_result(col4, "OpenRouter",    "🟢", results["OpenRouter"],    "#2e1a1a")

    st.divider()
    st.markdown("### ⚡ Speed & Token Summary")
    summary = []
    for name, r in results.items():
        if not r["error"]:
            summary.append({
                "Model": name,
                "Time (s)": r["time"],
                "Tokens Used": r["tokens"],
                "Words": len(r["text"].split()) if r["text"] else 0
            })
    if summary:
        st.dataframe(summary, use_container_width=True)

    # Show lock warning if last query just used
    if st.session_state.locked:
        st.error(
            "🔒 You have used all 5 queries. "
            "This session is now locked. "
            "Contact akshaysharma2009@gmail.com for access."
        )

elif run and not query.strip():
    st.warning("⚠️ Please enter a question first.")