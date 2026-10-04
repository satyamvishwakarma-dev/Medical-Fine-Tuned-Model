"""
BioMistral Medical Q&A - Streamlit Clinical Web Interface
Interactive medical Q&A assistant powered by fine-tuned BioMistral-7B running on Ollama.
"""

import json
import time
from datetime import datetime
import requests
import streamlit as st

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "biomistral-med"

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BioMistral Clinical Q&A",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Clinical Custom CSS Theme
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Medical Aesthetic */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card */
    .med-header-banner {
        background: linear-gradient(135deg, #0d9488 0%, #06b6d4 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.25);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .med-header-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .med-header-sub {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    
    /* Medical Disclaimer Banner */
    .med-disclaimer {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 0.85rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        line-height: 1.45;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Quick Prompt Pills */
    .prompt-chip-btn {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #0f172a;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        text-decoration: none;
    }

    /* Scrollable chat spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 950px;
    }
    
    /* Streamlit chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
    }
    
    div[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #0d9488 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Diagnostics & Health Check
# -----------------------------------------------------------------------------
def get_ollama_status():
    """Verify Ollama connectivity and model availability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            has_model = any(MODEL_NAME in m for m in models)
            return True, has_model
        return False, False
    except Exception:
        return False, False


ollama_up, model_loaded = get_ollama_status()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 Clinical Parameters")
    
    if ollama_up and model_loaded:
        st.success(f"🟢 **{MODEL_NAME}** Ready")
    elif ollama_up and not model_loaded:
        st.warning(f"🟡 Ollama Active, but `{MODEL_NAME}` missing.")
    else:
        st.error("🔴 Ollama Offline (`localhost:11434`)")
        st.caption("Start the Ollama application on Windows.")

    st.markdown("---")
    temperature = st.slider("Temperature (Focus vs Diversity)", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max Answer Length (Tokens)", 64, 1024, 512, 32)
    top_p = st.slider("Top-P Nucleus Sampling", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.markdown("### ℹ️ Architecture Specs")
    st.caption("**Base:** BioMistral-7B (PubMed PMC pre-trained)")
    st.caption("**Fine-Tuning:** QLoRA 4-bit NF4, r=16")
    st.caption("**Quantization:** Q4_K_M GGUF")
    st.caption("**Mode:** Single-Turn Clinical Inference")

    if st.button("🧹 Clear Chat Consultation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# Main Medical Interface
# -----------------------------------------------------------------------------
st.markdown("""
<div class="med-header-banner">
    <div>
        <div class="med-header-title">🩺 BioMistral Clinical Q&A</div>
        <div class="med-header-sub">Fine-Tuned Medical AI • Local On-Device Inference</div>
    </div>
    <div style="font-size: 0.8rem; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px;">
        Model: biomistral-med
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="med-disclaimer">
    <strong>⚠️ CLINICAL DISCLAIMER:</strong> This AI model is an educational and research demonstrator fine-tuned on PubMed Central and medical QA datasets. It does <em>not</em> constitute medical advice, clinical diagnosis, or emergency decision making. Always consult a licensed physician or healthcare specialist.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Streaming Generator
# -----------------------------------------------------------------------------
def stream_answer(question: str):
    """Send clinical prompt to Ollama and yield streaming tokens."""
    payload = {
        "model": MODEL_NAME,
        "prompt": question.strip(),
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p
        },
    }
    with requests.post(OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break


# -----------------------------------------------------------------------------
# Chat Session Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Exploration Pills when no history exists
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Clinical Inquiries:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Hallmark symptoms of Type 2 Diabetes", use_container_width=True):
            st.session_state.prefill_prompt = "What are the hallmark symptoms and diagnostic criteria for Type 2 Diabetes?"
        if st.button("💊 First-line medication for hypertension", use_container_width=True):
            st.session_state.prefill_prompt = "What are the first-line antihypertensive medications for an adult with stage 1 hypertension and no comorbidities?"
    with col2:
        if st.button("🌡️ Pediatric fever clinical red flags", use_container_width=True):
            st.session_state.prefill_prompt = "What are the clinical red flag signs in pediatric patients presenting with acute fever?"
        if st.button("🧪 Metformin contraindications & lactic acidosis", use_container_width=True):
            st.session_state.prefill_prompt = "Explain the mechanism of action, contraindications, and lactic acidosis risk of Metformin."

# Render Conversation Feed
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if a prompt was prefilled by a quick button
default_prompt = st.session_state.pop("prefill_prompt", None)

# Handle Question Input
question = st.chat_input("Inquire clinical question, symptoms, or medication guidance...") or default_prompt

if question:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question))
        except requests.exceptions.ConnectionError:
            answer = "⚠️ **Connection Error:** Could not reach Ollama at `http://localhost:11434`. Ensure Ollama is running."
            st.error(answer)
        except requests.exceptions.RequestException as err:
            answer = f"⚠️ **Inference Request Failed:** {err}"
            st.error(answer)

    # Save to session history
    st.session_state.messages.append({"role": "assistant", "content": answer})
