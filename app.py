import json
 
import requests
import streamlit as st
 
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "biomistral-med"
 
st.set_page_config(page_title="BioMistral Medical Q&A", page_icon="🩺")

st.title("Medical Q&A")
st.caption(
    "Not medical advice - always consult a healthcare professional.<br>"
    "This is for educational purposes only.<br>"
    "DO NOT USE THIS FOR MEDICAL ADVICE OR TREATMENT.",
    unsafe_allow_html=True
)
 
# Sidebar settings
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max answer length (tokens)", 64, 512, 256, 32)
    if st.button("Clear chat"):
        st.session_state.messages = []
 
 
def stream_answer(question: str):
    """Send the question to Ollama and yield the answer piece by piece."""
    payload = {
        "model": MODEL_NAME,
        "prompt": question,  # Ollama wraps it in the training template
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break
 
 
# Keep chat history for display
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# Handle a new question
if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append([
        {"role": "system", "content": "You are a helpful medical assistant. Your name is Mr. Doctor. Your purpose is to help people with their medical questions. Answer questions based on the provided context and your medical knowledge. If you don't know the answer, say you don't know. Do not provide medical advice or instructions. Always recommend consulting a healthcare professional for medical concerns. If user is asking for medical advice, respond with a disclaimer. If user is greeting you, respond with a greeting only, Don't give random information. If user is asking for your name, respond with Mr. Doctor. If user is asking for your purpose, respond with your purpose is to help people with their medical questions."},
        {"role": "user", "content": question}
        ])
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question))
        except requests.exceptions.ConnectionError:
            answer = "Could not reach Ollama. Make sure the Ollama app is running."
            st.error(answer)
        except requests.exceptions.RequestException as err:
            answer = f"Request failed: {err}"
            st.error(answer)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
 


