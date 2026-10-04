# 🩺 BioMistral Medical Q&A (Fine-Tuned Local Assistant)

A domain-adapted medical Q&A LLM running fully locally on Windows with an interactive, reactive, scrollable clinical web interface.

> [!WARNING]
> **Educational & Research Demonstrator Only:** This AI model is an educational prototype and does **not** provide formal medical advice, clinical diagnosis, or emergency decision making. Always consult a licensed healthcare professional.

---

## 🔬 Model & Training Pipeline

- **Base Model:** `BioMistral/BioMistral-7B` (Mistral-7B adapted to the biomedical domain on PubMed Central)
- **Method:** QLoRA supervised fine-tuning (4-bit NF4 base + LoRA adapters $r=16, \alpha=32$)
- **Dataset:** `lavita/medical-qa-datasets` (`all-processed`, 20,000 samples)
- **Quantization:** Converted to GGUF format and quantized to `Q4_K_M` (~4.4 GB)
- **Inference Runtime:** Local [Ollama](https://ollama.ai) engine with GPU/CPU memory offloading (~50% GPU / ~50% CPU on 4GB VRAM)

### Prompt Template
```
### Instruction:
{instruction}

### Input:


### Response:
{answer}
```
*Note: The model is trained on single-turn Q&A. The Ollama `Modelfile` manages template wrapping, so apps send the direct question.*

---

## 🚀 Running the Frontend Applications

Make sure Ollama is running and has the model registered:
```powershell
ollama list    # Verify 'biomistral-med' is listed
```

### Option 1: Modern Reactive Clinical Web App (Recommended)
Features real-time token streaming (SSE), clinical aesthetic with ECG heartbeat rhythm, scrollable message feed, auto-scroll with snap pill, prompt chips, and session export.

**Quick Launch:**
```powershell
.\start_frontend.ps1
```
Or manually:
```powershell
.\.venv\Scripts\python.exe server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### Option 2: Streamlit Medical Chat App
An enhanced Streamlit chat UI with medical branding, parameter sliders, and real-time streaming:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📁 Project Structure

```
Medical-Fine-Tuned-Model/
├── AGENTS.md                  # Project specification and developer guide
├── app.py                     # Medical Streamlit chat interface
├── server.py                  # FastAPI server launcher
├── start_frontend.ps1         # One-click launcher script
├── frontend/                  # Clinical web application
│   ├── index.html             # Semantic medical layout with triage hero
│   ├── style.css              # Clinical design system (dark/light themes, ECG pulse)
│   └── js/app.js              # Reactive streaming controller & scroll tracking
└── src/
    └── medical_fine_tuned_model/
        ├── backend/
        │   └── backend.py     # FastAPI endpoints (/api/health, /api/chat)
        └── model/
            ├── Modelfile      # Ollama configuration
            └── biomistral-med-q4_k_m.gguf # Quantized model weights
```
