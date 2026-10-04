"""
BioMistral Medical Q&A Server Launcher
Run with: python server.py
"""

import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import uvicorn
from medical_fine_tuned_model.backend.backend import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🩺 BioMistral Medical Q&A Server Starting...")
    print("📍 Local Web App: http://localhost:8000")
    print("📍 Interactive API: http://localhost:8000/docs")
    print("📍 Connected to Ollama: http://localhost:11434")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
