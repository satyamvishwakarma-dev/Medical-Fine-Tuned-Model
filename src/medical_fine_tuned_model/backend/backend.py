"""
BioMistral Medical Q&A - FastAPI Backend Server
Streams responses from local Ollama model (biomistral-med) and serves the clinical web frontend.
"""

import json
import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("medical-backend")

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "biomistral-med"

# Workspace paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="BioMistral Medical Q&A API",
    description="Local medical inference server powered by fine-tuned BioMistral-7B",
    version="1.0.0"
)

# Enable CORS for local dev / cross-origin web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Clinical question or patient query")
    temperature: float = Field(0.3, ge=0.0, le=1.5, description="Sampling temperature")
    max_tokens: int = Field(512, ge=32, le=2048, description="Maximum tokens to generate")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    repeat_penalty: float = Field(1.1, ge=0.8, le=2.0, description="Repetition penalty")


@app.get("/api/health")
async def health_check():
    """Check Ollama connectivity and verify biomistral-med model status."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                has_model = any(MODEL_NAME in m for m in models)
                
                # Check running processes
                ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
                running_models = []
                if ps_resp.status_code == 200:
                    running_models = [m.get("name", "") for m in ps_resp.json().get("models", [])]

                return {
                    "status": "online",
                    "ollama_connected": True,
                    "target_model": MODEL_NAME,
                    "model_available": has_model,
                    "model_active_in_memory": any(MODEL_NAME in m for m in running_models),
                    "available_models": models
                }
            return {
                "status": "warning",
                "ollama_connected": False,
                "message": f"Ollama returned HTTP {resp.status_code}"
            }
    except Exception as exc:
        logger.warning(f"Ollama health check failed: {exc}")
        return {
            "status": "offline",
            "ollama_connected": False,
            "error": str(exc),
            "message": "Ollama server is not responding at http://localhost:11434. Ensure Ollama is running."
        }


async def stream_ollama_generator(request_data: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Ollama /api/generate as Server-Sent Events.
    Uses the exact prompt format configured in Ollama's Modelfile.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": request_data.prompt.strip(),
        "stream": True,
        "options": {
            "temperature": request_data.temperature,
            "num_predict": request_data.max_tokens,
            "top_p": request_data.top_p,
            "repeat_penalty": request_data.repeat_penalty,
        }
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    error_payload = json.dumps({"error": f"Ollama error ({response.status_code}): {error_detail.decode()}"})
                    yield f"data: {error_payload}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            token_text = chunk.get("response", "")
                            is_done = chunk.get("done", False)
                            
                            data = {
                                "token": token_text,
                                "done": is_done,
                            }
                            if is_done:
                                data["eval_count"] = chunk.get("eval_count", 0)
                                data["eval_duration"] = chunk.get("eval_duration", 0)
                                data["total_duration"] = chunk.get("total_duration", 0)
                            
                            yield f"data: {json.dumps(data)}\n\n"
                            if is_done:
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            err_data = json.dumps({"error": "Cannot connect to Ollama at http://localhost:11434. Please start the Ollama application."})
            yield f"data: {err_data}\n\n"
        except Exception as exc:
            err_data = json.dumps({"error": f"Inference stream failed: {str(exc)}"})
            yield f"data: {err_data}\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Real-time streaming inference endpoint for medical Q&A."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    
    return StreamingResponse(
        stream_ollama_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Mount the frontend directory to serve static assets and UI
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at: {FRONTEND_DIR}")


def start():
    """Helper entry point for running the server."""
    import uvicorn
    print("\n" + "=" * 60)
    print("🩺 BioMistral Medical Q&A Server Starting...")
    print("📍 Frontend: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/docs")
    print("📍 Ollama Endpoint: http://localhost:11434")
    print("=" * 60 + "\n")
    uvicorn.run("medical_fine_tuned_model.backend.backend:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()