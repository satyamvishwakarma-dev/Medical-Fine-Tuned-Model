"""Merge the LoRA adapter into BioMistral-7B and save a standalone model (CPU only)."""

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "BioMistral/BioMistral-7B"
ADAPTER_DIR = r"E:\Projects\Medical-Fine-Tuned-Model\src\medical_fine_tuned_model\model\BioMistral"  # <-- change this
OUTPUT_DIR = r"E:\Projects\Medical-Fine-Tuned-Model\src\medical_fine_tuned_model\model\biomistral-merged"

print("Loading base model in fp16 on CPU (needs ~15 GB RAM)...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Attaching adapter and merging...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.merge_and_unload()  # bakes LoRA weights into the base weights

print(f"Saving merged model to '{OUTPUT_DIR}'...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Ollama/llama.cpp like having the original SentencePiece file next to the model
try:
    hf_hub_download(BASE_MODEL, "tokenizer.model", local_dir=OUTPUT_DIR)
except Exception as err:
    print("Skipped tokenizer.model download:", err)

print("Done.")
