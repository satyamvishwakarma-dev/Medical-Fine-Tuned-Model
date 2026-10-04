# Project context: BioMistral medical Q&A (fine-tuned, runs locally)

Drop this file in the project root (works as `AGENTS.md`, or rename/copy to `CLAUDE.md`)
so a CLI coding agent knows the project without re-explaining it.

## Goal
A fine-tuned medical Q&A LLM running fully locally on a Windows laptop, with a Streamlit chat UI.
Learning project (software engineering student -> AI engineer). Not for real medical advice.

## Model
- Base model: `BioMistral/BioMistral-7B` (Mistral-7B, further pre-trained on PubMed Central)
- Method: QLoRA supervised fine-tuning (4-bit NF4 base + LoRA adapters)
- Dataset: `lavita/medical-qa-datasets`, config `all-processed` (columns: instruction, input, output)
  - random 20,000-example subset, 1 epoch
- LoRA: r=16, alpha=32, dropout=0.05
  - targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - about 0.58% of parameters trainable
- Training settings: max seq length 512, per-GPU batch 4, grad accumulation 4
  (effective batch 32 on 2 GPUs), lr 2e-4, fp16 (T4 has no bf16), gradient checkpointing,
  paged_adamw_8bit, loss masked (-100) on prompt tokens so only the answer is learned
- Hardware for training: Kaggle free tier, 2x Tesla T4, `torchrun --nproc_per_node=2` (DDP),
  `device_map={"": local_rank}`

## Prompt format (training and inference must match exactly)
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{answer}<eos>
```
At inference the user question goes in the instruction slot and Input is left empty.
The Ollama `TEMPLATE` reproduces this, so apps send only the raw question.

## Artifacts and pipeline
1. LoRA adapter (adapter weights + tokenizer files) saved on Kaggle as `.../finetuned-model/final`
2. Merge adapter into the fp16 base model on CPU (`merge_and_unload`), done on Kaggle
   (the laptop connection was too slow for the 14 GB base download)
3. Convert merged model to GGUF fp16 with llama.cpp `convert_hf_to_gguf.py`
4. Quantize to Q4_K_M with `llama-quantize` -> `biomistral-med-q4_k_m.gguf` (about 4.1 GiB)
5. Register in Ollama with a Modelfile as `biomistral-med`
6. Streamlit app calls the Ollama HTTP API

Kaggle notebook used for steps 2-4: `kaggle_merge_quantize.ipynb`
(intermediates in `/kaggle/temp`, only the final file in `/kaggle/working`).

## Local environment
- OS/shell: Windows, PowerShell. Give all commands in PowerShell syntax, not bash.
- Laptop: 4 GB VRAM GPU, 32 GB RAM. The 4.4 GB model is split between GPU and CPU by Ollama.
- Ollama app installed (server on `http://localhost:11434`)
- Python with `streamlit` and `requests`

## Files
- `biomistral-med-q4_k_m.gguf` - quantized model
- `Modelfile` - Ollama config (see below)
- `app.py` - Streamlit chat UI (streams answers from Ollama `/api/generate`)
- `merge.py` - local CPU merge script (unused, replaced by the Kaggle notebook)
- `chat.py` - direct transformers + PEFT 4-bit inference script (needs ~6 GB VRAM, not used on this laptop)

## Modelfile
```
FROM ./biomistral-med-q4_k_m.gguf

TEMPLATE """### Instruction:
{{ .Prompt }}

### Input:


### Response:
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "</s>"
PARAMETER stop "### Instruction:"
```
Write it without a byte-order mark (PowerShell `Set-Content -Encoding utf8` can add one).

## Common commands (PowerShell)
```powershell
# build / run the model
ollama create biomistral-med -f Modelfile
ollama run biomistral-med "What are the common symptoms of type 2 diabetes?"
ollama list
ollama ps            # shows GPU/CPU split

# run the UI
pip install streamlit requests
streamlit run app.py

# call the API directly
$body = @{ model = "biomistral-med"; prompt = "What is hypertension?"; stream = $false } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method Post -Body $body -ContentType "application/json"
```

## Gotchas learned
- `ollama run <file>.gguf` fails ("pulling manifest"): a GGUF must be registered with `ollama create` first, then run by the model name.
- A LoRA adapter alone cannot run; it needs the matching base model.
- Merge into the unquantized fp16 base, not the 4-bit one.
- Kaggle `/kaggle/working` is limited to about 20 GB; keep big intermediates in `/kaggle/temp`.
- Kaggle file tooltips can show a stale size while a file is still being written; check with `ls -lh`.
- Browser downloads of `file (1).gguf` should be renamed before use.
- The model is single-turn (trained on single Q&A), so chat history is displayed but not sent back to it.

## Conventions for the agent
- Prefer clean, simple, easy-to-read code with short comments (user is a student).
- Give PowerShell commands.
- Keep the medical disclaimer visible in any UI.
- Do not change the prompt template without retraining or updating the Modelfile to match.

## Possible next steps
- Run a fixed set of 15-20 test questions and review the answers
- Add retrieval (RAG) over trusted medical sources
- Evaluate quality against the base model
- Write a README and publish the project (check the licenses of BioMistral and the dataset first)
