# Qwen3.5-9B Pack-LoRA — Runpod Serverless Worker

A Fireworks-trained PEFT LoRA adapter (r=8, alpha=32) for the instruct model
`Qwen/Qwen3.5-9B`, packaged as a Runpod Serverless worker on top of
`vllm/vllm-openai:v0.30.0` (OpenAI-compatible API with Qwen3
reasoning/tool-call parsing, plus the repo-owned `handler.py` Runpod handler
so the GitHub integration detects `runpod.serverless.start`).

At container start, `handler.py` calls `src/model_setup.py` to locate the base
model (Runpod's HF model cache, or a one-time `snapshot_download`), merges the
adapter into the base weights on CPU (`src/merge_lora.py`), writes the merged
model to `MERGED_MODEL_DIR`, spawns `vllm serve`, waits for `/health`, then
starts `runpod.serverless.start({"handler": handler, ...})`. The merge is
fingerprinted, so restarts skip it if the output directory persists.

## Deploy on Runpod (GitHub integration)

1. Push this repo to GitHub (the 52.8 MB adapter is stored as a normal git blob,
   under the 100 MB file limit — do NOT re-add it to Git LFS).
2. Runpod → Serverless → New Endpoint → **GitHub** source, select this repo,
   Dockerfile at repo root.
3. Endpoint settings:
   - **Model field: `Qwen/Qwen3.5-9B`** — makes Runpod pre-cache the base model
     on workers so the first-boot merge doesn't download 19 GB.
   - **Container disk ≥ 40 GB** (base ~19 GB + merged ~19 GB + headroom).
   - **GPU ≥ 24 GB** (L4 / A5000 / RTX 4090) with the default
     `MAX_MODEL_LEN=16384`; 48 GB+ GPUs can raise `MAX_MODEL_LEN`.
4. Deploy, wait for workers to be ready. First cold start runs the merge
   (~a few minutes); subsequent starts reuse it.

To trigger a rebuild after changing the Dockerfile/scripts, push a commit or
create a GitHub release.

### Optional: merge only once with a network volume

Attach a network volume and set `MERGED_MODEL_DIR=/runpod-volume/pack-lora-merged`.
The merged weights then persist across workers/restarts; the fingerprint marker
(`pack_lora_merged.json`) skips re-merging.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL_ID` | `Qwen/Qwen3.5-9B` | HF repo id of the base model |
| `BASE_MODEL_REVISION` | — | Pin a base revision/commit |
| `ADAPTER_DIR` | `/app/adapter` | Adapter dir baked into the image |
| `MERGED_MODEL_DIR` | `/models/qwen3.5-9b-pack-lora` | Where merged weights are written |
| `HF_TOKEN` | — | For gated base repos |
| `SERVED_MODEL_NAME` | `qwen3.5-9b-pack-lora` | Model name in the OpenAI API |
| `MAX_MODEL_LEN` | `16384` | vLLM `--max-model-len` |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM `--gpu-memory-utilization` |
| `MAX_CONCURRENCY` | `16` | Runpod worker concurrency |
| `VLLM_STARTUP_TIMEOUT` | `1800` | vLLM `/health` wait timeout (s) |
| `REQUEST_TIMEOUT` | `3600` | Per-request timeout to vLLM (s) |
| `VLLM_EXTRA_ARGS` | — | Extra `vllm serve` args (e.g. `--language-model-only` to skip loading the vision tower for text-only use) |
| `VLLM_PORT` | `8000` | Local vLLM listen port |

## Requests

OpenAI-compatible (chat completions):

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
    api_key="<RUNPOD_API_KEY>",
)
resp = client.chat.completions.create(
    model="qwen3.5-9b-pack-lora",
    messages=[{"role": "user", "content": "Hello"}],
    # disable thinking output:
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

Native Runpod `/runsync` (see `test_input.json` for the shape):

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello"}],
        "sampling_params": {"temperature": 0.7, "max_tokens": 256}}}'
```

Shorthand jobs return the vLLM JSON plus convenience top-level `text` (and
`reasoning_content` when the model emits thinking). `stream: true` streams raw
SSE chunks. `max_new_tokens` maps to `max_tokens`; `openai_route`/`openai_input`
or `route`/`body` proxy arbitrary vLLM endpoints.

## Local testing

On a GPU machine with the image built, `python3 handler.py` loads the model and
starts the serverless loop; Runpod SDK local mode can feed it
`test_input.json` (`--test_input`/`rp_input`). `pytest tests/` runs the merge
and `build_request`/proxy unit tests on CPU.

## Standalone offline merge

```bash
pip install torch safetensors
python3 src/merge_lora.py --base /path/to/Qwen3.5-9B --adapter . --out /path/to/merged
```
