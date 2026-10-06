FROM vllm/vllm-openai:v0.30.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m ensurepip --upgrade 2>/dev/null || true; \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt \
    && test "$(python3 -c 'import vllm; print(vllm.__version__)')" = "0.30.0"

# LoRA adapter weights baked into the image (52.8 MB). The base model is NOT
# downloaded at build time; it is resolved from Runpod's HF model cache (or
# downloaded once) at container start and merged on first boot.
COPY adapter_config.json adapter_model.safetensors.index.json adapter_model-00001-of-00001.safetensors /app/adapter/
COPY src/ /app/src/
COPY handler.py test_input.json /app/

# Fail the build early if the adapter was committed as a Git LFS pointer
# instead of real weights (Runpod's GitHub build does not run `git lfs pull`).
RUN head -c 24 /app/adapter/adapter_model-00001-of-00001.safetensors \
      | grep -q "version https://git-lfs" \
      && echo "ERROR: adapter_model-00001-of-00001.safetensors is a Git LFS pointer. Store it as a normal git blob." >&2 && exit 1 \
      || true
RUN python3 -m py_compile /app/handler.py /app/src/merge_lora.py /app/src/model_setup.py

ENV BASE_MODEL_ID="Qwen/Qwen3.5-9B" \
    ADAPTER_DIR="/app/adapter" \
    MERGED_MODEL_DIR="/models/qwen3.5-9b-pack-lora" \
    SERVED_MODEL_NAME="qwen3.5-9b-pack-lora" \
    MAX_MODEL_LEN="16384" \
    GPU_MEMORY_UTILIZATION="0.90" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    PYTHONUNBUFFERED="1"

# The vllm-openai base image's ENTRYPOINT is `vllm serve`; replace it with our
# repo-owned Runpod serverless handler.
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
CMD []
