FROM runpod/worker-v1-vllm:v2.28.0

# LoRA adapter weights baked into the image (52.8 MB). The base model is NOT
# downloaded at build time; it is resolved from Runpod's HF model cache (or
# downloaded once) at container start and merged on first boot.
COPY adapter_config.json adapter_model.safetensors.index.json adapter_model-00001-of-00001.safetensors /opt/pack-lora/adapter/
COPY src/merge_lora.py src/start.py /opt/pack-lora/

# Fail the build early if the adapter was committed as a Git LFS pointer
# instead of real weights (Runpod's GitHub build does not run `git lfs pull`).
RUN head -c 24 /opt/pack-lora/adapter/adapter_model-00001-of-00001.safetensors \
      | grep -q "version https://git-lfs" \
      && echo "ERROR: adapter_model-00001-of-00001.safetensors is a Git LFS pointer. Store it as a normal git blob." >&2 && exit 1 \
      || true
RUN python3 -m py_compile /opt/pack-lora/merge_lora.py /opt/pack-lora/start.py

ENV BASE_MODEL_ID="Qwen/Qwen3.5-9B" \
    ADAPTER_DIR="/opt/pack-lora/adapter" \
    MERGED_MODEL_DIR="/models/qwen3.5-9b-pack-lora" \
    OPENAI_SERVED_MODEL_NAME_OVERRIDE="qwen3.5-9b-pack-lora" \
    MAX_MODEL_LEN="16384" \
    GPU_MEMORY_UTILIZATION="0.90" \
    REASONING_PARSER="qwen3" \
    ENABLE_AUTO_TOOL_CHOICE="true" \
    TOOL_CALL_PARSER="qwen3_coder" \
    VLLM_STARTUP_TIMEOUT="1800"

ENTRYPOINT ["python3", "/opt/pack-lora/start.py"]
