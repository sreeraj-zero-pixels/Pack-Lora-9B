"""Base-model resolution + LoRA merge for the Runpod worker.

`prepare_merged_model()` locates the base model (Runpod HF cache or a one-time
snapshot_download), merges the baked-in adapter into MERGED_MODEL_DIR (skipped
when the fingerprint marker matches) and returns the merged dir. Called once
at container start by handler.py.
"""

import json
import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_lora

log = logging.getLogger("model_setup")

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3.5-9B")
BASE_MODEL_REVISION = os.environ.get("BASE_MODEL_REVISION") or None
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/app/adapter")
MERGED_MODEL_DIR = os.environ.get("MERGED_MODEL_DIR", "/models/qwen3.5-9b-pack-lora")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*"]


def _snapshot_is_complete(snap_dir):
    index = os.path.join(snap_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return False
    try:
        with open(os.path.realpath(index)) as f:
            shards = set(json.load(f)["weight_map"].values())
    except Exception:
        return False
    return all(
        os.path.isfile(os.path.join(snap_dir, s)) for s in shards
    )


def _find_cached_snapshot(hf_home):
    repo_dir = os.path.join(
        hf_home, "models--" + BASE_MODEL_ID.replace("/", "--")
    )
    snaps_root = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snaps_root):
        return None

    if BASE_MODEL_REVISION:
        cand = os.path.join(snaps_root, BASE_MODEL_REVISION)
        if _snapshot_is_complete(cand):
            return cand
        log.warning("Requested revision %s not complete in cache", BASE_MODEL_REVISION)

    refs_main = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            cand = os.path.join(snaps_root, f.read().strip())
        if _snapshot_is_complete(cand):
            return cand

    candidates = [
        os.path.join(snaps_root, d)
        for d in sorted(os.listdir(snaps_root), reverse=True)
    ]
    for cand in candidates:
        if os.path.isdir(cand) and _snapshot_is_complete(cand):
            return cand
    return None


def resolve_base_dir():
    hf_homes = []
    if os.environ.get("HF_HOME"):
        hf_homes.append(os.environ["HF_HOME"])
    hf_homes.append("/runpod-volume/huggingface-cache/hub")

    for hf_home in dict.fromkeys(hf_homes):
        snap = _find_cached_snapshot(hf_home)
        if snap:
            log.info("Using cached base snapshot: %s", snap)
            return snap

    log.info("Base model not in cache; downloading %s", BASE_MODEL_ID)
    from huggingface_hub import snapshot_download

    last_err = None
    for cache_dir in dict.fromkeys(hf_homes + ["/models/hf-cache"]):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            snap = snapshot_download(
                BASE_MODEL_ID,
                revision=BASE_MODEL_REVISION,
                token=HF_TOKEN,
                cache_dir=cache_dir,
                allow_patterns=ALLOW_PATTERNS,
            )
            if _snapshot_is_complete(snap):
                return snap
            last_err = RuntimeError(f"Downloaded snapshot {snap} is incomplete")
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot_download into %s failed: %s", cache_dir, e)
            last_err = e
    raise RuntimeError(f"Could not resolve base model {BASE_MODEL_ID}: {last_err}")


def prepare_merged_model():
    """Resolve the base model, merge the adapter if needed, return merged dir."""
    t0 = time.time()
    base_dir = resolve_base_dir()
    log.info("Base model dir resolved in %.1fs: %s", time.time() - t0, base_dir)

    t1 = time.time()
    if merge_lora.marker_matches(MERGED_MODEL_DIR, ADAPTER_DIR, base_dir):
        log.info("Merged model up to date at %s; skipping merge", MERGED_MODEL_DIR)
    else:
        log.info("Merging adapter %s into %s", ADAPTER_DIR, MERGED_MODEL_DIR)
        if os.path.isdir(MERGED_MODEL_DIR):
            shutil.rmtree(MERGED_MODEL_DIR)
        merge_lora.merge(base_dir, ADAPTER_DIR, MERGED_MODEL_DIR)
        log.info("Merge finished in %.1fs", time.time() - t1)
    return MERGED_MODEL_DIR
