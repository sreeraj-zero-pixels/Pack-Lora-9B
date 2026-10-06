"""Merge a PEFT LoRA adapter into a base HF model's safetensors shards.

Importable via ``merge(base_dir, adapter_dir, out_dir)`` and runnable as a CLI:

    python3 merge_lora.py --base /path/to/base --adapter /path/to/adapter --out /path/to/out

The adapter keys are expected in HF PEFT layout, e.g.
``base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight``.
Stripping the ``base_model.model.`` prefix and replacing the ``.lora_[AB].weight``
suffix with ``.weight`` yields the base tensor name.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

log = logging.getLogger("merge_lora")

ADAPTER_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
LFS_MAGIC = b"version https://git-lfs"
MARKER_NAME = "pack_lora_merged.json"


def _is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(len(LFS_MAGIC)) == LFS_MAGIC


def _assert_not_lfs(path):
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer file, not real weights. "
            "Pull LFS objects (git lfs pull) or store the file as a normal blob."
        )


def adapter_key_to_base(key):
    """Map an adapter tensor key to the base model tensor name, or None."""
    if not key.startswith(ADAPTER_PREFIX):
        return None
    rest = key[len(ADAPTER_PREFIX):]
    if rest.endswith(LORA_A_SUFFIX):
        return rest[: -len(LORA_A_SUFFIX)] + ".weight"
    if rest.endswith(LORA_B_SUFFIX):
        return rest[: -len(LORA_B_SUFFIX)] + ".weight"
    return None


def load_adapter(adapter_dir):
    """Return {base_tensor_name: (A, B)} pairs from a PEFT LoRA adapter dir."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("peft_type", "LORA") != "LORA":
        raise RuntimeError(f"Unsupported peft_type: {cfg.get('peft_type')}")
    if cfg.get("use_dora"):
        raise RuntimeError("DoRA adapters are not supported by this merge script")
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        scaling = alpha / math.sqrt(r)
    else:
        scaling = alpha / r

    shard_files = sorted(
        f for f in os.listdir(adapter_dir)
        if f.startswith("adapter_model") and f.endswith(".safetensors")
    )
    if not shard_files:
        raise RuntimeError(f"No adapter_model*.safetensors files found in {adapter_dir}")

    loras_a = {}
    loras_b = {}
    for fname in shard_files:
        path = os.path.join(adapter_dir, fname)
        _assert_not_lfs(path)
        tensors = load_file(path)
        for key, tensor in tensors.items():
            base_name = adapter_key_to_base(key)
            if base_name is None:
                raise RuntimeError(f"Unrecognized adapter tensor key: {key}")
            if key.endswith(LORA_A_SUFFIX):
                loras_a[base_name] = tensor
            else:
                loras_b[base_name] = tensor

    missing_b = sorted(set(loras_a) - set(loras_b))
    missing_a = sorted(set(loras_b) - set(loras_a))
    if missing_a or missing_b:
        raise RuntimeError(
            f"Unpaired LoRA tensors: missing lora_A for {missing_a[:5]}, "
            f"missing lora_B for {missing_b[:5]}"
        )

    pairs = {}
    for name, a in loras_a.items():
        b = loras_b[name]
        if b.shape[1] != a.shape[0]:
            raise RuntimeError(
                f"LoRA rank mismatch for {name}: A {tuple(a.shape)}, B {tuple(b.shape)}"
            )
        pairs[name] = (a, b)
    return pairs, scaling


def fingerprint(adapter_dir, base_dir):
    """Fingerprint of adapter weights+config and the base snapshot identity."""
    h = hashlib.sha256()
    for fname in sorted(os.listdir(adapter_dir)):
        if fname == "adapter_config.json" or (
            fname.startswith("adapter_model") and fname.endswith(".safetensors")
        ):
            h.update(fname.encode())
            with open(os.path.join(adapter_dir, fname), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
    h.update(os.path.basename(os.path.realpath(base_dir)).encode())
    return h.hexdigest()


def _copy_non_safetensors(base_dir, out_dir):
    for fname in sorted(os.listdir(base_dir)):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(base_dir, fname)
        dst = os.path.join(out_dir, fname)
        if not os.path.isfile(src):
            continue
        real = os.path.realpath(src)
        if os.path.isdir(real):
            continue
        shutil.copyfile(real, dst)


def merge(base_dir, adapter_dir, out_dir):
    pairs, scaling = load_adapter(adapter_dir)
    log.info("Loaded %d adapter modules, scaling=%.6f", len(pairs), scaling)

    base_index_path = os.path.join(base_dir, "model.safetensors.index.json")
    with open(os.path.realpath(base_index_path)) as f:
        weight_map = json.load(f)["weight_map"]

    not_found = sorted(set(pairs) - set(weight_map))
    if not_found:
        raise RuntimeError(
            f"{len(not_found)} adapter modules not found in base model, "
            f"e.g. {not_found[:5]}"
        )

    tmp_dir = f"{out_dir}.tmp-{os.getpid()}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    shard_to_targets = {}
    for name in pairs:
        shard_to_targets.setdefault(weight_map[name], []).append(name)

    merged_count = 0
    for shard_name in sorted(set(weight_map.values())):
        shard_path = os.path.realpath(os.path.join(base_dir, shard_name))
        targets = shard_to_targets.get(shard_name, [])
        tensors = load_file(shard_path)
        for name in targets:
            a, b = pairs[name]
            w = tensors[name]
            delta = (b.float() @ a.float()) * scaling
            if delta.shape != w.shape:
                raise RuntimeError(
                    f"Delta shape {tuple(delta.shape)} != base weight shape "
                    f"{tuple(w.shape)} for {name}"
                )
            tensors[name] = (w.float() + delta).to(w.dtype)
            merged_count += 1
        save_file(tensors, os.path.join(tmp_dir, shard_name), metadata={"format": "pt"})
        log.info(
            "Wrote %s (%d merged in this shard, %d/%d total)",
            shard_name, len(targets), merged_count, len(pairs),
        )
        del tensors

    _copy_non_safetensors(base_dir, tmp_dir)

    marker = {
        "fingerprint": fingerprint(adapter_dir, base_dir),
        "num_merged_modules": merged_count,
        "scaling": scaling,
        "base_dir": os.path.basename(os.path.realpath(base_dir)),
    }
    with open(os.path.join(tmp_dir, MARKER_NAME), "w") as f:
        json.dump(marker, f, indent=2)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    log.info("Merge complete: %d modules merged into %s", merged_count, out_dir)
    return marker


def marker_matches(out_dir, adapter_dir, base_dir):
    marker_path = os.path.join(out_dir, MARKER_NAME)
    if not os.path.isfile(marker_path):
        return False
    try:
        with open(marker_path) as f:
            marker = json.load(f)
        return marker.get("fingerprint") == fingerprint(adapter_dir, base_dir)
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model")
    parser.add_argument("--base", required=True, help="Base model directory")
    parser.add_argument("--adapter", required=True, help="Adapter directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    merge(args.base, args.adapter, args.out)


if __name__ == "__main__":
    sys.exit(main())
