import json
import os
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import merge_lora

PREFIX = merge_lora.ADAPTER_PREFIX


def _write_base(base_dir):
    """Tiny 2-shard base model, including a stacked-rank qkv-like module."""
    base_dir = str(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    shard1 = {
        "model.layers.0.qkv_proj.weight": torch.randn(12, 8, dtype=torch.bfloat16),
        "model.layers.0.o_proj.weight": torch.randn(4, 6, dtype=torch.bfloat16),
    }
    shard2 = {
        "model.layers.1.down_proj.weight": torch.randn(4, 6, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(10, 8, dtype=torch.bfloat16),
    }
    save_file(shard1, os.path.join(base_dir, "model-00001-of-00002.safetensors"))
    save_file(shard2, os.path.join(base_dir, "model-00002-of-00002.safetensors"))
    weight_map = {
        "model.layers.0.qkv_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.o_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.down_proj.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    with open(os.path.join(base_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump({"model_type": "tiny"}, f)
    with open(os.path.join(base_dir, "tokenizer_config.json"), "w") as f:
        json.dump({}, f)
    return base_dir, {**shard1, **shard2}


def _write_adapter(adapter_dir, modules, r=2, alpha=8):
    """modules: {base_name: (in_dim, out_dim, ranks)} -> writes adapter shard.

    ranks: multiplier on A rows / B cols for stacked-rank modules (1 normal).
    Returns {base_name: (A, B)}.
    """
    os.makedirs(adapter_dir, exist_ok=True)
    tensors = {}
    pairs = {}
    for name, (in_dim, out_dim, k) in modules.items():
        stem = name[: -len(".weight")]
        a = torch.randn(r * k, in_dim, dtype=torch.bfloat16)
        b = torch.randn(out_dim, r * k, dtype=torch.bfloat16)
        tensors[PREFIX + stem + ".lora_A.weight"] = a
        tensors[PREFIX + stem + ".lora_B.weight"] = b
        pairs[name] = (a, b)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model-00001-of-00001.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha,
                   "use_dora": False, "use_rslora": False}, f)
    return pairs


def test_merge_end_to_end(tmp_path):
    base_dir, base_tensors = _write_base(tmp_path / "base")
    r, alpha = 2, 8
    pairs = _write_adapter(
        str(tmp_path / "adapter"),
        {
            "model.layers.0.qkv_proj.weight": (8, 12, 3),   # stacked rank: A[3r,in], B[out,3r]
            "model.layers.0.o_proj.weight": (6, 4, 1),
            "model.layers.1.down_proj.weight": (6, 4, 1),
            # lm_head.weight intentionally not adapted
        },
        r=r, alpha=alpha,
    )
    out_dir = str(tmp_path / "out")
    marker = merge_lora.merge(base_dir, str(tmp_path / "adapter"), out_dir)

    scaling = alpha / r
    for name, (a, b) in pairs.items():
        shard = "model-00001-of-00002.safetensors" if "layers.0" in name else "model-00002-of-00002.safetensors"
        merged = load_file(os.path.join(out_dir, shard))[name]
        expected = (base_tensors[name].float() + scaling * (b.float() @ a.float())).to(torch.bfloat16)
        assert torch.allclose(merged.float(), expected.float(), atol=2e-2), name

    # untouched tensor byte-identical
    orig = load_file(os.path.join(base_dir, "model-00002-of-00002.safetensors"))["lm_head.weight"]
    merged_lm = load_file(os.path.join(out_dir, "model-00002-of-00002.safetensors"))["lm_head.weight"]
    assert torch.equal(orig, merged_lm)

    # non-safetensors copied
    assert json.load(open(os.path.join(out_dir, "config.json")))["model_type"] == "tiny"
    assert os.path.isfile(os.path.join(out_dir, "tokenizer_config.json"))
    assert os.path.isfile(os.path.join(out_dir, "model.safetensors.index.json"))

    # marker written and matches
    assert os.path.isfile(os.path.join(out_dir, merge_lora.MARKER_NAME))
    assert marker["num_merged_modules"] == 3
    assert merge_lora.marker_matches(out_dir, str(tmp_path / "adapter"), base_dir)

    # no leftover tmp dir
    assert not os.path.exists(out_dir + ".tmp")


def test_missing_module_errors(tmp_path):
    base_dir, _ = _write_base(tmp_path / "base")
    _write_adapter(str(tmp_path / "adapter"),
                   {"model.layers.0.qkv_proj.weight": (8, 12, 1),
                    "model.layers.9.nonexistent.weight": (4, 4, 1)})
    with pytest.raises(RuntimeError, match="not found in base"):
        merge_lora.merge(base_dir, str(tmp_path / "adapter"), str(tmp_path / "out"))


def test_lfs_pointer_errors(tmp_path):
    base_dir, _ = _write_base(tmp_path / "base")
    adapter_dir = tmp_path / "adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump({"peft_type": "LORA", "r": 2, "lora_alpha": 8}, f)
    with open(adapter_dir / "adapter_model-00001-of-00001.safetensors", "w") as f:
        f.write("version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc123\nsize 12345\n")
    with pytest.raises(RuntimeError, match="LFS pointer"):
        merge_lora.merge(base_dir, str(adapter_dir), str(tmp_path / "out"))
