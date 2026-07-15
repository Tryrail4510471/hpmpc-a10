#!/usr/bin/env python3
"""Export BERT layer-0 attention-output LayerNorm gamma and beta."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


WEIGHT_KEY = "bert.encoder.layer.0.attention.output.LayerNorm.weight"
BIAS_KEY = "bert.encoder.layer.0.attention.output.LayerNorm.bias"
LEGACY_WEIGHT_KEY = "bert.encoder.layer.0.attention.output.LayerNorm.gamma"
LEGACY_BIAS_KEY = "bert.encoder.layer.0.attention.output.LayerNorm.beta"


def load_with_transformers(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    from transformers import BertModel

    model = BertModel.from_pretrained(model_name)
    layer_norm = model.encoder.layer[0].attention.output.LayerNorm
    return (
        layer_norm.weight.detach().cpu().numpy().astype(np.float32),
        layer_norm.bias.detach().cpu().numpy().astype(np.float32),
    )


def load_with_safetensors(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    with safe_open(model_path, framework="np", device="cpu") as weights:
        keys = set(weights.keys())
        weight_key = WEIGHT_KEY if WEIGHT_KEY in keys else LEGACY_WEIGHT_KEY
        bias_key = BIAS_KEY if BIAS_KEY in keys else LEGACY_BIAS_KEY
        if weight_key not in keys or bias_key not in keys:
            raise KeyError(f"missing attention-output LayerNorm parameters in {model_path}")
        gamma = np.asarray(weights.get_tensor(weight_key), dtype=np.float32)
        beta = np.asarray(weights.get_tensor(bias_key), dtype=np.float32)
    return gamma, beta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output", type=Path, default=Path("/tmp/bert_base_uncased_layer0_attention_ln.npz"))
    args = parser.parse_args()

    try:
        gamma, beta = load_with_transformers(args.model)
        backend = "transformers"
    except ImportError:
        gamma, beta = load_with_safetensors(args.model)
        backend = "huggingface_hub+safetensors"
    if gamma.shape != (768,) or beta.shape != (768,):
        raise ValueError(f"expected gamma/beta shape (768,), got {gamma.shape}/{beta.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        gamma=gamma,
        beta=beta,
        model=np.array(args.model),
        module=np.array("encoder.layer.0.attention.output.LayerNorm"),
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"exported {args.model} encoder.layer.0.attention.output.LayerNorm")
    print(f"backend={backend} output={args.output} sha256={digest}")
    print(
        f"gamma[min,max,mean]=({gamma.min():.9g},{gamma.max():.9g},{gamma.mean():.9g}) "
        f"beta[min,max,mean]=({beta.min():.9g},{beta.max():.9g},{beta.mean():.9g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
