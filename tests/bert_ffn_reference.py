#!/usr/bin/env python3
"""Generate a deterministic plaintext reference for BERT layer-0 FFN+LN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel


def main() -> int:
    default_dir = Path(__file__).resolve().parent / "data/bert_ffn_reference"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = BertModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model.eval()
    layer = model.encoder.layer[0]

    generator = torch.Generator(device="cpu").manual_seed(42)
    hidden_states = torch.randn((2, 128, 768), generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        intermediate = layer.intermediate(hidden_states)
        # BertOutput applies output.dense, disabled dropout, residual addition,
        # and output.LayerNorm in the same order as HuggingFace inference.
        output = layer.output(intermediate, hidden_states)

    input_array = hidden_states.cpu().numpy()
    output_array = output.cpu().numpy()
    if input_array.shape != (2, 128, 768) or output_array.shape != input_array.shape:
        raise AssertionError(f"unexpected input/output shape: {input_array.shape}/{output_array.shape}")
    if not np.isfinite(input_array).all() or not np.isfinite(output_array).all():
        raise FloatingPointError("reference contains NaN or infinity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "bert_layer0_ffn_input_seed42.npy"
    output_path = args.output_dir / "bert_layer0_ffn_output_seed42.npy"
    metadata_path = args.output_dir / "bert_layer0_ffn_reference.json"
    np.save(input_path, input_array, allow_pickle=False)
    np.save(output_path, output_array, allow_pickle=False)
    metadata = {
        "model": args.model,
        "encoder_layer": 0,
        "operation": "intermediate.dense+GELU -> output.dense -> residual -> output.LayerNorm",
        "seed": 42,
        "dtype": "float32",
        "shape": list(input_array.shape),
        "input_file": input_path.name,
        "output_file": output_path.name,
        "input_mean": float(input_array.mean()),
        "input_std": float(input_array.std()),
        "output_mean": float(output_array.mean()),
        "output_std": float(output_array.std()),
        "output_min": float(output_array.min()),
        "output_max": float(output_array.max()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    reloaded_input = np.load(input_path, allow_pickle=False)
    reloaded_output = np.load(output_path, allow_pickle=False)
    if not np.array_equal(reloaded_input, input_array) or not np.array_equal(reloaded_output, output_array):
        raise AssertionError("saved NPY files do not round-trip exactly")
    print(f"input={input_path} shape={input_array.shape}")
    print(f"output={output_path} shape={output_array.shape}")
    print(f"metadata={metadata_path} finite=True PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
