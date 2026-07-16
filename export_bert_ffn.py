#!/usr/bin/env python3
"""Export BERT-base layer-0 FFN parameters in PIGEON Q12.20 format.

Binary layout (little endian):
    int32 total_parameter_count
    int32 parameters[total_parameter_count]

The payload is a C-order concatenation in PARAMETER_SPECS order. Linear
weights retain HuggingFace/PyTorch [out_features, in_features] layout, which
matches Pygeon's existing parameter_export.py flattening convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
COUNT_HEADER = struct.Struct("<i")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]


PARAMETER_SPECS = (
    ParameterSpec("intermediate.dense.weight", (3072, 768)),
    ParameterSpec("intermediate.dense.bias", (3072,)),
    ParameterSpec("output.dense.weight", (768, 3072)),
    ParameterSpec("output.dense.bias", (768,)),
    ParameterSpec("output.LayerNorm.weight", (768,)),
    ParameterSpec("output.LayerNorm.bias", (768,)),
)


def quantize_q20(value: np.ndarray) -> np.ndarray:
    """Round float values to Q12.20 and saturate to the int32 ring range."""
    scaled = np.rint(np.asarray(value, dtype=np.float64) * SCALE)
    return np.clip(scaled, INT32_MIN, INT32_MAX).astype("<i4")


def extract_layer0_ffn(model: BertModel) -> dict[str, np.ndarray]:
    layer = model.encoder.layer[0]
    tensors = {
        "intermediate.dense.weight": layer.intermediate.dense.weight,
        "intermediate.dense.bias": layer.intermediate.dense.bias,
        "output.dense.weight": layer.output.dense.weight,
        "output.dense.bias": layer.output.dense.bias,
        "output.LayerNorm.weight": layer.output.LayerNorm.weight,
        "output.LayerNorm.bias": layer.output.LayerNorm.bias,
    }
    result: dict[str, np.ndarray] = {}
    for spec in PARAMETER_SPECS:
        value = tensors[spec.name].detach().cpu().to(torch.float32).numpy()
        if value.shape != spec.shape:
            raise ValueError(f"{spec.name}: expected shape {spec.shape}, got {value.shape}")
        result[spec.name] = np.ascontiguousarray(value)
    return result


def export_bin(parameters: dict[str, np.ndarray], output: Path, model_name: str) -> Path:
    total_count = sum(int(np.prod(spec.shape)) for spec in PARAMETER_SPECS)
    if total_count > INT32_MAX:
        raise OverflowError(f"parameter count {total_count} does not fit the PIGEON int32 header")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    element_offset = 0
    with output.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(total_count))
        for spec in PARAMETER_SPECS:
            fixed = quantize_q20(parameters[spec.name]).reshape(-1, order="C")
            stream.write(fixed.tobytes(order="C"))
            count = fixed.size
            manifest_entries.append(
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": "int32_le",
                    "element_offset": element_offset,
                    "byte_offset": COUNT_HEADER.size + element_offset * np.dtype("<i4").itemsize,
                    "count": count,
                }
            )
            element_offset += count
    expected_bytes = COUNT_HEADER.size + total_count * np.dtype("<i4").itemsize
    if output.stat().st_size != expected_bytes:
        raise IOError(f"wrote {output.stat().st_size} bytes, expected {expected_bytes}")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "format": "pigeon_parameter_stream_q20_v1",
        "model": model_name,
        "encoder_layer": 0,
        "fractional": FRACTIONAL,
        "scale": SCALE,
        "header": "int32_le total_parameter_count",
        "payload": "contiguous int32_le tensors in listed order",
        "total_parameter_count": total_count,
        "file_size_bytes": expected_bytes,
        "sha256": digest,
        "parameters": manifest_entries,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_bin(parameters: dict[str, np.ndarray], output: Path, tolerance: float) -> float:
    with output.open("rb") as stream:
        header = stream.read(COUNT_HEADER.size)
        if len(header) != COUNT_HEADER.size:
            raise ValueError("missing PIGEON parameter-count header")
        stored_count = COUNT_HEADER.unpack(header)[0]
        payload = np.fromfile(stream, dtype="<i4")
    expected_count = sum(value.size for value in parameters.values())
    if stored_count != expected_count or payload.size != expected_count:
        raise ValueError(
            f"parameter count mismatch: header={stored_count}, payload={payload.size}, expected={expected_count}"
        )

    offset = 0
    global_max_error = 0.0
    for spec in PARAMETER_SPECS:
        count = int(np.prod(spec.shape))
        restored = payload[offset : offset + count].reshape(spec.shape).astype(np.float64) / SCALE
        original = parameters[spec.name].astype(np.float64)
        max_error = float(np.max(np.abs(restored - original)))
        global_max_error = max(global_max_error, max_error)
        print(f"verify {spec.name:32s} shape={str(spec.shape):16s} max_abs_error={max_error:.9g}")
        offset += count
    if not np.isfinite(global_max_error) or global_max_error >= tolerance:
        raise AssertionError(f"round-trip error {global_max_error} is not below {tolerance}")
    return global_max_error


def main() -> int:
    default_output = Path(__file__).resolve().parent / "models/pretrained/bert_base_uncased_layer0_ffn_q20.bin"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    model = BertModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model.eval()
    parameters = extract_layer0_ffn(model)
    manifest_path = export_bin(parameters, args.output, args.model)
    max_error = verify_bin(parameters, args.output, args.tolerance)
    total_count = sum(value.size for value in parameters.values())
    print(f"exported {total_count:,} Q12.20 parameters to {args.output}")
    print(f"manifest={manifest_path} max_abs_error={max_error:.9g} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
