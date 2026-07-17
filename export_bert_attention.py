#!/usr/bin/env python3
"""Export BERT layer-0 attention parameters in PIGEON counted-Q20 format.

The output is one little-endian stream compatible with the existing FFN
exporter::

    int32 total_parameter_count
    int32 parameters[total_parameter_count]

The payload is C-order and uses HuggingFace ``[out_features, in_features]``
linear-weight layout.  Q, K and V are vertically concatenated so the MPC
implementation can issue one ``768 -> 2304`` projection.

Important scaling contract
--------------------------
The Q rows (weight and bias) are divided by ``sqrt(head_dim) == 8`` *before*
Q20 quantization.  Consequently the secret attention-score GEMM computes
``(Q / 8) @ K.T`` and MUST NOT divide its result by eight again.  Moving the
public scale into Q avoids a secret fixed-point division/truncation after the
score GEMM.  The generated manifest records this transform component by
component.

``--help`` and ``--describe-schema`` only use the Python standard library.
Actual export requires NumPy, PyTorch and Transformers; ``--self-test-format``
requires NumPy only and never downloads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
COUNT_HEADER = struct.Struct("<i")
MODEL_HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = 64
ATTENTION_SCALE_DIVISOR = 8
FORMAT_NAME = "pigeon_parameter_stream_q20_v1"
DEFAULT_MODEL = "google-bert/bert-base-uncased"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    components: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "attention.self.qkv.weight",
        (3 * MODEL_HIDDEN, MODEL_HIDDEN),
        (
            {
                "source": "attention.self.query.weight",
                "axis0_range": [0, MODEL_HIDDEN],
                "transform": "divide_by_8_before_q20",
            },
            {
                "source": "attention.self.key.weight",
                "axis0_range": [MODEL_HIDDEN, 2 * MODEL_HIDDEN],
                "transform": "identity",
            },
            {
                "source": "attention.self.value.weight",
                "axis0_range": [2 * MODEL_HIDDEN, 3 * MODEL_HIDDEN],
                "transform": "identity",
            },
        ),
    ),
    ParameterSpec(
        "attention.self.qkv.bias",
        (3 * MODEL_HIDDEN,),
        (
            {
                "source": "attention.self.query.bias",
                "axis0_range": [0, MODEL_HIDDEN],
                "transform": "divide_by_8_before_q20",
            },
            {
                "source": "attention.self.key.bias",
                "axis0_range": [MODEL_HIDDEN, 2 * MODEL_HIDDEN],
                "transform": "identity",
            },
            {
                "source": "attention.self.value.bias",
                "axis0_range": [2 * MODEL_HIDDEN, 3 * MODEL_HIDDEN],
                "transform": "identity",
            },
        ),
    ),
    ParameterSpec(
        "attention.output.dense.weight",
        (MODEL_HIDDEN, MODEL_HIDDEN),
        ({"source": "attention.output.dense.weight", "transform": "identity"},),
    ),
    ParameterSpec(
        "attention.output.dense.bias",
        (MODEL_HIDDEN,),
        ({"source": "attention.output.dense.bias", "transform": "identity"},),
    ),
    ParameterSpec(
        "attention.output.LayerNorm.weight",
        (MODEL_HIDDEN,),
        ({"source": "attention.output.LayerNorm.weight", "transform": "identity"},),
    ),
    ParameterSpec(
        "attention.output.LayerNorm.bias",
        (MODEL_HIDDEN,),
        ({"source": "attention.output.LayerNorm.bias", "transform": "identity"},),
    ),
)


def total_parameter_count(specs: Sequence[ParameterSpec] = PARAMETER_SPECS) -> int:
    return sum(spec.count for spec in specs)


def schema_document() -> dict[str, Any]:
    """Return a dependency-free description of the binary contract."""
    offset = 0
    entries: list[dict[str, Any]] = []
    for spec in PARAMETER_SPECS:
        entries.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "count": spec.count,
                "element_offset": offset,
                "byte_offset": COUNT_HEADER.size + offset * 4,
                "components": list(spec.components),
            }
        )
        offset += spec.count
    return {
        "format": FORMAT_NAME,
        "fractional": FRACTIONAL,
        "scale": SCALE,
        "dtype": "int32_le",
        "header": "int32_le total_parameter_count",
        "linear_weight_layout": "HuggingFace [out_features,in_features], C-order",
        "qkv_axis0_order": ["query_scaled_by_1_over_8", "key", "value"],
        "attention_score_contract": "score = projected_q @ transpose(projected_k); do not divide by 8 again",
        "attention_mask_contract": {
            "layout": "public additive [batch,1,1,key_length]",
            "recommended_masked_value": -16.0,
            "note": "-10000 is not representable in a signed int32 Q12.20 ring",
        },
        "total_parameter_count": offset,
        "file_size_bytes": COUNT_HEADER.size + offset * 4,
        "parameters": entries,
    }


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for export/self-test; activate the project Python environment"
        ) from error
    return np


def quantize_q20(value: Any) -> Any:
    """Round through float64, saturate, and return contiguous int32-le data."""
    np = _require_numpy()
    scaled = np.rint(np.asarray(value, dtype=np.float64) * SCALE)
    return np.ascontiguousarray(np.clip(scaled, INT32_MIN, INT32_MAX).astype("<i4"))


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float32(tensor: Any) -> Any:
    np = _require_numpy()
    value = tensor.detach().cpu().float().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def _base_bert(model: Any) -> Any:
    base = getattr(model, "bert", model)
    if not hasattr(base, "encoder"):
        raise TypeError("loaded checkpoint does not expose a BERT encoder")
    return base


def _validate_config(model: Any, layer_index: int) -> None:
    config = model.config
    expected = {
        "hidden_size": MODEL_HIDDEN,
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": 12,
    }
    mismatches = [
        f"{key}={getattr(config, key, None)!r} (expected {value})"
        for key, value in expected.items()
        if getattr(config, key, None) != value
    ]
    if mismatches:
        raise ValueError("checkpoint is not BERT-base compatible: " + ", ".join(mismatches))
    if not 0 <= layer_index < int(config.num_hidden_layers):
        raise ValueError(f"layer index {layer_index} is outside [0,{config.num_hidden_layers})")


def extract_attention_parameters(model: Any, layer_index: int = 0) -> dict[str, Any]:
    """Extract one layer and apply the query pre-scaling contract."""
    np = _require_numpy()
    base = _base_bert(model)
    _validate_config(base, layer_index)
    layer = base.encoder.layer[layer_index]
    self_attention = layer.attention.self
    output = layer.attention.output

    query_weight = _as_float32(self_attention.query.weight) / ATTENTION_SCALE_DIVISOR
    query_bias = _as_float32(self_attention.query.bias) / ATTENTION_SCALE_DIVISOR
    parameters = {
        "attention.self.qkv.weight": np.ascontiguousarray(
            np.concatenate(
                (
                    query_weight,
                    _as_float32(self_attention.key.weight),
                    _as_float32(self_attention.value.weight),
                ),
                axis=0,
            ),
            dtype=np.float32,
        ),
        "attention.self.qkv.bias": np.ascontiguousarray(
            np.concatenate(
                (
                    query_bias,
                    _as_float32(self_attention.key.bias),
                    _as_float32(self_attention.value.bias),
                ),
                axis=0,
            ),
            dtype=np.float32,
        ),
        "attention.output.dense.weight": _as_float32(output.dense.weight),
        "attention.output.dense.bias": _as_float32(output.dense.bias),
        "attention.output.LayerNorm.weight": _as_float32(output.LayerNorm.weight),
        "attention.output.LayerNorm.bias": _as_float32(output.LayerNorm.bias),
    }
    for spec in PARAMETER_SPECS:
        observed = parameters[spec.name]
        if tuple(observed.shape) != spec.shape:
            raise ValueError(f"{spec.name}: expected {spec.shape}, got {observed.shape}")
        if not np.isfinite(observed).all():
            raise FloatingPointError(f"{spec.name}: source contains NaN or infinity")
    return parameters


def export_parameter_stream(
    parameters: Mapping[str, Any],
    output: Path,
    *,
    model_name: str,
    layer_index: int,
    revision: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write the combined stream and its JSON manifest."""
    np = _require_numpy()
    count = total_parameter_count()
    if count > INT32_MAX:
        raise OverflowError(f"parameter count {count} does not fit the int32 header")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    manifest_entries: list[dict[str, Any]] = []
    element_offset = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(COUNT_HEADER.pack(count))
            for spec in PARAMETER_SPECS:
                if spec.name not in parameters:
                    raise KeyError(f"missing parameter {spec.name}")
                value = np.asarray(parameters[spec.name])
                if tuple(value.shape) != spec.shape:
                    raise ValueError(f"{spec.name}: expected {spec.shape}, got {value.shape}")
                fixed = quantize_q20(value).reshape(-1, order="C")
                stream.write(fixed.tobytes(order="C"))
                manifest_entries.append(
                    {
                        "name": spec.name,
                        "shape": list(spec.shape),
                        "dtype": "int32_le",
                        "element_offset": element_offset,
                        "byte_offset": COUNT_HEADER.size + element_offset * 4,
                        "count": spec.count,
                        "components": list(spec.components),
                    }
                )
                element_offset += spec.count
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    expected_bytes = COUNT_HEADER.size + count * 4
    if output.stat().st_size != expected_bytes:
        raise IOError(f"wrote {output.stat().st_size} bytes, expected {expected_bytes}")
    manifest = {
        "format": FORMAT_NAME,
        "model": model_name,
        "revision": revision,
        "encoder_layer": layer_index,
        "fractional": FRACTIONAL,
        "scale": SCALE,
        "header": "int32_le total_parameter_count",
        "payload": "contiguous int32_le tensors in parameters order",
        "linear_weight_layout": "HuggingFace [out_features,in_features], C-order",
        "qkv_axis0_order": ["query_scaled_by_1_over_8", "key", "value"],
        "attention_score_scaling": {
            "head_dim": HEAD_DIM,
            "sqrt_head_dim": ATTENTION_SCALE_DIVISOR,
            "implementation": "query_projection_pre_scaled",
            "query_weight_and_bias_factor": 1.0 / ATTENTION_SCALE_DIVISOR,
            "score_gemm_post_scale": None,
            "contract": "score=(Q/8)@K.T; do not divide the secret score again",
        },
        "attention_mask": {
            "layout": "public additive [batch,1,1,key_length]",
            "recommended_masked_value": -16.0,
            "note": "-10000 is outside signed int32 Q12.20; existing MPC exp maps -16 to zero",
        },
        "total_parameter_count": count,
        "file_size_bytes": expected_bytes,
        "sha256": sha256_file(output),
        "parameters": manifest_entries,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_parameter_stream(
    parameters: Mapping[str, Any], output: Path, tolerance: float = 1.0e-4
) -> float:
    """Reload and dequantize every tensor, enforcing the Q20 error bound."""
    np = _require_numpy()
    with output.open("rb") as stream:
        raw_header = stream.read(COUNT_HEADER.size)
        if len(raw_header) != COUNT_HEADER.size:
            raise ValueError(f"{output}: missing parameter-count header")
        stored_count = COUNT_HEADER.unpack(raw_header)[0]
        payload = np.fromfile(stream, dtype="<i4")
    expected_count = total_parameter_count()
    if stored_count != expected_count or payload.size != expected_count:
        raise ValueError(
            f"parameter count mismatch: header={stored_count}, payload={payload.size}, "
            f"expected={expected_count}"
        )

    offset = 0
    maximum = 0.0
    for spec in PARAMETER_SPECS:
        restored = (
            payload[offset : offset + spec.count].reshape(spec.shape).astype(np.float64) / SCALE
        )
        original = np.asarray(parameters[spec.name], dtype=np.float64)
        error = float(np.max(np.abs(restored - original))) if spec.count else 0.0
        maximum = max(maximum, error)
        print(f"verify {spec.name:36s} shape={str(spec.shape):16s} max_abs_error={error:.9g}")
        offset += spec.count
    if not np.isfinite(maximum) or maximum >= tolerance:
        raise AssertionError(f"round-trip error {maximum} is not below {tolerance}")
    return maximum


def run_numpy_format_self_test() -> None:
    """Exercise the exact production schema without Torch/Transformers."""
    np = _require_numpy()
    parameters: dict[str, Any] = {}
    for index, spec in enumerate(PARAMETER_SPECS):
        # Bounded deterministic values cover signs and non-grid-aligned Q20 values.
        flat = ((np.arange(spec.count, dtype=np.int64) + 31 * index) % 2047 - 1023) / 997.0
        parameters[spec.name] = np.ascontiguousarray(flat.reshape(spec.shape), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="bert_attention_export_test_") as directory:
        output = Path(directory) / "attention_q20.bin"
        manifest_path = export_parameter_stream(
            parameters,
            output,
            model_name="numpy-self-test",
            layer_index=0,
        )
        maximum = verify_parameter_stream(parameters, output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["total_parameter_count"] != total_parameter_count():
            raise AssertionError("manifest count does not match schema")
        if manifest["attention_score_scaling"]["score_gemm_post_scale"] is not None:
            raise AssertionError("manifest lost the query pre-scaling contract")
    print(
        f"NumPy format self-test PASS: count={total_parameter_count():,} "
        f"max_abs_error={maximum:.9g}"
    )


def _load_model(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "model export requires transformers and torch; activate the project venv first"
        ) from error
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only or args.offline),
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model = AutoModel.from_pretrained(args.model, **kwargs)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    default_output = (
        Path(__file__).resolve().parent
        / "models/pretrained/bert_base_uncased_layer0_attention_q20.bin"
    )
    parser = argparse.ArgumentParser(
        description="Export one BERT-base attention layer as a combined PIGEON Q12.20 stream."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="alias for --local-files-only; never contact the HuggingFace Hub",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--describe-schema",
        action="store_true",
        help="print the binary schema using only the standard library and exit",
    )
    parser.add_argument(
        "--self-test-format",
        action="store_true",
        help="run a NumPy-only counted-Q20 round-trip test and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_schema:
        print(json.dumps(schema_document(), indent=2))
        return 0
    if args.self_test_format:
        run_numpy_format_self_test()
        return 0
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.layer < 0:
        parser.error("--layer must be non-negative")

    model = _load_model(args)
    parameters = extract_attention_parameters(model, args.layer)
    manifest_path = export_parameter_stream(
        parameters,
        args.output,
        model_name=args.model,
        layer_index=args.layer,
        revision=args.revision,
        overwrite=args.overwrite,
    )
    maximum = verify_parameter_stream(parameters, args.output, args.tolerance)
    print(f"exported {total_parameter_count():,} Q12.20 parameters to {args.output}")
    print(f"manifest={manifest_path} max_abs_error={maximum:.9g} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
