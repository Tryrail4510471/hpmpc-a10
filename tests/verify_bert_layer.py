#!/usr/bin/env python3
"""Generate and verify a complete post-LN BERT encoder-layer reference.

The C++ boundary is exactly one ``BertLayer``::

    attention.self -> attention.output.dense
      -> residual(input) -> attention.output.LayerNorm
      -> intermediate.dense -> GELU -> output.dense
      -> residual(attention-LN) -> output.LayerNorm

Dropout is disabled by ``model.eval()``.  The public key-padding mask is saved
as ``uint8`` (1=keep, 0=mask); its additive HuggingFace equivalent is -16.0,
which is representable in the int32 Q12.20 ring.  Inputs are quantized and
dequantized before the plaintext run so the reference sees exactly the values
placed in the MPC input stream.

``--help`` and ``--describe-boundary`` require only the standard library.
Reference generation requires NumPy, PyTorch, and Transformers.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Sequence


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
COUNT_HEADER = struct.Struct("<I")
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
QUERY_SCALE_DIVISOR = 8.0
MAX_SEQUENCE_LENGTH = 128
DEFAULT_MODEL = "google-bert/bert-base-uncased"
DEFAULT_MASK_VALUE = -16.0
DEFAULT_TOLERANCE = 0.12


def boundary_document() -> dict[str, Any]:
    return {
        "input": "Q12.20 [batch,sequence_length,768] in token-contiguous SIMD packing",
        "public_mask": "uint8 [batch,sequence_length], 1=keep and 0=mask",
        "order": [
            "joint private QKV projection (query weight and bias pre-divided by 8)",
            "QK^T + public -16 mask + softmax + AV + attention output dense",
            "input residual + private-affine LayerNorm1",
            "Linear(768,3072) + GELU + Linear(3072,768)",
            "LayerNorm1 residual + private-affine LayerNorm2",
        ],
        "output": "Q12.20 [batch,sequence_length,768]",
        "tolerance": DEFAULT_TOLERANCE,
    }


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required; activate the hpmpc Python environment") from error
    return np


def quantize_q20(values: Any) -> Any:
    np = _require_numpy()
    scaled = np.rint(np.asarray(values, dtype=np.float64) * SCALE)
    return np.ascontiguousarray(np.clip(scaled, INT32_MIN, INT32_MAX).astype("<i4"))


def dequantize_q20(values: Any) -> Any:
    np = _require_numpy()
    return np.ascontiguousarray(np.asarray(values, dtype="<i4").astype(np.float32) / SCALE)


def write_counted_q20(path: Path, values: Any) -> None:
    np = _require_numpy()
    payload = quantize_q20(values).reshape(-1, order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(int(payload.size)))
        stream.write(np.ascontiguousarray(payload).tobytes(order="C"))


def write_counted_u8(path: Path, values: Any) -> None:
    np = _require_numpy()
    payload = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1, order="C")
    if not np.logical_or(payload == 0, payload == 1).all():
        raise ValueError("key-padding mask must contain only 0 or 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(int(payload.size)))
        stream.write(payload.tobytes(order="C"))


def read_counted_q20(path: Path, shape: tuple[int, ...]) -> Any:
    np = _require_numpy()
    with path.open("rb") as stream:
        header = stream.read(COUNT_HEADER.size)
        if len(header) != COUNT_HEADER.size:
            raise ValueError(f"{path}: missing uint32 count header")
        count = COUNT_HEADER.unpack(header)[0]
        payload = np.fromfile(stream, dtype="<i4")
    expected = int(np.prod(shape))
    if count != expected or payload.size != expected:
        raise ValueError(
            f"{path}: header/payload counts {count}/{payload.size}, expected {expected}"
        )
    return payload.reshape(shape).astype(np.float64) / SCALE


def load_mpc_output(path: Path, shape: tuple[int, ...]) -> Any:
    np = _require_numpy()
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError(f"{path}: NPZ must contain exactly one tensor")
        values = archive[archive.files[0]]
    else:
        values = read_counted_q20(path, shape)
    values = np.asarray(values, dtype=np.float64)
    if tuple(values.shape) != shape:
        raise ValueError(f"{path}: expected shape {shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{path}: MPC output contains NaN or infinity")
    return values


def _load_layer(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "Transformers and PyTorch are required; activate the hpmpc Python environment"
        ) from error
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only or args.offline),
        "attn_implementation": "eager",
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model = AutoModel.from_pretrained(args.model, **kwargs)
    model.eval()
    bert = getattr(model, "bert", model)
    config = bert.config
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": 12,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (expected {value})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    if mismatches:
        raise ValueError("checkpoint is not BERT-base compatible: " + ", ".join(mismatches))
    if not 0 <= args.layer < int(config.num_hidden_layers):
        raise ValueError(f"layer {args.layer} is outside [0,{config.num_hidden_layers})")
    return bert, bert.encoder.layer[args.layer]


def _manual_prescaled_layer(layer: Any, hidden_states: Any, additive_mask: Any) -> dict[str, Any]:
    """Mirror the C++ module boundaries, including the exported Q/8 contract."""
    import torch
    import torch.nn.functional as functional

    batch, sequence_length, hidden = hidden_states.shape
    self_attention = layer.attention.self
    query = functional.linear(
        hidden_states,
        self_attention.query.weight / QUERY_SCALE_DIVISOR,
        self_attention.query.bias / QUERY_SCALE_DIVISOR,
    )
    key = self_attention.key(hidden_states)
    value = self_attention.value(hidden_states)

    def split_heads(tensor: Any) -> Any:
        return tensor.view(batch, sequence_length, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    scores = torch.matmul(query, key.transpose(-1, -2)) + additive_mask
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, value)
    context = context.permute(0, 2, 1, 3).contiguous().view(
        batch, sequence_length, HIDDEN_SIZE
    )
    attention_projection = layer.attention.output.dense(context)
    attention_normalized = layer.attention.output.LayerNorm(
        attention_projection + hidden_states
    )
    intermediate = layer.intermediate(attention_normalized)
    ffn_projection = layer.output.dense(intermediate)
    output = layer.output.LayerNorm(ffn_projection + attention_normalized)
    return {
        "attention-projection": attention_projection,
        "attention-normalized": attention_normalized,
        "ffn-projection": ffn_projection,
        "output": output,
        "attention-probabilities": probabilities,
    }


def _native_layer(layer: Any, hidden_states: Any, additive_mask: Any) -> dict[str, Any]:
    self_outputs = layer.attention.self(
        hidden_states=hidden_states,
        attention_mask=additive_mask,
        head_mask=None,
        output_attentions=True,
    )
    context = self_outputs[0]
    probabilities = self_outputs[1]
    attention_projection = layer.attention.output.dense(context)
    attention_normalized = layer.attention.output(context, hidden_states)
    intermediate = layer.intermediate(attention_normalized)
    ffn_projection = layer.output.dense(intermediate)
    output = layer.output(intermediate, attention_normalized)
    return {
        "attention-projection": attention_projection,
        "attention-normalized": attention_normalized,
        "ffn-projection": ffn_projection,
        "output": output,
        "attention-probabilities": probabilities,
    }


def build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parent / "data/bert_layer_reference"
    parser = argparse.ArgumentParser(
        description="Generate and compare a deterministic complete BERT layer-0 reference."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--padding-tokens", type=int, default=16)
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mpc-output", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--describe-boundary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_boundary:
        print(json.dumps(boundary_document(), indent=2))
        return 0
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.sequence_length <= MAX_SEQUENCE_LENGTH:
        parser.error(f"--sequence-length must be in [1,{MAX_SEQUENCE_LENGTH}]")
    if not 0 <= args.padding_tokens < args.sequence_length:
        parser.error("--padding-tokens must be in [0,sequence-length)")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if not INT32_MIN / SCALE <= args.mask_value <= INT32_MAX / SCALE:
        parser.error("--mask-value is outside the int32 Q12.20 range")
    if args.mask_value >= 0:
        parser.error("--mask-value must be negative")

    np = _require_numpy()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; activate the hpmpc Python environment") from error

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    _, layer = _load_layer(args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch_size, args.sequence_length, HIDDEN_SIZE)
    raw_input = torch.randn(shape, generator=generator, dtype=torch.float32).numpy()
    input_q20 = quantize_q20(raw_input)
    hidden_states = torch.from_numpy(dequantize_q20(input_q20))

    key_padding_mask = torch.ones(
        (args.batch_size, args.sequence_length), dtype=torch.uint8
    )
    if args.padding_tokens:
        key_padding_mask[:, -args.padding_tokens :] = 0
    additive_mask = torch.zeros(
        (args.batch_size, 1, 1, args.sequence_length), dtype=torch.float32
    )
    additive_mask.masked_fill_(key_padding_mask[:, None, None, :] == 0, args.mask_value)

    with torch.inference_mode():
        manual = _manual_prescaled_layer(layer, hidden_states, additive_mask)
        native = _native_layer(layer, hidden_states, additive_mask)

    equivalence: dict[str, float] = {}
    for name in (
        "attention-projection",
        "attention-normalized",
        "ffn-projection",
        "output",
        "attention-probabilities",
    ):
        error = float(torch.max(torch.abs(manual[name] - native[name])).item())
        equivalence[name] = error
        if not np.isfinite(error) or error > 2.0e-5:
            raise AssertionError(
                f"pre-scaled-Q graph disagrees with HuggingFace at {name}: {error:.9g}"
            )

    arrays = {
        "input": hidden_states.cpu().numpy(),
        "key_padding_mask": key_padding_mask.cpu().numpy(),
        "attention_projection": manual["attention-projection"].cpu().numpy(),
        "attention_normalized": manual["attention-normalized"].cpu().numpy(),
        "ffn_projection": manual["ffn-projection"].cpu().numpy(),
        "output": manual["output"].cpu().numpy(),
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} contains NaN or infinity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bert_layer{args.layer}_seed{args.seed}"
    saved: dict[str, str] = {}
    for name, values in arrays.items():
        path = args.output_dir / f"{stem}_{name}.npy"
        np.save(path, values, allow_pickle=False)
        saved[name] = path.name
    input_bin = args.output_dir / f"{stem}_input_q20.bin"
    mask_bin = args.output_dir / f"{stem}_key_padding_mask_u8.bin"
    write_counted_q20(input_bin, arrays["input"])
    write_counted_u8(mask_bin, arrays["key_padding_mask"])
    saved["input_q20"] = input_bin.name
    saved["key_padding_mask_u8"] = mask_bin.name

    comparison: dict[str, Any] | None = None
    if args.mpc_output is not None:
        actual = load_mpc_output(args.mpc_output, shape)
        reference = arrays["output"].astype(np.float64)
        absolute = np.abs(actual - reference)
        comparison = {
            "mpc_output": str(args.mpc_output),
            "max_abs_error": float(absolute.max()),
            "mean_abs_error": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean((actual - reference) ** 2))),
            "finite": bool(np.isfinite(actual).all()),
            "tolerance": args.tolerance,
        }
        comparison["passed"] = bool(
            comparison["finite"] and comparison["max_abs_error"] < args.tolerance
        )

    metadata = {
        "model": args.model,
        "revision": args.revision,
        "encoder_layer": args.layer,
        "seed": args.seed,
        "shape": list(shape),
        "dtype": "float32",
        "fractional": FRACTIONAL,
        "input_reference_contract": "random input quantized to Q12.20 before plaintext inference",
        "post_ln_order": boundary_document()["order"],
        "mask": {
            "shape": list(key_padding_mask.shape),
            "padding_tokens_per_sample": args.padding_tokens,
            "additive_value": args.mask_value,
            "format": "uint8, 1=keep and 0=mask",
        },
        "manual_pre_scaled_q_vs_huggingface_max_abs": equivalence,
        "files": saved,
        "mpc_comparison": comparison,
    }
    metadata_path = args.output_dir / f"{stem}_reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"reference={saved['output']} shape={shape} post_ln=True")
    print(
        "pre-scaled-Q equivalence: "
        + ", ".join(f"{name}={error:.9g}" for name, error in equivalence.items())
    )
    if comparison is not None:
        status = "PASS" if comparison["passed"] else "FAIL"
        print(
            f"MPC max_abs_error={comparison['max_abs_error']:.9g} "
            f"mean_abs_error={comparison['mean_abs_error']:.9g} "
            f"threshold={args.tolerance:g} {status}"
        )
    print(f"metadata={metadata_path}")
    return 0 if comparison is None or comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
