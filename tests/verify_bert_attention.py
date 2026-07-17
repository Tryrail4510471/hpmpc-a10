#!/usr/bin/env python3
"""Generate and verify a deterministic BERT layer-0 attention reference.

The primary MPC module boundary is deliberately explicit:

``pre-residual``
    ``BertSelfAttention`` context followed by ``attention.output.dense``.
    It excludes dropout, residual addition, and output LayerNorm.  This is the
    boundary implemented by ``bert_attention.hpp`` so ``bert_layer.hpp`` can
    own the post-LN composition exactly once.

``full``
    Optional HuggingFace ``BertAttention`` output: dense, disabled eval-mode
    dropout, residual addition, and attention-output LayerNorm.

The script also evaluates attention probabilities with a public additive mask
and checks that masked-key probabilities are near zero.  The default mask is
``-16.0``: int32 Q12.20 cannot represent the conventional ``-10000.0``
(its range is [-2048, 2048)), while the existing MPC exponential path maps
values below roughly -11.09 to zero.

``--help`` and ``--describe-boundaries`` require only the Python standard
library.  ``--self-test-format`` requires NumPy but no model or network.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path
from typing import Any, Sequence


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
COUNT_HEADER = struct.Struct("<i")
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
QUERY_SCALE_DIVISOR = 8.0
DEFAULT_MODEL = "google-bert/bert-base-uncased"
DEFAULT_MASK_VALUE = -16.0
DEFAULT_TOLERANCE = 0.08


def boundary_document() -> dict[str, Any]:
    return {
        "input": "float32 [batch, sequence_length, 768]",
        "primary_mpc_boundary": "pre-residual",
        "boundaries": {
            "self-context": "QKV -> scaled scores -> public mask -> softmax -> attention_probs @ V",
            "pre-residual": "self-context -> attention.output.dense; excludes residual and LayerNorm",
            "full": "pre-residual -> dropout(eval: identity) -> residual(input) -> attention.output.LayerNorm",
        },
        "query_scaling_contract": {
            "export": "query weight and bias divided by 8 before Q20 quantization",
            "score": "(Q/8) @ K.T; no post-GEMM divide",
        },
        "attention_mask": {
            "shape": "[batch,1,1,sequence_length]",
            "default_masked_value": DEFAULT_MASK_VALUE,
            "reason": "Q12.20-safe and below the MPC exp zero cutoff",
        },
    }


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required; activate the project Python environment") from error
    return np


def quantize_q20(values: Any) -> Any:
    np = _require_numpy()
    scaled = np.rint(np.asarray(values, dtype=np.float64) * SCALE)
    return np.ascontiguousarray(np.clip(scaled, INT32_MIN, INT32_MAX).astype("<i4"))


def write_counted_q20(path: Path, values: Any) -> None:
    np = _require_numpy()
    payload = quantize_q20(values).reshape(-1, order="C")
    if payload.size > INT32_MAX:
        raise OverflowError(f"{path}: element count does not fit int32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(int(payload.size)))
        stream.write(np.ascontiguousarray(payload).tobytes(order="C"))


def read_counted_q20(path: Path, shape: tuple[int, ...] | None = None) -> Any:
    np = _require_numpy()
    with path.open("rb") as stream:
        header = stream.read(COUNT_HEADER.size)
        if len(header) != COUNT_HEADER.size:
            raise ValueError(f"{path}: missing int32 count header")
        count = COUNT_HEADER.unpack(header)[0]
        if count < 0:
            raise ValueError(f"{path}: negative count {count}")
        payload = np.fromfile(stream, dtype="<i4")
    if payload.size != count:
        raise ValueError(f"{path}: header count={count}, payload count={payload.size}")
    restored = payload.astype(np.float64) / SCALE
    if shape is not None:
        expected = int(np.prod(shape))
        if count != expected:
            raise ValueError(f"{path}: contains {count} values, expected {expected} for {shape}")
        restored = restored.reshape(shape)
    return restored


def load_mpc_output(path: Path, expected_shape: tuple[int, ...]) -> Any:
    np = _require_numpy()
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.files)
        if len(keys) != 1:
            raise ValueError(f"{path}: NPZ must contain exactly one array, found {keys}")
        values = archive[keys[0]]
    else:
        values = read_counted_q20(path, expected_shape)
    values = np.asarray(values, dtype=np.float64)
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{path}: MPC output contains NaN or infinity")
    return values


def run_numpy_format_self_test() -> None:
    np = _require_numpy()
    values = np.array(
        [[-16.0, -1.0000003, 0.0], [0.1250002, 1.5, 2047.0]], dtype=np.float32
    )
    with tempfile.TemporaryDirectory(prefix="bert_attention_verify_test_") as directory:
        path = Path(directory) / "tensor.bin"
        write_counted_q20(path, values)
        restored = read_counted_q20(path, tuple(values.shape))
        maximum = float(np.max(np.abs(restored - values.astype(np.float64))))
        if maximum > 0.5 / SCALE + 1.0e-12:
            raise AssertionError(f"Q20 round-trip error too large: {maximum}")
    print(f"NumPy counted-Q20 self-test PASS: max_abs_error={maximum:.9g}")


def _load_model(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "reference generation requires transformers and torch; activate the project venv first"
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
    return model


def _resolve_bert(model: Any, layer_index: int) -> tuple[Any, Any]:
    base = getattr(model, "bert", model)
    if not hasattr(base, "encoder"):
        raise TypeError("loaded checkpoint does not expose a BERT encoder")
    config = base.config
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
    if getattr(config, "position_embedding_type", "absolute") != "absolute":
        raise ValueError("manual reference currently supports absolute-position BERT attention only")
    if not 0 <= layer_index < int(config.num_hidden_layers):
        raise ValueError(f"layer {layer_index} is outside [0,{config.num_hidden_layers})")
    return base, base.encoder.layer[layer_index]


def _manual_attention_reference(layer: Any, hidden_states: Any, additive_mask: Any) -> dict[str, Any]:
    """Compute the exact exported-Q scaling graph in Torch float32."""
    import torch
    import torch.nn.functional as functional

    batch, sequence_length, hidden = hidden_states.shape
    if hidden != HIDDEN_SIZE:
        raise ValueError(f"expected hidden={HIDDEN_SIZE}, got {hidden}")
    self_attention = layer.attention.self

    # The exporter pre-divides query weight AND bias by eight.  The score GEMM
    # therefore has no secret post-scale/truncation operation.
    query_scaled = functional.linear(
        hidden_states,
        self_attention.query.weight / QUERY_SCALE_DIVISOR,
        self_attention.query.bias / QUERY_SCALE_DIVISOR,
    )
    key = self_attention.key(hidden_states)
    value = self_attention.value(hidden_states)

    def split_heads(tensor: Any) -> Any:
        return tensor.view(batch, sequence_length, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    query_scaled = split_heads(query_scaled)
    key = split_heads(key)
    value = split_heads(value)
    scores = torch.matmul(query_scaled, key.transpose(-1, -2))
    scores = scores + additive_mask
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, value)
    context = context.permute(0, 2, 1, 3).contiguous().view(batch, sequence_length, hidden)
    pre_residual = layer.attention.output.dense(context)
    full = layer.attention.output(context, hidden_states)
    return {
        "self-context": context,
        "pre-residual": pre_residual,
        "full": full,
        "attention-probabilities": probabilities,
    }


def _native_huggingface_reference(
    layer: Any, hidden_states: Any, additive_mask: Any
) -> dict[str, Any]:
    self_outputs = layer.attention.self(
        hidden_states=hidden_states,
        attention_mask=additive_mask,
        head_mask=None,
        output_attentions=True,
    )
    context = self_outputs[0]
    probabilities = self_outputs[1]
    pre_residual = layer.attention.output.dense(context)
    full_outputs = layer.attention(
        hidden_states=hidden_states,
        attention_mask=additive_mask,
        head_mask=None,
        output_attentions=True,
    )
    return {
        "self-context": context,
        "pre-residual": pre_residual,
        "full": full_outputs[0],
        "attention-probabilities": probabilities,
    }


def _safe_mask_value(value: float) -> None:
    minimum = INT32_MIN / SCALE
    maximum = INT32_MAX / SCALE
    if not minimum <= value <= maximum:
        raise ValueError(
            f"mask value {value} is outside int32 Q12.20 range [{minimum},{maximum}]; "
            "use -16 instead of the conventional -10000"
        )
    if value >= 0:
        raise ValueError("masked attention value must be negative")


def build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parent / "data/bert_attention_reference"
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace-compatible layer-0 BERT attention references."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="never contact HuggingFace Hub")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--padding-tokens",
        type=int,
        default=16,
        help="mask this many trailing key positions in every sample",
    )
    parser.add_argument("--mask-value", type=float, default=DEFAULT_MASK_VALUE)
    parser.add_argument(
        "--boundary",
        choices=("pre-residual", "full", "both"),
        default="both",
        help="reference outputs to save; pre-residual is the primary MPC boundary",
    )
    parser.add_argument("--mpc-output", type=Path)
    parser.add_argument(
        "--mpc-boundary", choices=("pre-residual", "full"), default="pre-residual"
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--masked-probability-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--describe-boundaries",
        action="store_true",
        help="print module boundaries without importing third-party packages",
    )
    parser.add_argument(
        "--self-test-format",
        action="store_true",
        help="run a NumPy-only counted-Q20 format test and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_boundaries:
        print(json.dumps(boundary_document(), indent=2))
        return 0
    if args.self_test_format:
        run_numpy_format_self_test()
        return 0
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.sequence_length <= 512:
        parser.error("--sequence-length must be in [1,512]")
    if not 0 <= args.padding_tokens < args.sequence_length:
        parser.error("--padding-tokens must be in [0,sequence-length)")
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.tolerance <= 0 or args.masked_probability_tolerance <= 0:
        parser.error("error tolerances must be positive")
    try:
        _safe_mask_value(args.mask_value)
    except ValueError as error:
        parser.error(str(error))

    np = _require_numpy()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; activate the project Python environment") from error

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = _load_model(args)
    _, layer = _resolve_bert(model, args.layer)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch_size, args.sequence_length, HIDDEN_SIZE)
    hidden_states = torch.randn(shape, generator=generator, dtype=torch.float32)
    additive_mask = torch.zeros(
        (args.batch_size, 1, 1, args.sequence_length), dtype=torch.float32
    )
    key_is_masked = torch.zeros(
        (args.batch_size, args.sequence_length), dtype=torch.bool
    )
    if args.padding_tokens:
        additive_mask[..., -args.padding_tokens :] = args.mask_value
        key_is_masked[:, -args.padding_tokens :] = True

    with torch.inference_mode():
        manual = _manual_attention_reference(layer, hidden_states, additive_mask)
        native = _native_huggingface_reference(layer, hidden_states, additive_mask)

    equivalence: dict[str, float] = {}
    for name in ("self-context", "pre-residual", "full", "attention-probabilities"):
        error = float(torch.max(torch.abs(manual[name] - native[name])).item())
        equivalence[name] = error
        if not np.isfinite(error) or error > 2.0e-5:
            raise AssertionError(
                f"query-pre-scaled graph disagrees with HuggingFace at {name}: {error}"
            )

    probabilities = manual["attention-probabilities"]
    if args.padding_tokens:
        expanded_mask = key_is_masked[:, None, None, :].expand_as(probabilities)
        maximum_masked_probability = float(probabilities[expanded_mask].max().item())
        if maximum_masked_probability >= args.masked_probability_tolerance:
            raise AssertionError(
                f"masked attention probability {maximum_masked_probability} is not below "
                f"{args.masked_probability_tolerance}"
            )
    else:
        maximum_masked_probability = 0.0

    arrays = {
        "input": hidden_states.cpu().numpy(),
        "attention-mask": additive_mask.cpu().numpy(),
        "self-context": manual["self-context"].cpu().numpy(),
        "pre-residual": manual["pre-residual"].cpu().numpy(),
        "full": manual["full"].cpu().numpy(),
        "attention-probabilities": probabilities.cpu().numpy(),
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise FloatingPointError(f"{name} reference contains NaN or infinity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bert_layer{args.layer}_attention_seed{args.seed}"
    saved: dict[str, str] = {}

    def save_array(label: str, array: Any) -> None:
        path = args.output_dir / f"{stem}_{label}.npy"
        np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
        reloaded = np.load(path, allow_pickle=False)
        if not np.array_equal(reloaded, np.asarray(array, dtype=np.float32)):
            raise AssertionError(f"{path}: NPY round-trip mismatch")
        saved[label] = path.name

    save_array("input", arrays["input"])
    save_array("attention_mask", arrays["attention-mask"])
    save_array("self_context", arrays["self-context"])
    save_array("attention_probabilities", arrays["attention-probabilities"])
    if args.boundary in ("pre-residual", "both"):
        save_array("pre_residual", arrays["pre-residual"])
    if args.boundary in ("full", "both"):
        save_array("full", arrays["full"])

    q20_input = args.output_dir / f"{stem}_input_q20.bin"
    q20_mask = args.output_dir / f"{stem}_attention_mask_q20.bin"
    write_counted_q20(q20_input, arrays["input"])
    write_counted_q20(q20_mask, arrays["attention-mask"])
    saved["input_q20"] = q20_input.name
    saved["attention_mask_q20"] = q20_mask.name
    if args.boundary in ("pre-residual", "both"):
        q20_pre_residual = args.output_dir / f"{stem}_pre_residual_q20.bin"
        write_counted_q20(q20_pre_residual, arrays["pre-residual"])
        saved["pre_residual_q20"] = q20_pre_residual.name
    if args.boundary in ("full", "both"):
        q20_full = args.output_dir / f"{stem}_full_q20.bin"
        write_counted_q20(q20_full, arrays["full"])
        saved["full_q20"] = q20_full.name

    comparison: dict[str, Any] | None = None
    if args.mpc_output is not None:
        expected = arrays[args.mpc_boundary]
        actual = load_mpc_output(args.mpc_output, tuple(expected.shape))
        absolute = np.abs(actual - expected.astype(np.float64))
        maximum = float(np.max(absolute))
        mean = float(np.mean(absolute))
        comparison = {
            "mpc_output": str(args.mpc_output),
            "boundary": args.mpc_boundary,
            "max_abs_error": maximum,
            "mean_abs_error": mean,
            "tolerance": args.tolerance,
            "passed": bool(maximum < args.tolerance),
        }
        if not comparison["passed"]:
            raise AssertionError(
                f"MPC {args.mpc_boundary} max_abs_error={maximum:.9g} is not below {args.tolerance}"
            )

    metadata = {
        "model": args.model,
        "revision": args.revision,
        "encoder_layer": args.layer,
        "seed": args.seed,
        "dtype": "float32",
        "output_directory": str(args.output_dir.resolve()),
        "input_shape": list(shape),
        "primary_mpc_boundary": "pre-residual",
        "saved_boundary": args.boundary,
        "boundary_definitions": boundary_document()["boundaries"],
        "query_scaling": boundary_document()["query_scaling_contract"],
        "attention_mask": {
            "shape": list(additive_mask.shape),
            "masked_value": args.mask_value,
            "padding_tokens_per_sample": args.padding_tokens,
            "max_masked_probability": maximum_masked_probability,
            "tolerance": args.masked_probability_tolerance,
            "note": "-16 is Q12.20-safe; -10000 is not representable in a 32-bit Q20 ring",
        },
        "manual_pre_scaled_q_vs_huggingface_max_abs": equivalence,
        "files": saved,
        "mpc_comparison": comparison,
    }
    metadata_path = args.output_dir / f"{stem}_reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"input={saved['input']} shape={shape}")
    print(
        f"primary_boundary=pre-residual mask={args.mask_value:g} "
        f"max_masked_probability={maximum_masked_probability:.9g}"
    )
    print(
        "pre-scaled-Q equivalence: "
        + ", ".join(f"{name}={error:.9g}" for name, error in equivalence.items())
    )
    for label, filename in saved.items():
        print(f"{label}={args.output_dir / filename}")
    if comparison is not None:
        print(
            f"MPC {comparison['boundary']} max_abs_error={comparison['max_abs_error']:.9g} "
            f"mean_abs_error={comparison['mean_abs_error']:.9g} PASS"
        )
    print(f"metadata={metadata_path} finite=True PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
