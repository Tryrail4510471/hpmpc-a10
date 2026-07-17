#!/usr/bin/env python3
"""Generate and verify fixtures for BertEmbedding FUNCTION_IDENTIFIER=207.

The public inputs use the exact C++ harness contract::

    uint32 element_count
    uint32 values[element_count]

Both token-id files must be copied unchanged to P0/P1/P2.  The private model
parameters remain in the full-export directory on MODELOWNER.  A revealed MPC
output is a counted-Q20 tensor (uint32 count followed by little-endian int32).

Without ``--mpc-output`` this command is a plaintext/Q20 preflight only.  With
``--mpc-output`` it compares the real three-party reveal against both native
HuggingFace BertEmbeddings and the quantized real-arithmetic proxy.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
HIDDEN_SIZE = 768
MAX_POSITIONS = 512
COUNT_HEADER = struct.Struct("<I")
# FID=207 defaults to the MRPC full-export bundle.  Using the unfine-tuned
# Google checkpoint as the reference silently compares against different
# embedding and LayerNorm parameters (about 5e-2 error after fine-tuning).
DEFAULT_MODEL = "textattack/bert-base-uncased-MRPC"
DEFAULT_THRESHOLD = 0.01


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required; activate the hpmpc Python environment") from error
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; activate the hpmpc Python environment") from error
    return torch


def quantize_q20(value: Any) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(value, dtype=torch.float64)
    return torch.clamp(torch.round(tensor * SCALE), INT32_MIN, INT32_MAX).to(torch.int64)


def dequantize_q20(value: Any) -> Any:
    return value.double() / SCALE


def write_counted_u32(path: Path, values: Any) -> None:
    np = _require_numpy()
    payload = np.ascontiguousarray(values, dtype="<u4").reshape(-1, order="C")
    if payload.size > (1 << 32) - 1:
        raise OverflowError(f"{path}: counted-u32 payload is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(int(payload.size)))
        stream.write(payload.tobytes(order="C"))


def read_counted_q20(path: Path, shape: tuple[int, ...]) -> Any:
    np = _require_numpy()
    expected = int(np.prod(shape))
    with path.open("rb") as stream:
        header = stream.read(COUNT_HEADER.size)
        if len(header) != COUNT_HEADER.size:
            raise ValueError(f"{path}: missing uint32 element-count header")
        count = COUNT_HEADER.unpack(header)[0]
        payload = np.fromfile(stream, dtype="<i4")
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


def pack_token_lanes(value: Any, lanes: int) -> Any:
    """[B,L,H] -> [ceil(B*L/lanes),H,lanes], matching C++."""
    torch = _require_torch()
    batch, sequence, hidden = value.shape
    flat = value.reshape(batch * sequence, hidden)
    packed = torch.zeros(
        ((flat.shape[0] + lanes - 1) // lanes, hidden, lanes), dtype=value.dtype
    )
    for logical_token in range(flat.shape[0]):
        packed[logical_token // lanes, :, logical_token % lanes] = flat[logical_token]
    return packed


def unpack_token_lanes(packed: Any, batch: int, sequence: int) -> Any:
    torch = _require_torch()
    logical_tokens = batch * sequence
    hidden = packed.shape[1]
    flat = torch.empty((logical_tokens, hidden), dtype=packed.dtype)
    for logical_token in range(logical_tokens):
        flat[logical_token] = packed[
            logical_token // packed.shape[2], :, logical_token % packed.shape[2]
        ]
    return flat.reshape(batch, sequence, hidden)


def load_embeddings(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "Transformers is required; activate the hpmpc Python environment"
        ) from error
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only or args.offline),
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model = AutoModel.from_pretrained(args.model, **kwargs).eval()
    bert = getattr(model, "bert", model)
    embeddings = bert.embeddings
    config = bert.config
    if int(config.hidden_size) != HIDDEN_SIZE:
        raise ValueError(f"checkpoint hidden_size={config.hidden_size}, expected {HIDDEN_SIZE}")
    if args.sequence > int(config.max_position_embeddings):
        raise ValueError("sequence exceeds the checkpoint position table")
    return model, embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--lanes", type=int, default=16, help="DATTYPE/BITLENGTH; 512/32=16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/data/bert_embedding_reference")
    )
    parser.add_argument("--mpc-output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.batch, args.sequence, args.lanes) <= 0:
        raise ValueError("batch, sequence and lanes must be positive")
    if args.sequence > MAX_POSITIONS:
        raise ValueError(f"sequence must be <= {MAX_POSITIONS}")
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")

    np = _require_numpy()
    torch = _require_torch()
    torch.manual_seed(args.seed)
    _, embeddings = load_embeddings(args)
    vocabulary = int(embeddings.word_embeddings.num_embeddings)
    token_type_count = int(embeddings.token_type_embeddings.num_embeddings)

    token_ids = torch.randint(
        0, vocabulary, (args.batch, args.sequence), dtype=torch.long
    )
    token_type_ids = torch.randint(
        0, token_type_count, (args.batch, args.sequence), dtype=torch.long
    )
    position_ids = torch.arange(args.sequence, dtype=torch.long).unsqueeze(0).expand(
        args.batch, -1
    )

    with torch.no_grad():
        reference = embeddings(
            input_ids=token_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).double()

        word_q = quantize_q20(embeddings.word_embeddings.weight)[token_ids]
        position_q = quantize_q20(embeddings.position_embeddings.weight)[position_ids]
        type_q = quantize_q20(embeddings.token_type_embeddings.weight)[token_type_ids]
        # C++ performs addition in the 32-bit ring before LayerNorm.
        summed_q = (word_q + position_q + type_q).to(torch.int32).to(torch.int64)
        summed = dequantize_q20(summed_q)
        normalized = torch.nn.functional.layer_norm(
            summed, (HIDDEN_SIZE,), None, None, 1.0e-5
        )
        gamma = dequantize_q20(quantize_q20(embeddings.LayerNorm.weight))
        beta = dequantize_q20(quantize_q20(embeddings.LayerNorm.bias))
        fixed_proxy = normalized * gamma + beta

    packed = pack_token_lanes(summed_q, args.lanes)
    if not torch.equal(unpack_token_lanes(packed, args.batch, args.sequence), summed_q):
        raise AssertionError("token-pack row/lane roundtrip failed")

    shape = (args.batch, args.sequence, HIDDEN_SIZE)
    reference_np = reference.cpu().numpy().astype(np.float32)
    fixed_proxy_np = fixed_proxy.cpu().numpy().astype(np.float32)
    proxy_error = np.abs(fixed_proxy_np.astype(np.float64) - reference_np.astype(np.float64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_ids_path = args.output_dir / "bert_embedding_token_ids_u32.bin"
    token_type_ids_path = args.output_dir / "bert_embedding_token_type_ids_u32.bin"
    reference_path = args.output_dir / "bert_embedding_reference.npy"
    fixed_proxy_path = args.output_dir / "bert_embedding_fixed_proxy.npy"
    write_counted_u32(token_ids_path, token_ids.cpu().numpy())
    write_counted_u32(token_type_ids_path, token_type_ids.cpu().numpy())
    np.save(reference_path, reference_np, allow_pickle=False)
    np.save(fixed_proxy_path, fixed_proxy_np, allow_pickle=False)

    metrics: dict[str, Any] = {
        "model": args.model,
        "shape": list(shape),
        "fractional": FRACTIONAL,
        "seed": args.seed,
        "layout": "logical_token=b*L+position; row=token//16; lane=token%16",
        "public_input_format": "little-endian uint32 count + uint32 payload",
        "token_ids": str(token_ids_path),
        "token_type_ids": str(token_type_ids_path),
        "reference": str(reference_path),
        "fixed_proxy": str(fixed_proxy_path),
        "huggingface_layernorm_eps": float(embeddings.LayerNorm.eps),
        "cpp_layernorm_eps": 1.0e-5,
        "proxy_vs_huggingface_max_abs": float(proxy_error.max()),
        "proxy_vs_huggingface_mean_abs": float(proxy_error.mean()),
        "threshold": args.threshold,
        "mpc_executed": args.mpc_output is not None,
    }

    if args.mpc_output is None:
        metrics["passed"] = bool(proxy_error.max() < args.threshold)
        metrics["scope"] = "plaintext Q20 proxy; no three-party MPC output supplied"
    else:
        mpc = load_mpc_output(args.mpc_output, shape)
        mpc_hf_error = np.abs(mpc - reference_np.astype(np.float64))
        mpc_proxy_error = np.abs(mpc - fixed_proxy_np.astype(np.float64))
        metrics.update(
            {
                "mpc_output": str(args.mpc_output),
                "mpc_vs_huggingface_max_abs": float(mpc_hf_error.max()),
                "mpc_vs_huggingface_mean_abs": float(mpc_hf_error.mean()),
                "mpc_vs_fixed_proxy_max_abs": float(mpc_proxy_error.max()),
                "mpc_vs_fixed_proxy_mean_abs": float(mpc_proxy_error.mean()),
                "no_nan_or_inf": bool(np.isfinite(mpc).all()),
                "passed": bool(mpc_hf_error.max() < args.threshold),
                "scope": "real revealed FUNCTION_IDENTIFIER=207 output",
            }
        )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
