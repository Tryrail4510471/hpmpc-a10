#!/usr/bin/env python3
"""Generate and verify FID=208 BERT Pooler/classifier test vectors.

The C++ harness consumes a counted-Q20 ``[B,L,768]`` tensor and emits either
``[B,768]`` pooled states or ``[B,2]`` MRPC logits.  This script creates the
input/reference artifacts, mirrors the 10-bit-limb fixed-point graph, and can
compare a real three-party reveal against HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification


DEFAULT_MODEL = "textattack/bert-base-uncased-MRPC"
Q = 20
LIMB = 10
SCALE = 1 << Q
LIMB_BASE = 1 << LIMB
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
HIDDEN = 768
CLASSES = 2
TANH_C1 = 0.9581543010268436
TANH_C3 = -0.2126759995022808
TANH_C5 = 0.024075099934863436
COUNT_HEADER = struct.Struct("<I")


def q20(value: torch.Tensor | float) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    return torch.clamp(torch.round(tensor * SCALE), I32_MIN, I32_MAX).to(torch.int64)


def deq20(value: torch.Tensor) -> torch.Tensor:
    return value.double() / SCALE


def write_counted_q20(path: Path, value: torch.Tensor | np.ndarray) -> None:
    array = np.asarray(torch.as_tensor(value).cpu(), dtype="<i4").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT_HEADER.pack(int(array.size)))
        stream.write(array.tobytes(order="C"))
    if path.stat().st_size != COUNT_HEADER.size + array.size * 4:
        raise IOError(f"{path}: counted-Q20 post-write size mismatch")


def read_counted_q20(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64))
    with path.open("rb") as stream:
        header = stream.read(COUNT_HEADER.size)
        if len(header) != COUNT_HEADER.size:
            raise ValueError(f"{path}: missing counted-Q20 header")
        count = COUNT_HEADER.unpack(header)[0]
        if count != expected:
            raise ValueError(f"{path}: element count {count}, expected {expected}")
        payload = np.frombuffer(stream.read(), dtype="<i4")
    if payload.size != expected:
        raise ValueError(f"{path}: payload has {payload.size} elements, expected {expected}")
    return payload.astype(np.float64).reshape(shape) / SCALE


def mul_limb_q20(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Elementwise proxy for layernorm_mpc::multiply_fixed."""
    left_high = left >> LIMB
    right_high = right >> LIMB
    left_low = left - left_high * LIMB_BASE
    right_low = right - right_high * LIMB_BASE
    return left_high * right_high + ((left_high * right_low + left_low * right_high) >> LIMB)


def gemm_limb_q20(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Proxy for private_gemm_fixed; value=[B,I], weight=[O,I]."""
    value_high = value >> LIMB
    weight_high = weight >> LIMB
    value_low = value - value_high * LIMB_BASE
    weight_low = weight - weight_high * LIMB_BASE
    high_high = value_high @ weight_high.T
    cross = value_high @ weight_low.T + value_low @ weight_high.T
    return high_high + (cross >> LIMB) + bias


def tanh_piecewise_q20(value: torch.Tensor) -> torch.Tensor:
    inside = (value >= q20(-2.0)) & (value < q20(2.0))
    polynomial_x = torch.where(inside, value, torch.zeros_like(value))
    x2 = mul_limb_q20(polynomial_x, polynomial_x)
    inner = q20(TANH_C3) + mul_limb_q20(x2, q20(TANH_C5))
    outer = q20(TANH_C1) + mul_limb_q20(x2, inner)
    polynomial = mul_limb_q20(polynomial_x, outer)
    return torch.where(
        value < q20(-2.0),
        q20(-1.0),
        torch.where(value >= q20(2.0), q20(1.0), polynomial),
    )


def pack_token_lanes(value: torch.Tensor, lanes: int) -> torch.Tensor:
    batch, sequence, hidden = value.shape
    flat = value.reshape(batch * sequence, hidden)
    packed = torch.zeros(((flat.shape[0] + lanes - 1) // lanes, hidden, lanes), dtype=value.dtype)
    for logical_token in range(flat.shape[0]):
        packed[logical_token // lanes, :, logical_token % lanes] = flat[logical_token]
    return packed


def gather_cls_lanes(packed: torch.Tensor, batch: int, sequence: int) -> torch.Tensor:
    """Mirror bert_pooler::detail::gather_cls and return logical [B,H]."""
    cls = torch.empty((batch, packed.shape[1]), dtype=packed.dtype)
    for batch_index in range(batch):
        logical_cls = batch_index * sequence
        cls[batch_index] = packed[logical_cls // packed.shape[2], :, logical_cls % packed.shape[2]]
    return cls


def error_metrics(actual: Any, expected: Any) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {"max_abs": float(np.max(difference)), "mean_abs": float(np.mean(difference))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("pooler", "classifier"), default="pooler")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--classifier-linear-threshold", type=float, default=0.01)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/data/bert_pooler_reference")
    )
    parser.add_argument("--mpc-output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if min(args.batch, args.sequence, args.lanes) <= 0:
        parser.error("batch, sequence and lanes must be positive")
    if args.threshold <= 0 or args.classifier_linear_threshold <= 0:
        parser.error("error thresholds must be positive")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only or args.offline),
    ).eval()
    base = getattr(model, "bert", None)
    classifier = getattr(model, "classifier", None)
    if base is None or base.pooler is None:
        parser.error("selected checkpoint does not expose bert.pooler")
    if classifier is None or tuple(classifier.weight.shape) != (CLASSES, HIDDEN):
        parser.error("selected checkpoint must expose a 768->2 classifier")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(
        args.batch, args.sequence, HIDDEN, generator=generator, dtype=torch.float32
    )
    with torch.inference_mode():
        pooled_reference = base.pooler(hidden).double()
        # Dropout is identity in eval mode, matching the inference graph.
        classifier_reference = classifier(model.dropout(pooled_reference.float())).double()

    hidden_q = q20(hidden)
    packed = pack_token_lanes(hidden_q, args.lanes)
    cls_q = gather_cls_lanes(packed, args.batch, args.sequence)
    if not torch.equal(cls_q, hidden_q[:, 0, :]):
        raise AssertionError("global token-pack CLS gather failed")

    projected_q = gemm_limb_q20(
        cls_q, q20(base.pooler.dense.weight), q20(base.pooler.dense.bias)
    )
    pooled_proxy_q = tanh_piecewise_q20(projected_q)
    pooled_proxy = deq20(pooled_proxy_q)
    classifier_proxy_q = gemm_limb_q20(
        pooled_proxy_q, q20(classifier.weight), q20(classifier.bias)
    )
    classifier_proxy = deq20(classifier_proxy_q)

    # Isolate the two-class Linear from the Pooler tanh approximation. This is
    # the PROMPT-E classifier <0.01 gate; the end-to-end classifier stage uses
    # the separate Pooler threshold.
    classifier_from_exact_q = deq20(
        gemm_limb_q20(q20(pooled_reference), q20(classifier.weight), q20(classifier.bias))
    )
    pooler_proxy_error = error_metrics(pooled_proxy.numpy(), pooled_reference.numpy())
    classifier_proxy_error = error_metrics(
        classifier_proxy.numpy(), classifier_reference.numpy()
    )
    classifier_linear_error = error_metrics(
        classifier_from_exact_q.numpy(), classifier_reference.numpy()
    )

    grid = torch.linspace(-8.0, 8.0, 1_000_001, dtype=torch.float64)
    polynomial = TANH_C1 * grid + TANH_C3 * grid**3 + TANH_C5 * grid**5
    approximate = torch.where(
        grid < -2.0,
        -torch.ones_like(grid),
        torch.where(grid >= 2.0, torch.ones_like(grid), polynomial),
    )
    ideal_tanh_max_abs = float(torch.max(torch.abs(approximate - torch.tanh(grid))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_bin = args.output_dir / "bert_pooler_input_q20.bin"
    pooler_reference_path = args.output_dir / "bert_pooler_reference.npy"
    classifier_reference_path = args.output_dir / "bert_classifier_reference.npy"
    write_counted_q20(input_bin, hidden_q)
    np.save(pooler_reference_path, pooled_reference.float().numpy(), allow_pickle=False)
    np.save(classifier_reference_path, classifier_reference.float().numpy(), allow_pickle=False)
    write_counted_q20(args.output_dir / "bert_pooler_reference_q20.bin", q20(pooled_reference))
    write_counted_q20(
        args.output_dir / "bert_classifier_reference_q20.bin", q20(classifier_reference)
    )
    torch.save(
        {
            "hidden": hidden,
            "pooler_reference": pooled_reference.float(),
            "pooler_fixed_proxy": pooled_proxy.float(),
            "classifier_reference": classifier_reference.float(),
            "classifier_fixed_proxy": classifier_proxy.float(),
        },
        args.output_dir / "pooler_reference.pt",
    )

    reference = pooled_reference if args.stage == "pooler" else classifier_reference
    stage_proxy_error = pooler_proxy_error if args.stage == "pooler" else classifier_proxy_error
    mpc_comparison: dict[str, Any] | None = None
    if args.mpc_output is not None:
        actual = read_counted_q20(args.mpc_output, tuple(reference.shape))
        if not np.isfinite(actual).all():
            raise FloatingPointError(f"{args.mpc_output}: MPC output contains NaN or infinity")
        mpc_error = error_metrics(actual, reference.numpy())
        mpc_comparison = {
            "path": str(args.mpc_output),
            **mpc_error,
            "threshold": args.threshold,
            "passed": bool(mpc_error["max_abs"] < args.threshold),
        }

    passed = bool(
        stage_proxy_error["max_abs"] < args.threshold
        and classifier_linear_error["max_abs"] < args.classifier_linear_threshold
        and ideal_tanh_max_abs <= 0.035973
        and (mpc_comparison is None or mpc_comparison["passed"])
    )
    metrics = {
        "model": args.model,
        "stage": args.stage,
        "input_shape": list(hidden.shape),
        "output_shape": list(reference.shape),
        "fractional": Q,
        "layout": "logical_token=b*L+pos; local Trio share-lane gather for CLS",
        "pooler_fixed_proxy_vs_huggingface": pooler_proxy_error,
        "classifier_fixed_proxy_vs_huggingface": classifier_proxy_error,
        "classifier_linear_only_vs_huggingface": {
            **classifier_linear_error,
            "threshold": args.classifier_linear_threshold,
        },
        "ideal_piecewise_tanh_max_abs": ideal_tanh_max_abs,
        "stage_threshold": args.threshold,
        "mpc_comparison": mpc_comparison,
        "input_q20": str(input_bin),
        "reference_npy": str(
            pooler_reference_path if args.stage == "pooler" else classifier_reference_path
        ),
        "passed": passed,
    }
    (args.output_dir / f"metrics_{args.stage}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
