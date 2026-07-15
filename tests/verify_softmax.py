#!/usr/bin/env python3
"""Generate and verify real-3PC fixtures for FUNCTION_IDENTIFIER=200."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np


TENSOR_HEADER = struct.Struct("<4I")
PART_HEADER = struct.Struct("<6I")
CASES = {
    "T1": {"shape": (1, 1, 4, 4), "seed": 0, "atol": 1.0e-3},
    "T2": {"shape": (1, 1, 8, 8), "seed": 42, "atol": 1.0e-2},
    "T3": {"shape": (2, 12, 128, 128), "seed": 42, "atol": 1.0e-2},
    "T4": {"shape": (1, 1, 128, 128), "seed": 0, "atol": 1.0e-2},
}


def build_case(case: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to generate fixtures; --check-only can use an existing reference without it"
        ) from error

    spec = CASES[case]
    shape = spec["shape"]
    if case == "T1":
        logits = torch.zeros(shape, dtype=torch.float32)
    elif case in {"T2", "T3"}:
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        logits = torch.randn(shape, generator=generator, dtype=torch.float32)
    else:
        logits = torch.zeros(shape, dtype=torch.float32)
        flat = logits.reshape(-1, shape[-1])
        rows = torch.arange(flat.shape[0])
        flat[rows, rows.remainder(shape[-1])] = 10.0
    reference = torch.softmax(logits, dim=-1)
    return logits.numpy(), reference.numpy()


def write_tensor(path: Path, value: np.ndarray) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(TENSOR_HEADER.pack(*value.shape))
        stream.write(value.tobytes(order="C"))


def read_full_tensor(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with path.open("rb") as stream:
        actual_shape = TENSOR_HEADER.unpack(stream.read(TENSOR_HEADER.size))
        payload = stream.read()
    if actual_shape != shape:
        raise ValueError(f"output shape is {actual_shape}, expected {shape}")
    expected = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError(f"output payload has {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype="<f4").reshape(shape).copy()


def part_sort_key(path: Path) -> int:
    match = re.search(r"\.part-(\d+)-of-(\d+)\.bin$", path.name)
    if match is None:
        raise ValueError(f"invalid part filename: {path}")
    return int(match.group(1))


def read_partitioned_tensor(base: Path, shape: tuple[int, ...]) -> np.ndarray:
    parts = sorted(base.parent.glob(base.name + ".part-*-of-*.bin"), key=part_sort_key)
    if not parts:
        raise FileNotFoundError(f"no full output or output parts found for {base}")
    rows = int(np.prod(shape[:-1]))
    width = shape[-1]
    merged = np.empty((rows, width), dtype=np.float32)
    covered = np.zeros(rows, dtype=bool)
    expected_parts = None
    for path in parts:
        with path.open("rb") as stream:
            header = PART_HEADER.unpack(stream.read(PART_HEADER.size))
            payload = stream.read()
        tensor_shape = header[:4]
        row_begin, row_count = header[4:]
        if tensor_shape != shape:
            raise ValueError(f"{path}: shape {tensor_shape}, expected {shape}")
        if row_begin + row_count > rows or covered[row_begin : row_begin + row_count].any():
            raise ValueError(f"{path}: overlapping or out-of-range rows")
        expected_bytes = row_count * width * np.dtype("<f4").itemsize
        if len(payload) != expected_bytes:
            raise ValueError(f"{path}: payload has {len(payload)} bytes, expected {expected_bytes}")
        merged[row_begin : row_begin + row_count] = np.frombuffer(payload, dtype="<f4").reshape(row_count, width)
        covered[row_begin : row_begin + row_count] = True
        match = re.search(r"-of-(\d+)\.bin$", path.name)
        count = int(match.group(1)) if match else 0
        expected_parts = count if expected_parts is None else expected_parts
        if count != expected_parts:
            raise ValueError("inconsistent output part counts")
    if expected_parts != len(parts) or not covered.all():
        raise ValueError(f"incomplete output: found {len(parts)}/{expected_parts} parts, covered {covered.sum()}/{rows} rows")
    return merged.reshape(shape)


def read_output(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return read_full_tensor(path, shape) if path.exists() else read_partitioned_tensor(path, shape)


def verify(case: str, output_path: Path, reference: np.ndarray, atol: float) -> dict[str, float | bool | str]:
    mpc = read_output(output_path, reference.shape)
    absolute = np.abs(mpc - reference)
    row_sums = mpc.sum(axis=-1)
    metrics: dict[str, float | bool | str] = {
        "case": case,
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
        "finite": bool(np.isfinite(mpc).all()),
        "nonnegative": bool((mpc >= -atol).all()),
        "atol": atol,
    }
    if case == "T1":
        metrics["max_uniform_error"] = float(np.max(np.abs(mpc - 0.25)))
    if case == "T4":
        flat = mpc.reshape(-1, mpc.shape[-1])
        rows = np.arange(flat.shape[0])
        maxima = flat[rows, rows % flat.shape[1]]
        mask = np.ones_like(flat, dtype=bool)
        mask[rows, rows % flat.shape[1]] = False
        metrics["dominant_min"] = float(maxima.min())
        metrics["other_max"] = float(flat[mask].max())
    metrics["passed"] = bool(
        metrics["finite"] and metrics["nonnegative"] and metrics["max_abs_error"] < atol
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="T3")
    parser.add_argument("--input", type=Path, default=Path("/tmp/hpmpc_softmax_input.bin"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/hpmpc_softmax_output.bin"))
    parser.add_argument("--reference", type=Path, default=Path("/tmp/hpmpc_softmax_reference.npy"))
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--atol", type=float)
    args = parser.parse_args()

    logits = None
    generated_reference = None
    if not args.check_only:
        logits, generated_reference = build_case(args.case)
        write_tensor(args.input, logits)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.reference, generated_reference)
        print(f"generated {args.case}: shape={logits.shape} input={args.input}")
    if args.generate_only:
        return 0

    if args.reference.exists():
        reference = np.load(args.reference)
    else:
        if generated_reference is None:
            _, generated_reference = build_case(args.case)
        reference = generated_reference
    atol = args.atol if args.atol is not None else float(CASES[args.case]["atol"])
    metrics = verify(args.case, args.output, reference, atol)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print("PASSED" if metrics["passed"] else "FAILED")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
