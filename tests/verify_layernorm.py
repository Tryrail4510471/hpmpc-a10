#!/usr/bin/env python3
"""Generate and verify fixtures for LayerNorm FUNCTION_IDENTIFIER=201."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np


TENSOR_HEADER = struct.Struct("<3I")
PART_HEADER = struct.Struct("<5I")
EPSILON = 1.0e-5
CASES = {
    "T1": {"shape": (1, 1, 4), "seed": 0, "atol": 1.0e-3},
    "T2": {"shape": (4, 128, 768), "seed": 42, "atol": 1.0e-2},
    "T3": {"shape": (4, 128, 768), "seed": 42, "atol": 1.0e-2},
    "T4": {"shape": (1, 1, 768), "seed": 0, "atol": 1.0e-3},
}


def build_case(case: str, weights_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = CASES[case]
    shape = spec["shape"]
    rng = np.random.default_rng(spec["seed"])
    if case == "T1":
        value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(shape)
        gamma = np.ones(shape[-1], dtype=np.float32)
        beta = np.zeros(shape[-1], dtype=np.float32)
    elif case == "T4":
        value = np.full(shape, 5.0, dtype=np.float32)
        gamma = np.ones(shape[-1], dtype=np.float32)
        beta = np.zeros(shape[-1], dtype=np.float32)
    else:
        value = rng.normal(0.0, 1.0, shape).astype(np.float32)
        gamma = np.ones(shape[-1], dtype=np.float32)
        beta = np.zeros(shape[-1], dtype=np.float32)
        if case == "T3":
            if weights_path is None or not weights_path.exists():
                raise FileNotFoundError("T3 requires --weights from tests/export_ln_weights.py")
            with np.load(weights_path) as weights:
                gamma = np.asarray(weights["gamma"], dtype=np.float32)
                beta = np.asarray(weights["beta"], dtype=np.float32)
            if gamma.shape != (shape[-1],) or beta.shape != (shape[-1],):
                raise ValueError(f"LayerNorm weights must both have shape ({shape[-1]},)")
    value64 = value.astype(np.float64)
    mean = value64.mean(axis=-1, keepdims=True)
    variance = np.mean(value64 * value64, axis=-1, keepdims=True) - mean * mean
    reference = gamma.astype(np.float64) * (value64 - mean) / np.sqrt(variance + EPSILON) + beta.astype(np.float64)
    return value, gamma, beta, reference.astype(np.float32)


def write_fixture(path: Path, value: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(TENSOR_HEADER.pack(*value.shape))
        stream.write(np.asarray(value, dtype="<f4", order="C").tobytes(order="C"))
        stream.write(np.asarray(gamma, dtype="<f4", order="C").tobytes(order="C"))
        stream.write(np.asarray(beta, dtype="<f4", order="C").tobytes(order="C"))


def read_full(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with path.open("rb") as stream:
        actual_shape = TENSOR_HEADER.unpack(stream.read(TENSOR_HEADER.size))
        payload = stream.read()
    if actual_shape != shape:
        raise ValueError(f"output shape is {actual_shape}, expected {shape}")
    expected = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError(f"output payload has {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype="<f4").reshape(shape).copy()


def part_number(path: Path) -> int:
    match = re.search(r"\.part-(\d+)-of-(\d+)\.bin$", path.name)
    if match is None:
        raise ValueError(f"invalid part filename: {path}")
    return int(match.group(1))


def read_parts(base: Path, shape: tuple[int, ...]) -> np.ndarray:
    parts = sorted(base.parent.glob(base.name + ".part-*-of-*.bin"), key=part_number)
    if not parts:
        raise FileNotFoundError(f"no output or output parts found for {base}")
    rows = shape[0] * shape[1]
    hidden = shape[2]
    merged = np.empty((rows, hidden), dtype=np.float32)
    covered = np.zeros(rows, dtype=bool)
    expected_parts = None
    for path in parts:
        with path.open("rb") as stream:
            header = PART_HEADER.unpack(stream.read(PART_HEADER.size))
            payload = stream.read()
        if header[:3] != shape:
            raise ValueError(f"{path}: shape {header[:3]}, expected {shape}")
        row_begin, row_count = header[3:]
        if row_begin + row_count > rows or covered[row_begin : row_begin + row_count].any():
            raise ValueError(f"{path}: overlapping or out-of-range rows")
        expected_bytes = row_count * hidden * np.dtype("<f4").itemsize
        if len(payload) != expected_bytes:
            raise ValueError(f"{path}: payload has {len(payload)} bytes, expected {expected_bytes}")
        merged[row_begin : row_begin + row_count] = np.frombuffer(payload, dtype="<f4").reshape(row_count, hidden)
        covered[row_begin : row_begin + row_count] = True
        match = re.search(r"-of-(\d+)\.bin$", path.name)
        part_count = int(match.group(1)) if match else 0
        expected_parts = part_count if expected_parts is None else expected_parts
        if part_count != expected_parts:
            raise ValueError("inconsistent part counts")
    if expected_parts != len(parts) or not covered.all():
        raise ValueError(f"incomplete output: {len(parts)}/{expected_parts} parts, {covered.sum()}/{rows} rows")
    return merged.reshape(shape)


def verify(case: str, output_path: Path, reference: np.ndarray, atol: float) -> dict[str, float | bool | str]:
    actual = read_full(output_path, reference.shape) if output_path.exists() else read_parts(output_path, reference.shape)
    absolute = np.abs(actual - reference)
    metrics: dict[str, float | bool | str] = {
        "case": case,
        "shape": str(tuple(reference.shape)),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "finite": bool(np.isfinite(actual).all()),
        "atol": atol,
    }
    if case == "T4":
        metrics["max_constant_output_abs"] = float(np.abs(actual).max())
    metrics["passed"] = bool(metrics["finite"] and metrics["max_abs_error"] < atol)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="T3")
    parser.add_argument("--input", type=Path, default=Path("/tmp/hpmpc_layernorm_input.bin"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/hpmpc_layernorm_output.bin"))
    parser.add_argument("--reference", type=Path, default=Path("/tmp/hpmpc_layernorm_reference.npy"))
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--atol", type=float)
    args = parser.parse_args()

    generated_reference = None
    if not args.check_only:
        value, gamma, beta, generated_reference = build_case(args.case, args.weights)
        write_fixture(args.input, value, gamma, beta)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.reference, generated_reference)
        print(f"generated {args.case}: shape={value.shape} input={args.input}")
    if args.generate_only:
        return 0
    if args.reference.exists():
        reference = np.load(args.reference)
    else:
        if generated_reference is None:
            _, _, _, generated_reference = build_case(args.case, args.weights)
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
