#!/usr/bin/env python3
"""Prepare Q20 FFN fixtures, merge split-role outputs, and compare them."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

SCALE = 1 << 20
HIDDEN = 768
COUNT = struct.Struct("<I")
CASES = {
    "T1": {"rows": 4, "stage": "core", "reference": "bert_layer0_ffn_relu_core_seed42.npy"},
    "T2": {"rows": 256, "stage": "core", "reference": "bert_layer0_ffn_relu_core_seed42.npy"},
    "T3": {"rows": 256, "stage": "full", "reference": "bert_layer0_ffn_relu_full_seed42.npy"},
    "T4": {"rows": 128, "stage": "full", "reference": "bert_layer0_ffn_relu_full_seed42.npy"},
}


def q20(value: np.ndarray) -> np.ndarray:
    scaled = np.rint(value.astype(np.float64) * SCALE)
    return np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype("<i4")


def write_counted(path: Path, values: np.ndarray) -> None:
    payload = np.asarray(values, dtype="<i4", order="C").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT.pack(payload.size))
        stream.write(payload.tobytes())


def read_counted(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        raw = stream.read(COUNT.size)
        if len(raw) != COUNT.size:
            raise ValueError(f"{path}: missing count header")
        count = COUNT.unpack(raw)[0]
        payload = np.frombuffer(stream.read(), dtype="<i4")
    if payload.size != count:
        raise ValueError(f"{path}: payload has {payload.size} elements, expected {count}")
    return payload.copy()


def part_number(path: Path) -> tuple[int, int]:
    match = re.search(r"\.part-(\d+)-of-(\d+)\.bin$", path.name)
    if match is None:
        raise ValueError(f"invalid output part name: {path}")
    return int(match.group(1)), int(match.group(2))


def read_output(base: Path, rows: int) -> np.ndarray:
    if base.exists():
        payload = read_counted(base)
        if payload.size != rows * HIDDEN:
            raise ValueError(f"{base}: {payload.size} values, expected {rows * HIDDEN}")
        return payload.reshape(rows, HIDDEN).astype(np.float64) / SCALE

    parts = sorted(base.parent.glob(base.name + ".part-*-of-*.bin"), key=lambda p: part_number(p)[0])
    if not parts:
        raise FileNotFoundError(f"no output or parts found for {base}")
    expected_parts = part_number(parts[0])[1]
    if len(parts) != expected_parts:
        raise ValueError(f"incomplete output: found {len(parts)}/{expected_parts} parts")
    merged = np.empty((rows, HIDDEN), dtype=np.float64)
    covered = np.zeros(rows, dtype=bool)
    for path in parts:
        worker, total = part_number(path)
        if total != expected_parts:
            raise ValueError("inconsistent output part totals")
        row_begin = rows * worker // total
        row_end = rows * (worker + 1) // total
        payload = read_counted(path)
        expected = (row_end - row_begin) * HIDDEN
        if payload.size != expected:
            raise ValueError(f"{path}: {payload.size} values, expected {expected}")
        merged[row_begin:row_end] = payload.reshape(row_end - row_begin, HIDDEN) / SCALE
        covered[row_begin:row_end] = True
    if not covered.all():
        raise ValueError(f"output covers only {covered.sum()}/{rows} rows")
    return merged


def select_case(case: str, reference_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    spec = CASES[case]
    full_input = np.load(reference_dir / "bert_layer0_ffn_input_seed42.npy", allow_pickle=False).reshape(-1, HIDDEN)
    reference = np.load(reference_dir / spec["reference"], allow_pickle=False).reshape(-1, HIDDEN)
    hf_path = reference_dir / "bert_layer0_ffn_output_seed42.npy"
    hf_reference = np.load(hf_path, allow_pickle=False).reshape(-1, HIDDEN) if hf_path.exists() else None
    rows = spec["rows"]
    if case == "T1":
        indexes = np.arange(4)
    else:
        indexes = np.arange(rows)
    return full_input[indexes].copy(), reference[indexes].copy(), None if hf_reference is None else hf_reference[indexes].copy()


def metrics(actual: np.ndarray, reference: np.ndarray, atol: float) -> dict[str, float | bool]:
    delta = np.abs(actual - reference.astype(np.float64))
    finite = bool(np.isfinite(actual).all())
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean((actual - reference) ** 2))),
        "finite": finite,
        "atol": atol,
        "passed": bool(finite and delta.max() < atol),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("tests/data/bert_ffn_reference"))
    parser.add_argument("--input-bin", type=Path, default=Path("/tmp/hpmpc_bert_ffn_input_q20.bin"))
    parser.add_argument("--output-bin", type=Path, default=Path("/tmp/hpmpc_bert_ffn_output_q20.bin"))
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-hf", action="store_true",
                        help="also require the native HuggingFace GELU reference to pass")
    args = parser.parse_args()

    value, reference, hf_reference = select_case(args.case, args.reference_dir)
    if not args.check_only:
        write_counted(args.input_bin, q20(value))
        if args.reference_out:
            args.reference_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.reference_out, reference, allow_pickle=False)
        print(f"prepared {args.case}: rows={value.shape[0]} stage={CASES[args.case]['stage']} input={args.input_bin}")
    if args.prepare_only:
        return 0

    actual = read_output(args.output_bin, value.shape[0])
    result: dict[str, object] = {
        "case": args.case,
        "stage": CASES[args.case]["stage"],
        "shape": [1, 4, HIDDEN] if args.case == "T1" else ([2, 128, HIDDEN] if value.shape[0] == 256 else [1, 128, HIDDEN]),
        "relu_reference": metrics(actual, reference, args.atol),
    }
    if hf_reference is not None and CASES[args.case]["stage"] == "full":
        result["huggingface_gelu_reference"] = metrics(actual, hf_reference, args.atol)
        result["reference_contract_note"] = "FID=202 uses ReLU; HuggingFace BertIntermediate uses GELU"
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = bool(result["relu_reference"]["passed"])
    if args.require_hf and "huggingface_gelu_reference" in result:
        passed = passed and bool(result["huggingface_gelu_reference"]["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
