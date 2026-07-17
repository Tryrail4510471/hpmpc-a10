#!/usr/bin/env python3
"""Compare reconstructed FID=203 logits with a saved HuggingFace reference."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


COUNT = struct.Struct("<I")
SCALE = 1 << 20


def read_counted_q20(path: Path):
    import numpy as np

    with path.open("rb") as stream:
        header = stream.read(COUNT.size)
        if len(header) != COUNT.size:
            raise ValueError(f"{path}: missing count header")
        count = COUNT.unpack(header)[0]
        payload = np.fromfile(stream, dtype="<i4")
    if payload.size != count:
        raise ValueError(f"{path}: header count={count}, payload count={payload.size}")
    return payload.astype(np.float64) / SCALE


def read_counted_u8(path: Path):
    import numpy as np

    with path.open("rb") as stream:
        header = stream.read(COUNT.size)
        if len(header) != COUNT.size:
            raise ValueError(f"{path}: missing count header")
        count = COUNT.unpack(header)[0]
        payload = np.fromfile(stream, dtype="u1")
    if payload.size != count:
        raise ValueError(f"{path}: header count={count}, payload count={payload.size}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--mpc-output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument(
        "--boundary-dir", type=Path,
        help="optional FID=203 debug-boundary directory",
    )
    parser.add_argument("--boundary-tolerance", type=float, default=0.12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.tolerance <= 0 or args.boundary_tolerance <= 0:
        parser.error("tolerances must be positive")

    import numpy as np

    reference = np.load(args.reference_dir / "logits.npy", allow_pickle=False).astype(np.float64)
    actual = read_counted_q20(args.mpc_output)
    if actual.size != reference.size:
        raise ValueError(f"MPC has {actual.size} logits, reference has {reference.size}")
    actual = actual.reshape(reference.shape)
    absolute = np.abs(actual - reference)
    reference_prediction = reference.argmax(axis=-1)
    mpc_prediction = actual.argmax(axis=-1)
    report = {
        "reference": str(args.reference_dir / "logits.npy"),
        "mpc_output": str(args.mpc_output),
        "shape": list(reference.shape),
        "reference_logits": reference.tolist(),
        "mpc_logits": actual.tolist(),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "finite": bool(np.isfinite(actual).all()),
        "reference_prediction": reference_prediction.tolist(),
        "mpc_prediction": mpc_prediction.tolist(),
        "prediction_match": bool(np.array_equal(reference_prediction, mpc_prediction)),
        "tolerance": args.tolerance,
    }
    boundary_passed = True
    if args.boundary_dir is not None:
        archive_path = args.reference_dir / "encoder_boundaries.npz"
        mask_path = args.reference_dir / "attention_mask_u8.bin"
        with np.load(archive_path, allow_pickle=False) as archive:
            keys = tuple(sorted(archive.files))
            if keys != tuple(f"boundary_{index:02d}" for index in range(13)):
                raise ValueError(f"{archive_path}: expected exactly 13 encoder boundaries")
            mask = read_counted_u8(mask_path).astype(bool)
            boundary_metrics = []
            for index in range(13):
                filename = (
                    "boundary_00_embedding_q20.bin"
                    if index == 0 else f"boundary_{index}_q20.bin"
                )
                expected = archive[f"boundary_{index:02d}"].astype(np.float64)
                observed = read_counted_q20(args.boundary_dir / filename)
                if observed.size != expected.size:
                    raise ValueError(
                        f"{filename}: {observed.size} elements, expected {expected.size}"
                    )
                observed = observed.reshape(expected.shape)
                error = np.abs(observed - expected)
                valid = error.reshape(-1, expected.shape[-1])[mask]
                metric = {
                    "boundary": index,
                    "max_abs_error": float(error.max()),
                    "mean_abs_error": float(error.mean()),
                    "valid_token_max_abs_error": float(valid.max()),
                    "cls_max_abs_error": float(error[:, 0, :].max()),
                    "finite": bool(np.isfinite(observed).all()),
                }
                boundary_metrics.append(metric)
        maximum_valid = max(item["valid_token_max_abs_error"] for item in boundary_metrics)
        boundary_passed = bool(
            all(item["finite"] for item in boundary_metrics)
            and maximum_valid < args.boundary_tolerance
        )
        report.update(
            boundary_metrics=boundary_metrics,
            boundary_valid_token_max_abs_error=maximum_valid,
            boundary_tolerance=args.boundary_tolerance,
            boundary_passed=boundary_passed,
        )
    report["passed"] = bool(
        report["finite"] and report["max_abs_error"] < args.tolerance
        and report["prediction_match"] and boundary_passed
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
