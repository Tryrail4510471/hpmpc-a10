#!/usr/bin/env python3
"""Verify the Q12.20 GELU approximation and an optional revealed MPC output."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np


SCALE = 1 << 20
KEY_POINTS = np.asarray([-5.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0, 5.0])
EDGES = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
COEFFICIENTS_Q20 = np.asarray(
    [
        [-11572, -123823, -442797, -529959],
        [109037, 468294, 526061, -1013],
        [-109040, 468302, 522510, -1013],
        [11572, -123823, 1491373, -529959],
    ],
    dtype=np.int64,
)


def exact_gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return x * 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def approximate(x: np.ndarray) -> np.ndarray:
    result = np.where(x >= 4.0, x, 0.0)
    coefficients = COEFFICIENTS_Q20.astype(np.float64) / SCALE
    for index, (left, right) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        selected = (x >= left) & (x < right)
        result[selected] = np.polyval(coefficients[index], x[selected])
    return result


def write_counted_q20(path: Path, values: np.ndarray) -> None:
    raw = np.rint(values * SCALE).clip(-(1 << 31), (1 << 31) - 1).astype("<i4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<I", raw.size))
        stream.write(raw.tobytes())


def read_counted_q20(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = struct.unpack("<I", stream.read(4))[0]
        values = np.fromfile(stream, dtype="<i4")
    if values.size != count:
        raise ValueError(f"{path}: header={count}, payload={values.size}")
    return values.astype(np.float64) / SCALE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-input", type=Path)
    parser.add_argument("--mpc-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    if args.write_input:
        write_counted_q20(args.write_input, KEY_POINTS)

    grid = np.linspace(-8.0, 8.0, 200_001)
    fit_error = float(np.max(np.abs(approximate(grid) - exact_gelu(grid))))
    print(f"approximation_max_abs={fit_error:.9f}")
    for x, expected, actual in zip(KEY_POINTS, exact_gelu(KEY_POINTS), approximate(KEY_POINTS)):
        print(f"x={x:6.2f} expected={expected: .8f} approximation={actual: .8f} error={abs(actual-expected):.8f}")
    if fit_error >= args.threshold:
        raise AssertionError(f"approximation error {fit_error} >= {args.threshold}")

    if args.mpc_output:
        actual = read_counted_q20(args.mpc_output)
        if actual.shape != KEY_POINTS.shape:
            raise ValueError(f"MPC output shape {actual.shape}, expected {KEY_POINTS.shape}")
        mpc_error = float(np.max(np.abs(actual - exact_gelu(KEY_POINTS))))
        print(f"mpc_max_abs={mpc_error:.9f}")
        if not np.isfinite(actual).all() or mpc_error >= args.threshold:
            raise AssertionError(f"MPC GELU error {mpc_error} >= {args.threshold}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
