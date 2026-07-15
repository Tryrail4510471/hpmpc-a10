#!/usr/bin/env python3
"""Generate/check FUNCTION_IDENTIFIER=200 hpmpc Softmax binary fixtures."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


SHAPE = (2, 12, 128, 128)
HEADER = struct.Struct("<4I")


def write_tensor(path: Path, value: np.ndarray) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    if value.shape != SHAPE:
        raise ValueError(f"expected shape {SHAPE}, got {value.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(HEADER.pack(*SHAPE))
        stream.write(value.tobytes(order="C"))


def read_tensor(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        shape = HEADER.unpack(stream.read(HEADER.size))
        payload = stream.read()
    if shape != SHAPE:
        raise ValueError(f"MPC output shape is {shape}, expected {SHAPE}")
    expected_bytes = int(np.prod(SHAPE)) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"MPC output has {len(payload)} payload bytes; expected {expected_bytes}")
    return np.frombuffer(payload, dtype="<f4").reshape(SHAPE).copy()


def generate(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    reference = torch.softmax(logits, dim=-1)
    return logits.numpy(), reference.numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/tmp/hpmpc_softmax_input.bin"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/hpmpc_softmax_output.bin"))
    parser.add_argument("--reference", type=Path, default=Path("/tmp/hpmpc_softmax_reference.npy"))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--row-sum-atol", type=float, default=2.0e-3)
    args = parser.parse_args()

    logits, reference = generate(args.seed)
    write_tensor(args.input, logits)
    np.save(args.reference, reference)
    print(f"generated input: {args.input} shape={SHAPE} seed={args.seed}")
    print(f"saved reference: {args.reference}")
    if args.generate_only:
        return 0

    mpc = read_tensor(args.output)
    max_abs_error = float(np.max(np.abs(mpc - reference)))
    mean_abs_error = float(np.mean(np.abs(mpc - reference)))
    row_sum_error = float(np.max(np.abs(mpc.sum(axis=-1) - 1.0)))
    finite = bool(np.isfinite(mpc).all())
    nonnegative = bool((mpc >= -args.atol).all())

    print(f"max_abs_error={max_abs_error:.8g}")
    print(f"mean_abs_error={mean_abs_error:.8g}")
    print(f"max_row_sum_error={row_sum_error:.8g}")
    print(f"finite={finite} nonnegative={nonnegative}")

    passed = (
        finite
        and nonnegative
        and max_abs_error <= args.atol
        and row_sum_error <= args.row_sum_atol
    )
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
