#!/usr/bin/env python3
"""Fit the Q12.20 piecewise-cubic GELU used by hpmpc.

A single cubic on [-4, 4] has about 0.30 maximum error and cannot meet the
requested 0.01 bound.  Four cubic intervals retain low multiplicative depth
while bringing the quantized maximum error below 0.005.
"""

from __future__ import annotations

import numpy as np


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
EDGES = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def main() -> int:
    x = np.linspace(-4.0, 4.0, 100_001, dtype=np.float64)
    exact = gelu_tanh(x)
    single = np.polyfit(x, exact, 3)
    single_error = float(np.max(np.abs(np.polyval(single, x) - exact)))
    print(f"single_cubic_max_error={single_error:.9f} (cannot satisfy <0.01)")

    global_error = 0.0
    for index, (left, right) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        selected = (x >= left) & (x <= right)
        coefficients = np.polyfit(x[selected], exact[selected], 3)
        quantized = np.rint(coefficients * SCALE).astype(np.int64)
        restored = quantized.astype(np.float64) / SCALE
        error = float(np.max(np.abs(np.polyval(restored, x[selected]) - exact[selected])))
        global_error = max(global_error, error)
        a3, a2, a1, a0 = coefficients
        print(
            f"segment={index} range=[{left:g},{right:g}] "
            f"a3={a3:.12f} a2={a2:.12f} a1={a1:.12f} a0={a0:.12f} "
            f"q20={quantized.tolist()} max_error={error:.9f}"
        )
    print(f"piecewise_q20_max_error={global_error:.9f}")
    if not global_error < 0.01:
        raise AssertionError(f"quantized piecewise fit error {global_error} is not below 0.01")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
