# BERT LayerNorm MPC implementation report

Date: 2026-07-15 (Asia/Shanghai)

## Environment

- Three NVIDIA A10 24 GB hosts, Ubuntu 22.04, 8 cores and 32 GB RAM
- hpmpc, `PROTOCOL=5` (Trio semi-honest 3PC), `DATTYPE=512`
- Q12.20 fixed point: `BITLENGTH=32`, `FRACTIONAL=20`, `TRUNC_APPROACH=2`
- Recommended build: `PROCESS_NUM=8 SPLITROLES=1 USE_CUDA_GEMM=2 arch=sm_86`
- P0 is the source repository; P1/P2 are synchronized from P0

## Files and integration

- Implementation: `programs/functions/transformer/layernorm_mpc.hpp`
- Verifier: `tests/verify_layernorm.py`
- Existing BatchNorm reference: `nn/PIGEON/headers/batch_normalization_1d_layer.h`
- Dispatch change:

```diff
 #elif FUNCTION_IDENTIFIER == 200
 #include "programs/functions/transformer/softmax_mpc.hpp"
+#elif FUNCTION_IDENTIFIER == 201
+#include "programs/functions/transformer/layernorm_mpc.hpp"
 #elif FUNCTION_IDENTIFIER < 400
```

The BatchNorm class cannot supply LayerNorm statistics because its inference path consumes public moving mean/variance. The implementation reuses its safe inference pattern—public affine parameters, arithmetic shares, exact truncation, and bias addition—but computes mean and variance online per token.

## Protocol

For `R = batch * sequence`, `H = 768`, and `L = DATTYPE / BITLENGTH = 16`, define `P = ceil(R/L)` packed arithmetic rows.

1. Mean: local arithmetic reduction over H, then public fixed-point multiplication by `1/H` and exact rescaling.
2. Variance: one vectorized secure square over `P*H`, immediate exact truncation, public scaling by `1/H`, local reduction, one secure square for the mean, then `E[x^2] - mean^2 + epsilon`. Square terms are scaled before reduction to prevent Q12.20 accumulator wraparound.
3. Inverse square root: one batched seven-threshold comparison selects an eight-entry seed LUT. Exactly two Newton updates execute `r = r*(3-z*r*r)/2`.
4. Normalize and affine transform: one vectorized secure multiplication for `(x-mean)*r`, a public per-column gamma multiplication with exact rescaling, and local beta addition.

Every secret product is truncated before its output is consumed by the next product.

## Round and communication model

Let:

- `R_T(n)` / `C_T(n)` be the rounds/bytes of hpmpc exact truncation for `n` packed arithmetic shares. Exact truncation internally uses A2B, a Boolean adder, and B2A, so it is not a zero-round local shift.
- `R_C(7P)` / `C_C(7P)` be the batched LTZ comparison cost for the seven LUT thresholds.
- `w = DATTYPE/8 = 64` bytes per packed share.

The overflow-safe limb multiplication used here has:

```text
R_M(n) = 3*R_T(n) + 1
C_M(n) = 3*C_T(n) + 2*n*w
```

The two 10-bit operand truncations and the cross-term truncation account for the three truncations. The high-high and combined cross products share one communication round and transmit two packed ring elements per item. Public fixed multiplication uses no Beaver multiplication but needs two exact truncations:

```text
R_P(n) = 2*R_T(n)
C_P(n) = 2*C_T(n)
```

Per-step sequential costs of the current implementation are:

| Step | Secret products | Sequential rounds | Communication expression |
|---|---:|---:|---:|
| Mean | 0 | `R_P(256P)+R_T(P)` | `C_P(256P)+C_T(P)` |
| Variance + nonnegative clamp | `P*H + 2P` | `R_M(PH)+R_P(PH)+R_M(P)+R_C(P)+1` | `C_M(PH)+C_P(PH)+C_M(P)+C_C(P)+P*w` |
| LUT + two Newton updates | `6P` | `R_C(7P)+6R_M(P)+2R_T(P)` | `C_C(7P)+6C_M(P)+2C_T(P)` |
| Normalize + gamma/beta | `P*H` | `R_M(PH)+R_P(PH)` | `C_M(PH)+C_P(PH)` |

The formula supplied in the task needs three general secret multiplications per Newton update: `r*r`, `z*r^2`, and `r*correction`. Counting only two would omit the final secret multiplication. Across the full operator, the logical secret multiplication count is:

```text
R * (2H + 8) = 1544R
```

Input sharing and output reveal add two harness rounds and are not included in the operator table.

## LUT correction

The requested operation “shift the integer part of variance right by 28 bits” cannot be performed locally on additive shares. More importantly, for Q12.20 it maps every normal BERT variance below 256—including the expected range around 1—to index zero, so two Newton updates do not have a useful magnitude seed.

The implementation therefore performs one batched secure comparison against seven public thresholds (`0.25, 0.5, 0.75, 1, 1.5, 2, 4`) and selects one of eight public seeds. Values remain arithmetic shares before and after the existing LTZ/A2B primitive. This is a deliberate correctness correction, not an undocumented change.

## Verification

Compilation succeeded for both the three-party single-process diagnostic build and the recommended `PROCESS_NUM=8 SPLITROLES=1` build. Final real-three-host Trio checks produced:

| Case | Shape | Description | Maximum absolute error | Result |
|---|---:|---|---:|---|
| T1 | `[1,1,4]` | `[1,2,3,4]`, gamma=1, beta=0 | 0.000000715 | PASS |
| T2 | `[4,128,768]` | normal seed 42 | 0.001660824 | PASS (`atol=0.01`) |
| T3 | `[4,128,768]` | official BERT layer-0 attention LayerNorm gamma/beta | 0.003160477 | PASS (`atol=0.01`) |
| T4 | `[1,1,768]` | constant 5.0 | 0.000020981 | PASS (`atol=0.001`) |

The full logs and measured communication comparison are in `tests/results/layernorm_3pc_20260715/`.
