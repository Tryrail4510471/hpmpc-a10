# LayerNorm MPC real-three-host test report

Test date: 2026-07-15 (Asia/Shanghai)

## Environment

- Cluster: three NVIDIA A10 24 GB hosts, Ubuntu 22.04, 8 CPU cores and 32 GB RAM per host
- Network: private 15 Gbps network; P0 `172.24.178.175`, P1 `172.24.178.176`, P2 `172.24.178.177`
- Framework: hpmpc, `PROTOCOL=5` (Trio semi-honest 3PC)
- CPU/GPU: AVX-512, NVIDIA A10 `sm_86`, CUDA 12.x
- LayerNorm build: `PARTY=all FUNCTION_IDENTIFIER=201 PROTOCOL=5 DATTYPE=512 PROCESS_NUM=8 SPLITROLES=1 FRACTIONAL=20 TRUNC_APPROACH=2 USE_CUDA_GEMM=0`
- Repository: P0 `/home/user/hpmpc`; sources and executables relayed to P1/P2

The 48-worker build was used for T2 and T3. T1 and T4 contain only one logical token row, so they used the CPU-only single-worker build; launching 48 workers for one row would create 47 empty MPC programs and is rejected by the harness.

## Weight source

`tests/export_ln_weights.py` exported:

```text
google-bert/bert-base-uncased
encoder.layer.0.attention.output.LayerNorm
```

The source was the official `model.safetensors` file. Both modern `weight/bias` and legacy `gamma/beta` tensor names are supported. Export artifact SHA-256:

```text
f29862074d67c49160cde26809c9402fe43131e927fb10dca2beb653f05bb5b6
```

Observed parameter ranges:

- gamma: minimum `0.715300381`, maximum `2.99683928`, mean `0.958480537`
- beta: minimum `-4.20503426`, maximum `0.833606124`, mean `-0.0181732774`

## Numerical results

| Case | Shape | Input/parameters | Error threshold | Maximum absolute error | Mean absolute error | Finite | Result |
|---|---:|---|---:|---:|---:|---:|---:|
| T1 | `[1,1,4]` | `[1,2,3,4]`, gamma=1, beta=0 | `<0.001` | 0.000000715 | 0.000000477 | yes | PASS |
| T2 | `[4,128,768]` | standard normal, NumPy seed 42, gamma=1, beta=0 | `<0.01` | 0.001660824 | 0.000195788 | yes | PASS |
| T3 | `[4,128,768]` | seed 42, official layer-0 attention LayerNorm gamma/beta | `<0.01` | 0.003160477 | 0.000187648 | yes | PASS |
| T4 | `[1,1,768]` | constant 5.0, gamma=1, beta=0 | `<0.001` | 0.000020981 | 0.000020981 | yes | PASS |

All outputs were finite. No NaN or infinity was observed. T4 verifies that variance is lower-bounded at zero before epsilon is added and before Newton iteration.

The reference calculation uses float64 accumulation, population variance over the final dimension, epsilon `1e-5`, and the same affine transform as PyTorch/HuggingFace LayerNorm.

## Measured performance

T2 is the performance case. Framework communication is the sum of hpmpc online send and receive counters over all 48 workers on a physical host.

| Metric | P0 | P1 | P2 |
|---|---:|---:|---:|
| hpmpc online send (MB) | 103.477 | 103.477 | 103.477 |
| hpmpc online receive (MB) | 103.477 | 103.477 | 103.477 |
| hpmpc online total (MB/batch) | 206.954 | 206.954 | 206.954 |
| hpmpc total (MB/sample) | 51.738 | 51.738 | 51.738 |
| maximum worker latency (ms) | 522.004 | 525.462 | 525.128 |
| mean worker latency (ms) | 457.811 | 454.791 | 451.384 |

Controller-observed T2 wall time was `897.220 ms`. NIC byte deltas, including TCP and control overhead, were `233.560/233.724/234.116 MB` for P0/P1/P2.

## Softmax comparison

| Operator | Test shape | Samples | Framework communication per node/batch | Communication per sample | Maximum worker latency |
|---|---:|---:|---:|---:|---:|
| Stage 1 Softmax | `[2,12,128,128]` | 2 | 482.641 MB | 241.321 MB | 1457.325 ms |
| LayerNorm | `[4,128,768]` | 4 | 206.954 MB | 51.738 MB | 525.462 ms |

- Raw batch communication ratio: `206.954 / 482.641 = 42.88%`.
- Per-sample communication ratio: `51.738 / 241.321 = 21.44%`.
- LayerNorm therefore meets the requirement of less than 50% of Softmax communication under either comparison.

## Theory versus measurement

For T2, let logical rows `R=4*128=512`, hidden width `H=768`, SIMD lanes `L=512/32=16`, and workers `W=48`. Logical-row sharding gives one padded packed row per worker, so `P=48` packed rows are processed.

Logical secret multiplication count, including the nonnegative-variance mask, is:

```text
R * (2H + 8) = 512 * 1544 = 790,528
```

The executed packed multiplication count is:

```text
P * (2H + 8) = 48 * 1544 = 74,112
```

Let `C_T(n)` be communication for exact truncation of `n` packed shares, `C_C(n)` communication for batched LTZ, `C_M(n)` for the overflow-safe fixed-point multiplication, and `C_P(n)` for public fixed multiplication. In this implementation:

```text
C_M(n) = 3*C_T(n) + 2*n*64 bytes
C_P(n) = 2*C_T(n)
```

The operator communication expression, excluding input sharing and output reveal, is:

```text
mean:       C_P(256P) + C_T(P)
variance:   C_M(PH) + C_P(PH) + C_M(P) + C_C(P) + P*64
inv-sqrt:   C_C(7P) + 6*C_M(P) + 2*C_T(P)
normalize:  C_M(PH) + C_P(PH)
```

The direct arithmetic-product payload lower bound is about `9.48 MB` before truncation and comparisons. The measured `206.954 MB/node` is much larger because exact Q12.20 truncation performs A2B/Boolean-adder/B2A communication. The seed LUT and negative-variance clamp also call LTZ/A2B, so the current implementation is arithmetic-share vectorized but not literally an all-A-path circuit.

## Acceptance

| Requirement | Result |
|---|---|
| T1-T4 pass | PASS |
| T2 maximum absolute error < 0.01 | PASS (`0.001660824`) |
| Communication < 50% of Softmax | PASS (`21.44%` per sample) |
| No numerical anomaly | PASS |

Overall result: **PASS**.

Raw logs and metrics are stored on P0 under:

```text
/home/user/hpmpc/tests/results/layernorm_3pc_20260715/
```
