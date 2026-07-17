#!/usr/bin/env bash
set -euo pipefail

# Environment:
# - 3x NVIDIA A10 24GB, Ubuntu 22.04, 8 CPU cores/32GB RAM
# - 15Gbps private network, hpmpc PROTOCOL=5 (Trio semi-honest 3PC)
# - sm_86, CUDA 12.x, DATTYPE=512, FRACTIONAL=20
#
# This experiment uses process isolation deliberately. hpmpc's send/receive
# buffers and communication-round counters are process-global and are not safe
# for two protocol threads. The FFN and Softmax jobs therefore use independent
# executables and disjoint BASE_PORT ranges while sharing the same physical GPU
# and NIC. In pipeline mode, FFN(batch N) and Softmax(batch N+1) run together.

usage() {
    cat <<'EOF'
Usage: experiments/pipeline_comparison.sh [options]

Run this script on P0.

Options:
  --prepare-only       Build both port-isolated executables and rsync them.
  --run-only           Reuse prepared executables and only run experiments.
  --repeats N          Repetitions per mode and batch size (default: 5).
  --batches LIST       Comma-separated batches (default: 1,4,8,16,32).
  --result-dir PATH    Result directory relative to the repository.
  -h, --help           Show this help.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DO_PREPARE=1
DO_RUN=1
REPEATS=5
BATCH_LIST="1,4,8,16,32"
RESULT_DIR="results/pipeline_comparison"

P0_IP="${HPMPC_P0_IP:-172.24.178.175}"
P1_IP="${HPMPC_P1_IP:-172.24.178.176}"
P2_IP="${HPMPC_P2_IP:-172.24.178.177}"
P1_SSH="${HPMPC_P1_SSH:-user@172.24.178.176}"
P2_SSH="${HPMPC_P2_SSH:-user@172.24.178.177}"
REMOTE_ROOT="${HPMPC_REMOTE_ROOT:-/home/user/hpmpc}"
PYTHON="${HPMPC_PYTHON:-/home/user/.venvs/hpmpc/bin/python}"
NVCC="${HPMPC_NVCC:-/usr/local/cuda/bin/nvcc}"

SOFTMAX_PORT="${HPMPC_PIPELINE_SOFTMAX_PORT:-21000}"
FFN_PORT="${HPMPC_PIPELINE_FFN_PORT:-31000}"
SOFTMAX_BIN_DIR="executables/pipeline/softmax"
FFN_BIN_DIR="executables/pipeline/ffn"
DATA_DIR="experiments/pipeline_data"
WEIGHTS="${HPMPC_BERT_FFN_WEIGHTS:-nn/Pygeon/models/pretrained/bert_base_uncased_layer0_ffn_q20.bin}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-only) DO_PREPARE=1; DO_RUN=0; shift ;;
        --run-only) DO_PREPARE=0; DO_RUN=1; shift ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --batches) BATCH_LIST="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
IFS=',' read -r -a BATCHES <<< "$BATCH_LIST"
for batch in "${BATCHES[@]}"; do
    [[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch: $batch" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || PYTHON=python3

COMMON_BUILD=(
    -j8 PARTY=all PROTOCOL=5 DATTYPE=512 PROCESS_NUM=1 SPLITROLES=0
    FRACTIONAL=20 TRUNC_APPROACH=2 TRUNC_DELAYED=0
    MODELOWNER=0 DATAOWNER=1 PUBLIC_WEIGHTS=0
)

build_variant() {
    local fid="$1" port="$2" cuda="$3" destination="$4"
    echo "Building FID=$fid BASE_PORT=$port USE_CUDA_GEMM=$cuda"
    make "${COMMON_BUILD[@]}" \
        FUNCTION_IDENTIFIER="$fid" USE_CUDA_GEMM="$cuda" BASE_PORT="$port" \
        arch=sm_86 NVCC="$NVCC"
    mkdir -p "$destination"
    for party in 0 1 2; do
        install -m 0755 "executables/run-P${party}.o" "$destination/run-P${party}.o"
    done
}

prepare_experiment() {
    build_variant 200 "$SOFTMAX_PORT" 0 "$SOFTMAX_BIN_DIR"
    build_variant 202 "$FFN_PORT" 2 "$FFN_BIN_DIR"

    for host in "$P1_SSH" "$P2_SSH"; do
        ssh "$host" "mkdir -p '$REMOTE_ROOT/$SOFTMAX_BIN_DIR' '$REMOTE_ROOT/$FFN_BIN_DIR' '$REMOTE_ROOT/$DATA_DIR' '$REMOTE_ROOT/$RESULT_DIR/raw'"
        rsync -a "$SOFTMAX_BIN_DIR/" "$host:$REMOTE_ROOT/$SOFTMAX_BIN_DIR/"
        rsync -a "$FFN_BIN_DIR/" "$host:$REMOTE_ROOT/$FFN_BIN_DIR/"
    done
}

generate_inputs() {
    local batch="$1"
    local softmax_input="$DATA_DIR/softmax_b${batch}.bin"
    local ffn_input="$DATA_DIR/ffn_b${batch}_q20.bin"
    mkdir -p "$DATA_DIR"
    "$PYTHON" - "$batch" "$softmax_input" "$ffn_input" <<'PY'
import struct
import sys
from pathlib import Path

import numpy as np

batch = int(sys.argv[1])
softmax_path = Path(sys.argv[2])
ffn_path = Path(sys.argv[3])
rng = np.random.default_rng(42 + batch)

logits = rng.standard_normal((batch, 12, 128, 128)).astype("<f4")
with softmax_path.open("wb") as stream:
    stream.write(struct.pack("<4I", *logits.shape))
    stream.write(logits.tobytes(order="C"))

hidden = rng.standard_normal((batch, 128, 768)).astype(np.float64)
scaled = np.rint(hidden * (1 << 20))
q20 = np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype("<i4")
with ffn_path.open("wb") as stream:
    stream.write(struct.pack("<I", q20.size))
    stream.write(q20.tobytes(order="C"))

print(f"generated batch={batch} softmax={logits.shape} ffn={hidden.shape}")
PY

    for host in "$P1_SSH" "$P2_SSH"; do
        rsync -a "$softmax_input" "$ffn_input" "$host:$REMOTE_ROOT/$DATA_DIR/"
    done
}

remote_exec() {
    local host="$1"
    shift
    local command="" arg
    for arg in "$@"; do
        printf -v command '%s %q' "$command" "$arg"
    done
    ssh "$host" "cd '$REMOTE_ROOT' &&${command}"
}

party_addresses() {
    local party="$1"
    case "$party" in
        0) printf '%s\n%s\n' "$P1_IP" "$P2_IP" ;;
        1) printf '%s\n%s\n' "$P0_IP" "$P2_IP" ;;
        2) printf '%s\n%s\n' "$P0_IP" "$P1_IP" ;;
    esac
}

run_party() {
    local stage="$1" party="$2" batch="$3"
    local host="" bin input output
    local -a addresses env_args command
    mapfile -t addresses < <(party_addresses "$party")

    if [[ "$stage" == "softmax" ]]; then
        bin="$SOFTMAX_BIN_DIR/run-P${party}.o"
        input="$DATA_DIR/softmax_b${batch}.bin"
        output="$RESULT_DIR/softmax_output.bin"
        env_args=(
            "HPMPC_SOFTMAX_INPUT=$input"
            "HPMPC_SOFTMAX_OUTPUT=$output"
            "HPMPC_SOFTMAX_TOTAL_WORKERS=1"
        )
    else
        bin="$FFN_BIN_DIR/run-P${party}.o"
        input="$DATA_DIR/ffn_b${batch}_q20.bin"
        output="$RESULT_DIR/ffn_output_q20.bin"
        env_args=(
            "CUDA_VISIBLE_DEVICES=0"
            "HPMPC_BERT_FFN_INPUT=$input"
            "HPMPC_BERT_FFN_OUTPUT=$output"
            "HPMPC_BERT_FFN_WEIGHTS=$WEIGHTS"
            "HPMPC_BERT_FFN_ROWS=$((batch * 128))"
            "HPMPC_BERT_FFN_TOTAL_WORKERS=1"
            "HPMPC_BERT_FFN_STAGE=core"
        )
    fi
    command=(env "${env_args[@]}" "$bin" "${addresses[@]}")
    case "$party" in
        0) (cd "$REPO_ROOT" && "${command[@]}") ;;
        1) host="$P1_SSH"; remote_exec "$host" "${command[@]}" ;;
        2) host="$P2_SSH"; remote_exec "$host" "${command[@]}" ;;
    esac
}

launch_triplet() {
    local stage="$1" batch="$2" tag="$3"
    local -a pids=()
    local party status=0
    mkdir -p "$RESULT_DIR/raw"
    for party in 0 1 2; do
        run_party "$stage" "$party" "$batch" \
            >"$RESULT_DIR/raw/${tag}_${stage}_p${party}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    return "$status"
}

run_pair() {
    local mode="$1" batch="$2" repeat="$3"
    local tag="${mode}_b${batch}_r${repeat}"
    local start_ns end_ns elapsed_ms softmax_sha256 ffn_sha256
    start_ns="$(date +%s%N)"
    if [[ "$mode" == "serial" ]]; then
        launch_triplet softmax "$batch" "$tag"
        launch_triplet ffn "$batch" "$tag"
    else
        # Two independent Trio sessions: FFN uses BASE_PORT=31000 and Softmax
        # uses BASE_PORT=21000. This is the steady-state slot corresponding to
        # Linear(batch N) || Softmax(batch N+1).
        launch_triplet ffn "$batch" "$tag" &
        local ffn_pid="$!"
        launch_triplet softmax "$batch" "$tag" &
        local softmax_pid="$!"
        local status=0
        if ! wait "$ffn_pid"; then status=1; fi
        if ! wait "$softmax_pid"; then status=1; fi
        [[ "$status" -eq 0 ]] || return "$status"
    fi
    end_ns="$(date +%s%N)"
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    softmax_sha256="$(sha256sum "$RESULT_DIR/softmax_output.bin" | awk '{print $1}')"
    ffn_sha256="$(sha256sum "$RESULT_DIR/ffn_output_q20.bin" | awk '{print $1}')"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$mode" "$batch" "$repeat" "$elapsed_ms" "$softmax_sha256" "$ffn_sha256" \
        | tee -a "$RESULT_DIR/timings.csv"
}

summarize() {
    "$PYTHON" - "$RESULT_DIR/timings.csv" "$RESULT_DIR/summary.csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

source, destination = sys.argv[1:]
groups = defaultdict(list)
hashes = defaultdict(lambda: {"softmax": set(), "ffn": set()})
with open(source, newline="") as stream:
    for row in csv.DictReader(stream):
        key = (row["mode"], int(row["batch"]))
        groups[key].append(float(row["wall_ms"]))
        hashes[int(row["batch"])]["softmax"].add(row["softmax_sha256"])
        hashes[int(row["batch"])]["ffn"].add(row["ffn_sha256"])

rows = []
for batch in sorted({batch for _, batch in groups}):
    serial = groups[("serial", batch)]
    pipeline = groups[("pipeline", batch)]
    serial_mean = statistics.mean(serial)
    pipeline_mean = statistics.mean(pipeline)
    rows.append({
        "batch": batch,
        "serial_mean_ms": f"{serial_mean:.3f}",
        "pipeline_mean_ms": f"{pipeline_mean:.3f}",
        "serial_median_ms": f"{statistics.median(serial):.3f}",
        "pipeline_median_ms": f"{statistics.median(pipeline):.3f}",
        "speedup": f"{serial_mean / pipeline_mean:.4f}",
        "saved_percent": f"{(1.0 - pipeline_mean / serial_mean) * 100.0:.3f}",
        "output_hash_match": len(hashes[batch]["softmax"]) == 1 and len(hashes[batch]["ffn"]) == 1,
    })

with open(destination, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
PY
}

cleanup() {
    local jobs
    jobs="$(jobs -pr || true)"
    if [[ -n "$jobs" ]]; then
        kill $jobs 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$DO_PREPARE" -eq 1 ]]; then
    prepare_experiment
fi

if [[ "$DO_RUN" -eq 1 ]]; then
    [[ -x "$SOFTMAX_BIN_DIR/run-P0.o" && -x "$FFN_BIN_DIR/run-P0.o" ]] || {
        echo "Missing prepared executables; run without --run-only first." >&2
        exit 2
    }
    [[ -r "$WEIGHTS" ]] || { echo "Missing FFN weights: $WEIGHTS" >&2; exit 2; }
    mkdir -p "$RESULT_DIR/raw"
    for host in "$P1_SSH" "$P2_SSH"; do
        ssh "$host" "mkdir -p '$REMOTE_ROOT/$RESULT_DIR/raw'"
    done
    printf 'mode,batch,repeat,wall_ms,softmax_sha256,ffn_sha256\n' > "$RESULT_DIR/timings.csv"
    for batch in "${BATCHES[@]}"; do
        generate_inputs "$batch"
        for repeat in $(seq 1 "$REPEATS"); do
            run_pair serial "$batch" "$repeat"
            run_pair pipeline "$batch" "$repeat"
        done
    done
    summarize
fi
