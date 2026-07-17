#!/usr/bin/env bash
set -uo pipefail

# Controller for one evidence-preserving FID=203 BERT-base Trio run.
#
# The controller may run on P0 or on a workstation that can reach all three
# public SSH addresses. MPC traffic always uses the private 172.24.178.x
# addresses. Private model weights are read only on P0 (MODELOWNER=0); this
# script transfers only the party executable and the three public input files.

usage() {
    cat <<'EOF'
Usage: experiments/run_bert_3pc.sh [options]

Options:
  --task mrpc|rte       Task-specific checkpoint/reference bundle (default: mrpc).
  --run-id ID           Evidence directory identifier (default: UTC timestamp).
  --output-root PATH    Local result root (default: results/bert_3pc).
  --result-dir PATH     Exact per-run directory (perf_benchmark.py adapter).
  --metrics-json PATH   Write the benchmark manifest expected by perf_benchmark.py.
  --fid 203             Full BERT entry point; every other FID is rejected.
  --batch 1             Only batch=1 is currently backed by public input files.
  --mode serial         Pipeline is not implemented and is explicitly rejected.
  --link-gbps N         Effective link rate used for logged NIC-util derivation.
  --no-build            Reuse and validate P0's archived FID=203 binaries.
  --debug-dir PATH      Collect P0 embedding/layer boundary reveals under PATH/ID.
  -h, --help            Show this help.

Fixed audited build contract:
  FUNCTION_IDENTIFIER=203 PROTOCOL=5 DATTYPE=512 PROCESS_NUM=1
  SPLITROLES=0 FRACTIONAL=20 USE_CUDA_GEMM=2 MODELOWNER=0 DATAOWNER=1
EOF
}

TASK="mrpc"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="results/bert_3pc"
NO_BUILD=0
DEBUG_ROOT=""
EXACT_RUN_DIR=""
METRICS_JSON=""
FID=203
BATCH=1
MODE="serial"
LINK_GBPS=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) [[ $# -ge 2 ]] || { echo "--task needs a value" >&2; exit 2; }; TASK="$2"; shift 2 ;;
        --run-id) [[ $# -ge 2 ]] || { echo "--run-id needs a value" >&2; exit 2; }; RUN_ID="$2"; shift 2 ;;
        --output-root) [[ $# -ge 2 ]] || { echo "--output-root needs a value" >&2; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
        --result-dir) [[ $# -ge 2 ]] || { echo "--result-dir needs a value" >&2; exit 2; }; EXACT_RUN_DIR="$2"; shift 2 ;;
        --metrics-json) [[ $# -ge 2 ]] || { echo "--metrics-json needs a value" >&2; exit 2; }; METRICS_JSON="$2"; shift 2 ;;
        --fid) [[ $# -ge 2 ]] || { echo "--fid needs a value" >&2; exit 2; }; FID="$2"; shift 2 ;;
        --batch) [[ $# -ge 2 ]] || { echo "--batch needs a value" >&2; exit 2; }; BATCH="$2"; shift 2 ;;
        --mode) [[ $# -ge 2 ]] || { echo "--mode needs a value" >&2; exit 2; }; MODE="$2"; shift 2 ;;
        --link-gbps) [[ $# -ge 2 ]] || { echo "--link-gbps needs a value" >&2; exit 2; }; LINK_GBPS="$2"; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --debug-dir) [[ $# -ge 2 ]] || { echo "--debug-dir needs a value" >&2; exit 2; }; DEBUG_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$TASK" == "mrpc" || "$TASK" == "rte" ]] || {
    echo "--task must be mrpc or rte" >&2
    exit 2
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "--run-id must contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
}
[[ -n "$OUTPUT_ROOT" ]] || { echo "--output-root must not be empty" >&2; exit 2; }
[[ "$FID" == 203 ]] || {
    echo "UNSUPPORTED_FID: run_bert_3pc.sh only implements full BERT FID=203" >&2
    exit 2
}
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "--batch must be a positive integer" >&2; exit 2; }
[[ "$BATCH" == 1 ]] || {
    echo "UNSUPPORTED_BATCH: only batch=1 has a real synchronized FID=203 input bundle; refusing to duplicate samples" >&2
    exit 2
}
[[ "$MODE" == serial || "$MODE" == pipeline ]] || { echo "--mode must be serial or pipeline" >&2; exit 2; }
[[ "$MODE" == serial ]] || {
    echo "UNSUPPORTED_PIPELINE: no independent FID=203 pipeline scheduler is implemented; refusing to label serial execution as pipeline" >&2
    exit 2
}
python3 - "$LINK_GBPS" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except ValueError as error:
    raise SystemExit("--link-gbps must be numeric") from error
if not math.isfinite(value) or value <= 0:
    raise SystemExit("--link-gbps must be finite and positive")
PY

for command in ssh scp python3; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required controller command is missing: $command" >&2
        exit 2
    }
done

readonly REMOTE_USER="user"
readonly REMOTE_REPO="/home/user/hpmpc"
readonly REMOTE_PYTHON="/home/user/.venvs/hpmpc/bin/python"
readonly ARCHIVE_REL="executables/bert_base_fid203"
readonly ARCHIVE_DIR="$REMOTE_REPO/$ARCHIVE_REL"
readonly REMOTE_RUN_ROOT="$REMOTE_REPO/results/bert_3pc_runs/$RUN_ID"
readonly REMOTE_INPUT_DIR="$REMOTE_RUN_ROOT/inputs"
readonly REMOTE_OUTPUT_DIR="$REMOTE_RUN_ROOT/output"
readonly REMOTE_LOG_DIR="$REMOTE_RUN_ROOT/logs"
readonly REMOTE_DEBUG_DIR="$REMOTE_RUN_ROOT/debug"

declare -ar PUBLIC_IP=("47.99.78.41" "118.178.242.191" "101.37.149.54")
declare -ar INTERNAL_IP=("172.24.178.175" "172.24.178.176" "172.24.178.177")
declare -ar SSH_TARGET=(
    "$REMOTE_USER@${PUBLIC_IP[0]}"
    "$REMOTE_USER@${PUBLIC_IP[1]}"
    "$REMOTE_USER@${PUBLIC_IP[2]}"
)
declare -ar SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout=15
    -o ServerAliveInterval=20
    -o ServerAliveCountMax=3
)

case "$TASK" in
    mrpc) EXPECTED_MODEL="textattack/bert-base-uncased-MRPC" ;;
    rte) EXPECTED_MODEL="textattack/bert-base-uncased-RTE" ;;
esac
readonly EXPECTED_MODEL
readonly WEIGHTS_DIR="$REMOTE_REPO/nn/Pygeon/weights/$TASK"
readonly REFERENCE_DIR="$REMOTE_REPO/tests/data/bert_base_reference/$TASK"

if [[ -n "$EXACT_RUN_DIR" ]]; then
    RUN_DIR="$EXACT_RUN_DIR"
else
    RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
fi
# Canonical local paths make the manifest/log containment contract unambiguous.
RUN_DIR="$(python3 - "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
if [[ -z "$METRICS_JSON" ]]; then
    METRICS_JSON="$RUN_DIR/benchmark_manifest.json"
else
    METRICS_JSON="$(python3 - "$METRICS_JSON" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
fi
[[ "$(dirname "$METRICS_JSON")" == "$RUN_DIR" ]] || {
    echo "--metrics-json must be directly inside --result-dir/run directory so evidence paths stay self-contained" >&2
    exit 2
}
if [[ -e "$RUN_DIR" ]]; then
    unexpected="$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 \
        ! -name runner.stdout.log ! -name runner.stderr.log -print -quit 2>/dev/null)"
    if [[ -n "$unexpected" ]]; then
        echo "refusing to overwrite non-empty evidence directory: $RUN_DIR (found $unexpected)" >&2
        exit 2
    fi
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/state"
if [[ -n "$DEBUG_ROOT" ]]; then
    LOCAL_DEBUG_DIR="$DEBUG_ROOT/$RUN_ID"
    if [[ -e "$LOCAL_DEBUG_DIR" && -n "$(find "$LOCAL_DEBUG_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "refusing to overwrite non-empty debug directory: $LOCAL_DEBUG_DIR" >&2
        exit 2
    fi
    mkdir -p "$LOCAL_DEBUG_DIR"
else
    LOCAL_DEBUG_DIR=""
fi

# A run starts failed and becomes passed only in the final manifest after every
# gate succeeds. An interrupted run therefore never leaves a fabricated PASS.
python3 - "$RUN_DIR/manifest.json" "$RUN_ID" "$TASK" <<'PY'
import json, sys
path, run_id, task = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump({"schema": "hpmpc_bert_3pc_run_v1", "run_id": run_id,
               "task": task, "status": "running", "passed": False}, stream, indent=2)
    stream.write("\n")
PY

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hpmpc-bert-3pc.XXXXXX")"
declare -a CHILD_PIDS=()
cleanup() {
    local pid
    for pid in "${CHILD_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

FAILURE_FILE="$RUN_DIR/state/failures.txt"
: >"$FAILURE_FILE"
fail() {
    printf '%s\n' "$*" | tee -a "$FAILURE_FILE" >&2
}

now_ns() {
    # Portable across GNU/Linux P0 and a macOS controller.
    python3 -c 'import time; print(time.time_ns())'
}

remote_hash() {
    local party="$1" path="$2"
    ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" \
        "test -f '$path' && sha256sum '$path' | cut -d ' ' -f1"
}

copy_from_party() {
    local party="$1" source="$2" destination="$3"
    scp -q "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}:$source" "$destination"
}

copy_to_party() {
    local party="$1" source="$2" destination="$3"
    scp -q "${SSH_OPTIONS[@]}" "$source" "${SSH_TARGET[$party]}:$destination"
}

BUILD_STATUS="not_started"
PRECHECK_STATUS="not_started"
SYNC_STATUS="not_started"
INPUT_HASH_STATUS="not_started"
BINARY_HASH_STATUS="not_started"
OUTPUT_HASH_STATUS="not_run"
VERIFY_STATUS="not_run"
RUN_WALL_MS=""
declare -a PARTY_EXIT=("null" "null" "null")

precheck_cluster() {
    local party
    for party in 0 1 2; do
        if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" \
            "test -d '$REMOTE_REPO' && test -x '$REMOTE_REPO/scripts/run.sh'" \
            >"$RUN_DIR/logs/precheck_p${party}.log" 2>&1; then
            fail "P${party} SSH/repository precheck failed"
            return 1
        fi
    done

    # Task binding is deliberately not configurable: the reference and private
    # weight bundle must describe the same fine-tuned checkpoint.
    if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[0]}" bash -s -- \
        "$REMOTE_PYTHON" "$WEIGHTS_DIR" "$REFERENCE_DIR" "$TASK" "$EXPECTED_MODEL" \
        >"$RUN_DIR/logs/task_binding.log" 2>&1 <<'REMOTE'
set -euo pipefail
python="$1"; weights="$2"; reference="$3"; task="$4"; expected_model="$5"
test -x "$python"
test -d "$weights"
test -d "$reference"
test -f "$weights/manifest.json"
test -f "$weights/pooler_weight.bin"
test -f "$weights/pooler_bias.bin"
test -f "$weights/classifier_${task}_weight.bin"
test -f "$weights/classifier_${task}_bias.bin"
test -f "$reference/reference.json"
test -f "$reference/logits.npy"
for name in token_ids_u32.bin token_type_ids_u32.bin attention_mask_u8.bin; do
    test -f "$reference/$name"
done
"$python" - "$weights/manifest.json" "$reference/reference.json" "$task" "$expected_model" <<'PY'
import json, sys
manifest_path, reference_path, task, expected = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
with open(reference_path, encoding="utf-8") as stream:
    reference = json.load(stream)
assert manifest.get("fractional") == 20, manifest.get("fractional")
assert manifest.get("tasks") == [task], manifest.get("tasks")
models = manifest.get("models", {})
assert models.get("encoder") == expected, models
assert models.get(task) == expected, models
assert reference.get("task") == task, reference.get("task")
assert reference.get("model") == expected, reference.get("model")
assert reference.get("batch") == 1, reference.get("batch")
assert reference.get("sequence_length") == 128, reference.get("sequence_length")
print(f"PASS task={task} model={expected} batch=1 sequence=128 fractional=20")
PY
REMOTE
    then
        fail "P0 task/weights/reference binding precheck failed"
        return 1
    fi

    # A stale private bundle on P1/P2 is a privacy/configuration failure. The
    # non-model parties receive no model tensor from this script.
    for party in 1 2; do
        if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" \
            "test ! -e '$WEIGHTS_DIR/manifest.json'" \
            >"$RUN_DIR/logs/private_weight_guard_p${party}.log" 2>&1; then
            fail "P${party} contains a private $TASK weight bundle; refusing to run"
            return 1
        fi
    done
    return 0
}

build_or_validate() {
    local build_mode="build"
    [[ "$NO_BUILD" -eq 1 ]] && build_mode="reuse"
    if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[0]}" bash -s -- \
        "$REMOTE_REPO" "$ARCHIVE_DIR" "$build_mode" \
        >"$RUN_DIR/logs/build.log" 2>&1 <<'REMOTE'
set -euo pipefail
repo="$1"; archive="$2"; mode="$3"
cd "$repo"
if [[ "$mode" == "build" ]]; then
    PATH="/usr/local/cuda/bin:$PATH" make -j8 \
        PARTY=all FUNCTION_IDENTIFIER=203 PROTOCOL=5 DATTYPE=512 \
        PROCESS_NUM=1 SPLITROLES=0 FRACTIONAL=20 USE_CUDA_GEMM=2 \
        MODELOWNER=0 DATAOWNER=1 TRUNC_APPROACH=2 TRUNC_DELAYED=0 \
        PUBLIC_WEIGHTS=0 PRINT_TIMINGS=1 arch=sm_86 NVCC=/usr/local/cuda/bin/nvcc
    mkdir -p "$archive/flags"
    for party in 0 1 2; do
        install -m 0755 "executables/run-P${party}.o" "$archive/run-P${party}.o"
        install -m 0644 "executables/flags/run-P${party}.macro_flags" \
            "$archive/flags/run-P${party}.macro_flags"
    done
fi

for party in 0 1 2; do
    binary="$archive/run-P${party}.o"
    flags="$archive/flags/run-P${party}.macro_flags"
    test -x "$binary"
    test -s "$flags"
    line="$(cat "$flags")"
    for required in \
        PROTOCOL=5 FUNCTION_IDENTIFIER=203 DATTYPE=512 PROCESS_NUM=1 \
        FRACTIONAL=20 USE_CUDA_GEMM=2 MODELOWNER=0 DATAOWNER=1 \
        TRUNC_APPROACH=2 TRUNC_DELAYED=0 PUBLIC_WEIGHTS=0 \
        "PARTY=$party" SPLIT_ROLES_OFFSET=0; do
        grep -Eq "(^|[[:space:]])-D${required}([[:space:]]|$)" <<<"$line" || {
            echo "P${party} archived flag missing: -D${required}" >&2
            exit 1
        }
    done
done
if find "$archive" -maxdepth 1 -type f -name 'run-P*--*.o' -print -quit | grep -q .; then
    echo "split-role constellation binary found in fixed-role FID203 archive" >&2
    exit 1
fi
sha256sum "$archive"/run-P?.o "$archive"/flags/run-P?.macro_flags
REMOTE
    then
        fail "FID203 build/archive/flag validation failed on P0"
        return 1
    fi
    return 0
}

stage_and_sync() {
    local party filename debug_requested=0
    [[ -n "$DEBUG_ROOT" ]] && debug_requested=1
    if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[0]}" bash -s -- \
        "$REFERENCE_DIR" "$REMOTE_INPUT_DIR" "$REMOTE_OUTPUT_DIR" "$REMOTE_LOG_DIR" \
        "$REMOTE_DEBUG_DIR" "$debug_requested" >"$RUN_DIR/logs/stage_p0.log" 2>&1 <<'REMOTE'
set -euo pipefail
reference="$1"; inputs="$2"; output="$3"; logs="$4"; debug="$5"; debug_requested="$6"
mkdir -p "$inputs" "$output" "$logs"
if [[ "$debug_requested" == "1" ]]; then mkdir -p "$debug"; fi
for name in token_ids_u32.bin token_type_ids_u32.bin attention_mask_u8.bin; do
    install -m 0644 "$reference/$name" "$inputs/$name"
done
REMOTE
    then
        fail "P0 public-input staging failed"
        return 1
    fi

    mkdir -p "$TMP_DIR/inputs" "$TMP_DIR/binaries"
    for filename in token_ids_u32.bin token_type_ids_u32.bin attention_mask_u8.bin; do
        if ! copy_from_party 0 "$REMOTE_INPUT_DIR/$filename" "$TMP_DIR/inputs/$filename"; then
            fail "could not retrieve P0 public input $filename for relay"
            return 1
        fi
    done
    for party in 0 1 2; do
        if ! copy_from_party 0 "$ARCHIVE_DIR/run-P${party}.o" \
            "$TMP_DIR/binaries/run-P${party}.o"; then
            fail "could not retrieve archived P${party} binary from P0"
            return 1
        fi
    done

    for party in 1 2; do
        if ! ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" \
            "mkdir -p '$ARCHIVE_DIR' '$REMOTE_INPUT_DIR' '$REMOTE_OUTPUT_DIR' '$REMOTE_LOG_DIR' '$REMOTE_DEBUG_DIR'" \
            >"$RUN_DIR/logs/stage_p${party}.log" 2>&1; then
            fail "P${party} run-directory creation failed"
            return 1
        fi
        if ! copy_to_party "$party" "$TMP_DIR/binaries/run-P${party}.o" \
            "$ARCHIVE_DIR/run-P${party}.o"; then
            fail "P${party} binary synchronization failed"
            return 1
        fi
        for filename in token_ids_u32.bin token_type_ids_u32.bin attention_mask_u8.bin; do
            if ! copy_to_party "$party" "$TMP_DIR/inputs/$filename" \
                "$REMOTE_INPUT_DIR/$filename"; then
                fail "P${party} public input synchronization failed: $filename"
                return 1
            fi
        done
        ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" \
            "chmod 0755 '$ARCHIVE_DIR/run-P${party}.o'" >/dev/null 2>&1 || return 1
    done
    return 0
}

check_input_hashes() {
    local filename party hash reference_hash="" consistent
    : >"$RUN_DIR/state/input_hashes.tsv"
    for filename in token_ids_u32.bin token_type_ids_u32.bin attention_mask_u8.bin; do
        consistent=1
        reference_hash=""
        printf '%s' "$filename" >>"$RUN_DIR/state/input_hashes.tsv"
        for party in 0 1 2; do
            if ! hash="$(remote_hash "$party" "$REMOTE_INPUT_DIR/$filename")"; then
                hash="ERROR"
                consistent=0
            fi
            [[ -z "$reference_hash" ]] && reference_hash="$hash"
            [[ "$hash" == "$reference_hash" && "$hash" != "ERROR" ]] || consistent=0
            printf '\t%s' "$hash" >>"$RUN_DIR/state/input_hashes.tsv"
        done
        printf '\t%s\n' "$consistent" >>"$RUN_DIR/state/input_hashes.tsv"
        if [[ "$consistent" -ne 1 ]]; then
            fail "public input SHA256 mismatch: $filename"
            return 1
        fi
    done
    return 0
}

check_binary_hashes() {
    local party source_hash target_hash
    : >"$RUN_DIR/state/binary_hashes.tsv"
    for party in 0 1 2; do
        source_hash="$(remote_hash 0 "$ARCHIVE_DIR/run-P${party}.o" 2>/dev/null || true)"
        target_hash="$(remote_hash "$party" "$ARCHIVE_DIR/run-P${party}.o" 2>/dev/null || true)"
        printf 'P%s\t%s\t%s\t%s\n' "$party" "$source_hash" "$target_hash" \
            "$([[ -n "$source_hash" && "$source_hash" == "$target_hash" ]] && echo 1 || echo 0)" \
            >>"$RUN_DIR/state/binary_hashes.tsv"
        if [[ -z "$source_hash" || "$source_hash" != "$target_hash" ]]; then
            fail "P${party} binary SHA256 does not match the P0 archive"
            return 1
        fi
    done
    return 0
}

run_one_party() {
    local party="$1" peer_a peer_b debug_value="-"
    case "$party" in
        0) peer_a="${INTERNAL_IP[1]}"; peer_b="${INTERNAL_IP[2]}" ;;
        1) peer_a="${INTERNAL_IP[0]}"; peer_b="${INTERNAL_IP[2]}" ;;
        2) peer_a="${INTERNAL_IP[0]}"; peer_b="${INTERNAL_IP[1]}" ;;
    esac
    [[ -n "$DEBUG_ROOT" ]] && debug_value="$REMOTE_DEBUG_DIR"
    ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[$party]}" bash -s -- \
        "$REMOTE_REPO" "$ARCHIVE_DIR/run-P${party}.o" "$peer_a" "$peer_b" \
        "$TASK" "$WEIGHTS_DIR" "$REMOTE_INPUT_DIR" "$REMOTE_OUTPUT_DIR" \
        "$REMOTE_LOG_DIR" "$debug_value" "$party" <<'REMOTE'
set -uo pipefail
repo="$1"; binary="$2"; peer_a="$3"; peer_b="$4"; task="$5"; weights="$6"
inputs="$7"; output="$8"; logs="$9"; debug="${10}"; party="${11}"
[[ "$debug" == "-" ]] && debug=""
cd "$repo" || exit 90
mkdir -p "$output" "$logs"
[[ -z "$debug" ]] || mkdir -p "$debug"
unset HPMPC_BERT_BASE_DEBUG_DIR
log="$logs/p${party}.log"
gpu_log="$logs/p${party}_gpu.csv"
exit_file="$logs/p${party}.exit_code"
rm -f "$output/logits_q20.bin" "$exit_file" "$gpu_log"
start_ns="$(date +%s%N)"
env_args=(
    "CUDA_VISIBLE_DEVICES=0"
    "HPMPC_BERT_BASE_TASK=$task"
    "HPMPC_BERT_BASE_WEIGHTS_DIR=$weights"
    "HPMPC_BERT_BASE_BATCH=1"
    "HPMPC_BERT_BASE_SEQUENCE=128"
    "HPMPC_BERT_BASE_TOKEN_IDS=$inputs/token_ids_u32.bin"
    "HPMPC_BERT_BASE_TOKEN_TYPE_IDS=$inputs/token_type_ids_u32.bin"
    "HPMPC_BERT_BASE_ATTENTION_MASK=$inputs/attention_mask_u8.bin"
    "HPMPC_BERT_BASE_OUTPUT=$output/logits_q20.bin"
)
[[ -z "$debug" ]] || env_args+=("HPMPC_BERT_BASE_DEBUG_DIR=$debug")
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 1 >"$gpu_log" 2>&1 &
gpu_monitor_pid=$!
set +e
env "${env_args[@]}" "$binary" "$peer_a" "$peer_b" >"$log" 2>&1
status=$?
kill "$gpu_monitor_pid" 2>/dev/null || true
wait "$gpu_monitor_pid" 2>/dev/null || true
set -e
end_ns="$(date +%s%N)"
printf 'BERT_3PC party=%s task=%s wall_ms=%s exit_code=%s\n' \
    "$party" "$task" "$(( (end_ns - start_ns) / 1000000 ))" "$status" >>"$log"
printf '%s\n' "$status" >"$exit_file"
exit "$status"
REMOTE
}

launch_triplet() {
    local party pid start_ns end_ns
    start_ns="$(now_ns)"
    CHILD_PIDS=()
    for party in 0 1 2; do
        run_one_party "$party" >"$RUN_DIR/logs/ssh_p${party}.log" 2>&1 &
        CHILD_PIDS+=("$!")
    done
    for party in 0 1 2; do
        pid="${CHILD_PIDS[$party]}"
        if wait "$pid"; then
            PARTY_EXIT[$party]=0
        else
            PARTY_EXIT[$party]=$?
        fi
    done
    CHILD_PIDS=()
    end_ns="$(now_ns)"
    RUN_WALL_MS="$(( (end_ns - start_ns) / 1000000 ))"
    printf '%s\n' "$RUN_WALL_MS" >"$RUN_DIR/state/controller_wall_ms"
    printf 'P0\t%s\nP1\t%s\nP2\t%s\n' \
        "${PARTY_EXIT[0]}" "${PARTY_EXIT[1]}" "${PARTY_EXIT[2]}" \
        >"$RUN_DIR/state/party_exit_codes.tsv"
}

collect_run_artifacts() {
    local party
    for party in 0 1 2; do
        if ! copy_from_party "$party" "$REMOTE_LOG_DIR/p${party}.log" \
            "$RUN_DIR/logs/p${party}.log"; then
            fail "could not collect P${party} protocol log"
        fi
        if ! copy_from_party "$party" "$REMOTE_LOG_DIR/p${party}_gpu.csv" \
            "$RUN_DIR/logs/p${party}_gpu.csv"; then
            fail "could not collect P${party} GPU monitor log"
        fi
        copy_from_party "$party" "$REMOTE_LOG_DIR/p${party}.exit_code" \
            "$RUN_DIR/state/p${party}.remote_exit_code" 2>/dev/null || true
    done
    copy_from_party 0 "$REMOTE_OUTPUT_DIR/logits_q20.bin" \
        "$RUN_DIR/logits_q20.bin" 2>/dev/null || true
    if [[ -n "$LOCAL_DEBUG_DIR" ]]; then
        scp -q -r "${SSH_OPTIONS[@]}" "${SSH_TARGET[0]}:$REMOTE_DEBUG_DIR/." \
            "$LOCAL_DEBUG_DIR/" 2>"$RUN_DIR/logs/debug_collect.log" ||
            fail "could not collect P0 debug boundaries"
    fi
}

check_output_hashes() {
    local party hash reference_hash="" consistent=1
    : >"$RUN_DIR/state/output_hashes.tsv"
    for party in 0 1 2; do
        hash="$(remote_hash "$party" "$REMOTE_OUTPUT_DIR/logits_q20.bin" 2>/dev/null || true)"
        [[ -z "$reference_hash" ]] && reference_hash="$hash"
        [[ -n "$hash" && "$hash" == "$reference_hash" ]] || consistent=0
        printf 'P%s\t%s\n' "$party" "$hash" >>"$RUN_DIR/state/output_hashes.tsv"
    done
    if [[ "$consistent" -ne 1 ]]; then
        fail "party output SHA256 files are missing or inconsistent"
        return 1
    fi
    return 0
}

run_verifier() {
    local remote_report="$REMOTE_RUN_ROOT/verify_bert_base.json"
    local remote_log="$REMOTE_LOG_DIR/verify_bert_base.log"
    ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET[0]}" bash -s -- \
        "$REMOTE_REPO" "$REMOTE_PYTHON" "$REFERENCE_DIR" \
        "$REMOTE_OUTPUT_DIR/logits_q20.bin" "$remote_report" "$remote_log" <<'REMOTE'
set -uo pipefail
repo="$1"; python="$2"; reference="$3"; output="$4"; report="$5"; log="$6"
cd "$repo" || exit 90
set +e
"$python" tests/verify_bert_base.py \
    --reference-dir "$reference" --mpc-output "$output" \
    --tolerance 0.5 --report "$report" >"$log" 2>&1
status=$?
set -e
exit "$status"
REMOTE
    local status=$?
    copy_from_party 0 "$remote_report" "$RUN_DIR/verify_bert_base.json" 2>/dev/null || true
    copy_from_party 0 "$remote_log" "$RUN_DIR/logs/verify_bert_base.log" 2>/dev/null || true
    return "$status"
}

write_manifest() {
    python3 - "$RUN_DIR" "$RUN_ID" "$TASK" "$EXPECTED_MODEL" "$BUILD_STATUS" \
        "$PRECHECK_STATUS" "$SYNC_STATUS" "$INPUT_HASH_STATUS" "$BINARY_HASH_STATUS" \
        "$OUTPUT_HASH_STATUS" "$VERIFY_STATUS" "$NO_BUILD" "$RUN_WALL_MS" \
        "$WEIGHTS_DIR" "$REFERENCE_DIR" "$LOCAL_DEBUG_DIR" \
        "${PARTY_EXIT[0]}" "${PARTY_EXIT[1]}" "${PARTY_EXIT[2]}" <<'PY'
from __future__ import annotations
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

(run_dir_s, run_id, task, expected_model, build_status, precheck_status,
 sync_status, input_hash_status, binary_hash_status, output_hash_status,
 verify_status, no_build_s, wall_ms_s, weights_dir, reference_dir,
 debug_dir, *exit_values) = sys.argv[1:]
run_dir = Path(run_dir_s)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_tsv(name: str):
    path = run_dir / "state" / name
    if not path.exists():
        return []
    return [line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8") if line.strip()]

def parse_log(path: Path):
    result = {"latency_ms": None, "send_mb": None}
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    times = re.findall(r"Time measured to perform computation (?:chrono|getTime):\s*([0-9.eE+-]+)s", text)
    if times:
        result["latency_ms"] = float(times[-1]) * 1000.0
    send_line = re.findall(
        r"Sending to other players:\s*([0-9.eE+-]+)([kMGT]?B)\s*,\s*([0-9.eE+-]+)([kMGT]?B)", text
    )
    if send_line:
        scale = {"B": 1 / 1_000_000, "kB": 1 / 1000, "MB": 1, "GB": 1000, "TB": 1_000_000}
        a, au, b, bu = send_line[-1]
        result["send_mb"] = float(a) * scale[au] + float(b) * scale[bu]
    return result

def parse_gpu(path: Path):
    result = {"gpu_peak_memory_mb": None, "gpu_peak_utilization_percent": None,
              "gpu_mean_utilization_percent": None, "gpu_samples": 0}
    if not path.exists():
        return result
    utilization, memory = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            utilization.append(float(fields[1]))
            memory.append(float(fields[2]))
        except ValueError:
            continue
    if utilization:
        result.update({
            "gpu_peak_memory_mb": max(memory),
            "gpu_peak_utilization_percent": max(utilization),
            "gpu_mean_utilization_percent": sum(utilization) / len(utilization),
            "gpu_samples": len(utilization),
        })
    return result

failures_path = run_dir / "state" / "failures.txt"
failures = [] if not failures_path.exists() else [
    line.strip() for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()
]
input_hashes = {}
for row in read_tsv("input_hashes.tsv"):
    if len(row) == 5:
        input_hashes[row[0]] = {"p0": row[1], "p1": row[2], "p2": row[3],
                                "consistent": row[4] == "1"}
binary_hashes = {}
for row in read_tsv("binary_hashes.tsv"):
    if len(row) == 4:
        binary_hashes[row[0].lower()] = {"p0_archive": row[1], "deployed": row[2],
                                         "consistent": row[3] == "1"}
output_hashes = {row[0].lower(): row[1] for row in read_tsv("output_hashes.tsv") if len(row) == 2}

parties = {}
latencies, sends, gpu_peaks = [], [], []
for party, raw_exit in enumerate(exit_values):
    try:
        exit_code = int(raw_exit)
    except ValueError:
        exit_code = None
    log = run_dir / "logs" / f"p{party}.log"
    parsed = parse_log(log)
    gpu = parse_gpu(run_dir / "logs" / f"p{party}_gpu.csv")
    if parsed["latency_ms"] is not None:
        latencies.append(parsed["latency_ms"])
    if parsed["send_mb"] is not None:
        sends.append(parsed["send_mb"])
    if gpu["gpu_peak_memory_mb"] is not None:
        gpu_peaks.append(gpu["gpu_peak_memory_mb"])
    parties[f"p{party}"] = {
        "public_ip": ["47.99.78.41", "118.178.242.191", "101.37.149.54"][party],
        "internal_ip": ["172.24.178.175", "172.24.178.176", "172.24.178.177"][party],
        "exit_code": exit_code,
        "log": str(log),
        "log_sha256": sha256(log) if log.exists() else None,
        **parsed,
        **gpu,
    }

verify_path = run_dir / "verify_bert_base.json"
verification = None
if verify_path.exists():
    try:
        verification = json.loads(verify_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot parse verifier report: {error}")

logs = {}
for path in sorted((run_dir / "logs").glob("*")):
    if path.is_file():
        logs[path.name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

all_exits_zero = all(value.get("exit_code") == 0 for value in parties.values())
all_inputs_equal = bool(input_hashes) and all(value["consistent"] for value in input_hashes.values())
all_binaries_equal = bool(binary_hashes) and all(value["consistent"] for value in binary_hashes.values())
output_values = [value for value in output_hashes.values() if value]
all_outputs_equal = len(output_values) == 3 and len(set(output_values)) == 1
verified = bool(verification and verification.get("passed") is True and verify_status == "passed")
metrics_complete = len(latencies) == 3 and len(sends) == 3 and len(gpu_peaks) == 3
passed = bool(
    not failures and precheck_status == "passed" and build_status in {"built", "reused"}
    and sync_status == "passed" and input_hash_status == "passed"
    and binary_hash_status == "passed" and output_hash_status == "passed"
    and all_exits_zero and all_inputs_equal and all_binaries_equal
    and all_outputs_equal and verified and metrics_complete
)
if not passed and not failures:
    failures.append("one or more acceptance gates did not pass")

metrics = {
    "controller_wall_ms": int(wall_ms_s) if wall_ms_s else None,
    "end_to_end_latency_ms": max(latencies) if len(latencies) == 3 else None,
    "total_send_mb": sum(sends) if len(sends) == 3 else None,
    "cluster_peak_gpu_memory_mb": max(gpu_peaks) if len(gpu_peaks) == 3 else None,
    "complete": metrics_complete,
    "per_party": {
        name: {key: value for key, value in party.items() if key in {
            "latency_ms", "send_mb", "gpu_peak_memory_mb",
            "gpu_peak_utilization_percent", "gpu_mean_utilization_percent", "gpu_samples"
        }} for name, party in parties.items()
    },
}
metrics_path = run_dir / "native_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

manifest = {
    "schema": "hpmpc_bert_3pc_run_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "run_id": run_id,
    "task": task,
    "status": "passed" if passed else "failed",
    "passed": passed,
    "failure_reasons": failures,
    "cluster": {
        "ssh_user": "user", "repo": "/home/user/hpmpc",
        "gpu": "NVIDIA A10 24GB", "network": "15Gbps private",
        "parties": parties,
    },
    "contract": {
        "FUNCTION_IDENTIFIER": 203, "PROTOCOL": 5, "DATTYPE": 512,
        "PROCESS_NUM": 1, "SPLITROLES": 0, "FRACTIONAL": 20,
        "USE_CUDA_GEMM": 2, "MODELOWNER": 0, "DATAOWNER": 1,
        "TRUNC_APPROACH": 2, "TRUNC_DELAYED": 0, "PUBLIC_WEIGHTS": 0,
    },
    "task_binding": {
        "expected_model": expected_model, "weights_dir_p0_only": weights_dir,
        "reference_dir": reference_dir, "batch": 1, "sequence_length": 128,
    },
    "build": {"status": build_status, "no_build": no_build_s == "1",
              "archive": "/home/user/hpmpc/executables/bert_base_fid203",
              "binary_hashes": binary_hashes},
    "public_inputs": {"status": input_hash_status, "hashes": input_hashes},
    "synchronization": {"status": sync_status, "private_weights_transferred": False},
    "execution": {
        **metrics,
        "party_output_sha256": output_hashes,
        "output_hash_status": output_hash_status,
    },
    "metrics_file": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
    "verification": {"status": verify_status, "report": verification},
    "debug": {"enabled": bool(debug_dir), "collected_dir": debug_dir or None},
    "log_files": logs,
}
temporary = run_dir / "manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(run_dir / "manifest.json")
print(json.dumps({"manifest": str(run_dir / "manifest.json"), "passed": passed,
                  "failures": failures}, indent=2))
raise SystemExit(0 if passed else 1)
PY
}

write_benchmark_manifest() {
    python3 - "$RUN_DIR" "$METRICS_JSON" "$FID" "$BATCH" "$MODE" \
        "$LINK_GBPS" "$TASK" <<'PY'
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
fid, batch = int(sys.argv[3]), int(sys.argv[4])
mode, link_gbps, task = sys.argv[5], float(sys.argv[6]), sys.argv[7]
native_path = run_dir / "manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

failure_reasons: list[str] = []
native = {}
if native_path.is_file():
    try:
        native = json.loads(native_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failure_reasons.append(f"cannot parse native manifest: {error}")
else:
    failure_reasons.append("native manifest is missing")

party_logs = [run_dir / "logs" / f"p{party}.log" for party in range(3)]
log_text = "\n".join(
    path.read_text(encoding="utf-8", errors="replace")
    for path in party_logs if path.is_file()
)
oom = bool(re.search(
    r"(?:out of memory|cuda error.*memory|cudaErrorMemoryAllocation|std::bad_alloc|\boom\b)",
    log_text,
    flags=re.IGNORECASE,
))
native_passed = bool(native.get("passed") is True and native.get("status") == "passed")
status = "COMPLETE" if native_passed else ("OOM" if oom else "FAILED")
failure_reasons.extend(str(value) for value in native.get("failure_reasons", []) if value)

manifest = {
    "schema": "hpmpc_fid203_benchmark_run_v1",
    "status": status,
    "fid": fid,
    "layers": 12,
    "parties": 3,
    "scope": "full-bert-e2e",
    "batch": batch,
    "method": mode,
    "task": task,
    "link_gbps": link_gbps,
    "party_logs": [str(path.relative_to(run_dir)) for path in party_logs],
    "failure_reasons": failure_reasons,
    "native_manifest": str(native_path.relative_to(run_dir)),
    "native_manifest_sha256": sha256(native_path) if native_path.is_file() else None,
}

if status == "COMPLETE":
    execution = native.get("execution", {})
    per_party = execution.get("per_party", {})
    latency_ms = execution.get("controller_wall_ms")
    sends = [per_party.get(f"p{party}", {}).get("send_mb") for party in range(3)]
    gpu_values = [
        per_party.get(f"p{party}", {}).get("gpu_peak_memory_mb") for party in range(3)
    ]
    if (
        not isinstance(latency_ms, (int, float)) or not math.isfinite(float(latency_ms))
        or float(latency_ms) <= 0 or any(not isinstance(value, (int, float)) for value in sends)
        or any(not isinstance(value, (int, float)) for value in gpu_values)
    ):
        status = "FAILED"
        manifest["status"] = status
        manifest["failure_reasons"].append("native manifest lacks complete finite benchmark metrics")
    else:
        latency_ms = float(latency_ms)
        sends = [float(value) for value in sends]
        gpu_values = [float(value) for value in gpu_values]
        # Protocol logs report bytes sent to both peers.  The utilization value
        # is therefore effective peak per-host transmit utilization, not a
        # packet-counter claim; the basis is recorded explicitly.
        nic_util = max(sends) * 1_000_000 * 8 / (
            (latency_ms / 1000.0) * link_gbps * 1_000_000_000
        ) * 100.0
        manifest["metrics"] = {
            "latency_ms": latency_ms,
            "throughput_sps": batch / (latency_ms / 1000.0),
            "comm_gb": sum(sends) / 1000.0,
            "nic_util_pct": nic_util,
            "gpu_mem_mb": max(gpu_values),
        }
        manifest["metrics_basis"] = {
            "latency_ms": "controller wall time around concurrent three-party launch",
            "comm_gb": "sum of three parties' protocol-log send_mb / 1000",
            "nic_util_pct": "max per-party protocol-log send_mb / wall time / configured link_gbps",
            "gpu_mem_mb": "maximum nvidia-smi memory.used sample across all parties",
        }

temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps({"benchmark_manifest": str(output), "status": manifest["status"]}, indent=2))
raise SystemExit(0 if manifest["status"] == "COMPLETE" else 1)
PY
}

if precheck_cluster; then
    PRECHECK_STATUS="passed"
else
    PRECHECK_STATUS="failed"
fi

if [[ "$PRECHECK_STATUS" == "passed" ]]; then
    if build_or_validate; then
        [[ "$NO_BUILD" -eq 1 ]] && BUILD_STATUS="reused" || BUILD_STATUS="built"
    else
        BUILD_STATUS="failed"
    fi
fi

if [[ "$PRECHECK_STATUS" == "passed" && "$BUILD_STATUS" != "failed" ]]; then
    if stage_and_sync; then SYNC_STATUS="passed"; else SYNC_STATUS="failed"; fi
fi

if [[ "$SYNC_STATUS" == "passed" ]]; then
    if check_input_hashes; then INPUT_HASH_STATUS="passed"; else INPUT_HASH_STATUS="failed"; fi
    if check_binary_hashes; then BINARY_HASH_STATUS="passed"; else BINARY_HASH_STATUS="failed"; fi
fi

if [[ "$INPUT_HASH_STATUS" == "passed" && "$BINARY_HASH_STATUS" == "passed" ]]; then
    launch_triplet
    collect_run_artifacts
    if [[ "${PARTY_EXIT[0]}" -eq 0 && "${PARTY_EXIT[1]}" -eq 0 && "${PARTY_EXIT[2]}" -eq 0 ]]; then
        if check_output_hashes; then OUTPUT_HASH_STATUS="passed"; else OUTPUT_HASH_STATUS="failed"; fi
        if run_verifier; then
            VERIFY_STATUS="passed"
        else
            VERIFY_STATUS="failed"
            fail "verify_bert_base.py did not satisfy the numerical/prediction gate"
        fi
    else
        OUTPUT_HASH_STATUS="failed"
        VERIFY_STATUS="not_run"
        fail "one or more Trio parties exited nonzero: ${PARTY_EXIT[*]}"
    fi
else
    fail "execution skipped because synchronization/hash gates did not pass"
fi

native_status=0
benchmark_status=0
write_manifest || native_status=$?
write_benchmark_manifest || benchmark_status=$?
if [[ "$native_status" -eq 0 && "$benchmark_status" -eq 0 ]]; then
    exit 0
fi
exit 1
