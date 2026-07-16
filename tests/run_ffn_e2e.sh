#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --party {0|1|2} --case {T1|T2|T3|T4} [--mode single|constellation|split] [--profile]"
}

PARTY=""
CASE=""
MODE="single"
PROFILE=0
P0_IP="${HPMPC_P0_IP:-172.24.178.175}"
P1_IP="${HPMPC_P1_IP:-172.24.178.176}"
P2_IP="${HPMPC_P2_IP:-172.24.178.177}"
INPUT="${HPMPC_BERT_FFN_INPUT:-/tmp/hpmpc_bert_ffn_input_q20.bin}"
OUTPUT="${HPMPC_BERT_FFN_OUTPUT:-/tmp/hpmpc_bert_ffn_output_q20.bin}"
WEIGHTS="${HPMPC_BERT_FFN_WEIGHTS:-nn/Pygeon/models/pretrained/bert_base_uncased_layer0_ffn_q20.bin}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --party) PARTY="$2"; shift 2 ;;
        --case) CASE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --profile) PROFILE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$PARTY" =~ ^[012]$ ]] || { echo "--party must be 0, 1, or 2" >&2; exit 2; }
[[ "$CASE" =~ ^T[1-4]$ ]] || { echo "--case must be T1, T2, T3, or T4" >&2; exit 2; }
[[ "$MODE" == "single" || "$MODE" == "constellation" || "$MODE" == "split" ]] || {
    echo "--mode must be single, constellation, or split" >&2; exit 2;
}
[[ -r "$INPUT" ]] || { echo "missing input: $INPUT" >&2; exit 2; }
[[ -r "$WEIGHTS" ]] || { echo "missing weights: $WEIGHTS" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p results/ffn_logs

case "$CASE" in
    T1) ROWS=4; STAGE=core ;;
    T2) ROWS=256; STAGE=core ;;
    T3) ROWS=256; STAGE=full ;;
    T4) ROWS=128; STAGE=full ;;
esac

if [[ "$MODE" == "single" ]]; then
    SPLIT=0
    WORKERS=1
elif [[ "$MODE" == "constellation" ]]; then
    SPLIT=-1
    WORKERS=8
else
    SPLIT=1
    WORKERS=48
    if [[ "${HPMPC_ALLOW_OVERSUBSCRIBED_SPLIT:-0}" != "1" ]]; then
        echo "Refusing 48-worker split: private limb weights require about 54 GiB/node; set HPMPC_ALLOW_OVERSUBSCRIBED_SPLIT=1 to override." >&2
        exit 2
    fi
fi

export HPMPC_BERT_FFN_ROWS="$ROWS"
export HPMPC_BERT_FFN_TOTAL_WORKERS="$WORKERS"
export HPMPC_BERT_FFN_STAGE="$STAGE"
export HPMPC_BERT_FFN_INPUT="$INPUT"
export HPMPC_BERT_FFN_OUTPUT="$OUTPUT"
export HPMPC_BERT_FFN_WEIGHTS="$WEIGHTS"
rm -f "$OUTPUT" "$OUTPUT".part-*-of-*.bin

DMON_PID=""
cleanup() {
    if [[ -n "$DMON_PID" ]]; then
        kill "$DMON_PID" 2>/dev/null || true
        wait "$DMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$PROFILE" -eq 1 ]]; then
    stdbuf -oL nvidia-smi dmon -s u -d 1 -o DT > "results/ffn_logs/${CASE}_${MODE}_p${PARTY}_gpu_dmon.log" &
    DMON_PID=$!
fi

LOG="results/ffn_logs/${CASE}_${MODE}_p${PARTY}.log"
RX_START="$(< /sys/class/net/eth0/statistics/rx_bytes)"
TX_START="$(< /sys/class/net/eth0/statistics/tx_bytes)"
START_NS="$(date +%s%N)"
set +e
if [[ "$MODE" == "constellation" ]]; then
    case "$PARTY" in
        0) ./executables/run-P0--0-1-2.o "$P1_IP" "$P2_IP" >"$LOG" 2>&1 ;;
        1) ./executables/run-P1--0-1-2.o "$P0_IP" "$P2_IP" >"$LOG" 2>&1 ;;
        2) ./executables/run-P2--0-1-2.o "$P0_IP" "$P1_IP" >"$LOG" 2>&1 ;;
    esac
else
    scripts/run.sh -p "$PARTY" -s "$SPLIT" -a "$P0_IP" -b "$P1_IP" -c "$P2_IP" -g 1 >"$LOG" 2>&1
fi
STATUS=$?
set -e
END_NS="$(date +%s%N)"
RX_END="$(< /sys/class/net/eth0/statistics/rx_bytes)"
TX_END="$(< /sys/class/net/eth0/statistics/tx_bytes)"
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
cleanup
DMON_PID=""
printf 'FFN_E2E case=%s party=%s stage=%s rows=%s workers=%s elapsed_ms=%s status=%s\n' \
    "$CASE" "$PARTY" "$STAGE" "$ROWS" "$WORKERS" "$ELAPSED_MS" "$STATUS" | tee -a "$LOG"
printf 'FFN_NIC party=%s iface=eth0 rx_bytes=%s tx_bytes=%s elapsed_ms=%s\n' \
    "$PARTY" "$((RX_END - RX_START))" "$((TX_END - TX_START))" "$ELAPSED_MS" | tee -a "$LOG"
exit "$STATUS"
