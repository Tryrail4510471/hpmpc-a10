#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --party {0|1|2} [--mode single|process|split2|split] [--input PATH] [--output PATH]"
}

PARTY=""
MODE="single"
INPUT="/tmp/hpmpc_softmax_input.bin"
OUTPUT="/tmp/hpmpc_softmax_output.bin"
P0_IP="${HPMPC_P0_IP:-172.24.178.175}"
P1_IP="${HPMPC_P1_IP:-172.24.178.176}"
P2_IP="${HPMPC_P2_IP:-172.24.178.177}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --party) PARTY="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$PARTY" =~ ^[012]$ ]]; then
    echo "--party must be 0, 1, or 2" >&2
    exit 2
fi
if [[ "$MODE" != "single" && "$MODE" != "process" && "$MODE" != "split2" && "$MODE" != "split" ]]; then
    echo "--mode must be single, process, split2, or split" >&2
    exit 2
fi
if [[ ! -r "$INPUT" ]]; then
    echo "Input file is not readable: $INPUT" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HPMPC_SOFTMAX_INPUT="$INPUT"
export HPMPC_SOFTMAX_OUTPUT="$OUTPUT"
if [[ "$MODE" == "single" ]]; then
    export HPMPC_SOFTMAX_TOTAL_WORKERS=1
    exec scripts/run.sh -p "$PARTY" -s 0 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP"
elif [[ "$MODE" == "process" ]]; then
    # PROCESS_NUM=8 without the six split-role constellations: one worker per
    # physical CPU core on the current 8-core hosts.
    export HPMPC_SOFTMAX_TOTAL_WORKERS="${HPMPC_SOFTMAX_TOTAL_WORKERS:-8}"
    exec scripts/run.sh -p "$PARTY" -s 0 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP"
elif [[ "$MODE" == "split2" ]]; then
    # Two complementary role constellations, offsets 0 and 1. Together they
    # create process_offset values 0..15 and reduce the single-role imbalance.
    export HPMPC_SOFTMAX_TOTAL_WORKERS="${HPMPC_SOFTMAX_TOTAL_WORKERS:-16}"
    case "$PARTY" in
        0)
            ./executables/run-P0--0-1-2.o "$P1_IP" "$P2_IP" &
            ./executables/run-P0--0-2-1.o "$P2_IP" "$P1_IP" &
            ;;
        1)
            ./executables/run-P1--0-1-2.o "$P0_IP" "$P2_IP" &
            ./executables/run-P1--0-2-1.o "$P0_IP" "$P2_IP" &
            ;;
        2)
            ./executables/run-P2--0-1-2.o "$P0_IP" "$P1_IP" &
            ./executables/run-P2--0-2-1.o "$P0_IP" "$P1_IP" &
            ;;
    esac
    failures=0
    for job in $(jobs -p); do
        wait "$job" || failures=$((failures + 1))
    done
    exit "$failures"
else
    # SPLITROLES=1 creates six role constellations; each executable forks
    # PROCESS_NUM=8 children, giving process_offset values 0..47.
    export HPMPC_SOFTMAX_TOTAL_WORKERS="${HPMPC_SOFTMAX_TOTAL_WORKERS:-48}"
    exec scripts/run.sh -p "$PARTY" -s 1 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP" -g 1
fi
