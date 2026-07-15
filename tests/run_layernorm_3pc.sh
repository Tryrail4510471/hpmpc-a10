#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --party {0|1|2} [--mode single|process|split] [--input PATH] [--output PATH]"
}

PARTY=""
MODE="split"
INPUT="/tmp/hpmpc_layernorm_input.bin"
OUTPUT="/tmp/hpmpc_layernorm_output.bin"
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
if [[ "$MODE" != "single" && "$MODE" != "process" && "$MODE" != "split" ]]; then
    echo "--mode must be single, process, or split" >&2
    exit 2
fi
if [[ ! -r "$INPUT" ]]; then
    echo "Input file is not readable: $INPUT" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export HPMPC_LAYERNORM_INPUT="$INPUT"
export HPMPC_LAYERNORM_OUTPUT="$OUTPUT"

case "$MODE" in
    single)
        export HPMPC_LAYERNORM_TOTAL_WORKERS=1
        exec scripts/run.sh -p "$PARTY" -s 0 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP"
        ;;
    process)
        export HPMPC_LAYERNORM_TOTAL_WORKERS="${HPMPC_LAYERNORM_TOTAL_WORKERS:-8}"
        exec scripts/run.sh -p "$PARTY" -s 0 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP"
        ;;
    split)
        # Six role constellations times PROCESS_NUM=8 produce offsets 0..47.
        # The LayerNorm harness shards logical token rows before SIMD packing,
        # so [4,128,768] supplies work to all 48 workers.
        export HPMPC_LAYERNORM_TOTAL_WORKERS="${HPMPC_LAYERNORM_TOTAL_WORKERS:-48}"
        exec scripts/run.sh -p "$PARTY" -s 1 -a "$P0_IP" -b "$P1_IP" -c "$P2_IP" -g 1
        ;;
esac
