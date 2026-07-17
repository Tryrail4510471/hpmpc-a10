#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT}/experiments/run_bert_e2e_3pc.sh"
RESULT_DIR="${ROOT}/results/perf"
BATCHES="1,4,8,16,32,64"
METHODS="serial,pipeline"
REPEATS=5
SCOPE="full-bert-e2e"
NIC="${HPMPC_NIC:-eth0}"
LINK_GBPS="${HPMPC_LINK_GBPS:-15}"
P0_HOST="${HPMPC_P0_HOST:-localhost}"
P1_HOST="${HPMPC_P1_HOST:-user@118.178.242.191}"
P2_HOST="${HPMPC_P2_HOST:-user@101.37.149.54}"

usage() {
  cat <<'EOF'
Usage: experiments/perf_benchmark.sh [options]

Benchmarks a complete three-party BERT inference runner. The runner must accept:
  --batch N --mode serial|pipeline --result-dir DIR
and must write the three party logs below DIR.

Options:
  --runner PATH       complete BERT 3PC runner
  --result-dir DIR    output directory (default: results/perf)
  --batches CSV       default: 1,4,8,16,32,64
  --methods CSV       default: serial,pipeline
  --repeats N         default: 5; minimum: 3
  --scope NAME        default: full-bert-e2e
  --nic DEVICE        default: eth0
  --link-gbps N       per-host link capacity, default: 15

Environment:
  HPMPC_P1_HOST/HPMPC_P2_HOST override SSH destinations.

This script intentionally refuses to benchmark FID=202 alone as full BERT.
EOF
}

while (($#)); do
  case "$1" in
    --runner) RUNNER="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --batches) BATCHES="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --nic) NIC="$2"; shift 2 ;;
    --link-gbps) LINK_GBPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$RUNNER" ]]; then
  cat >&2 <<EOF
BLOCKED: complete BERT 3PC runner is missing or not executable:
  $RUNNER

FID=202 implements only the layer-0 FFN block and cannot produce GLUE accuracy or
full-model performance. Add the tokenizer/embeddings, 12 attention+FFN layers,
pooler and task classifier, then provide a runner matching --help above.
EOF
  exit 2
fi
if ((REPEATS < 3)); then
  echo "--repeats must be >= 3 for paper measurements" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR/runs"
RAW="$RESULT_DIR/perf_raw.csv"
SUMMARY="$RESULT_DIR/perf_summary.csv"
printf '%s\n' 'scope,method,batch,repeat,wall_ms,latency_ms_per_sample,throughput_samples_per_sec,communication_gb,network_percent,gpu_peak_mb,status' > "$RAW"

remote_shell() {
  local host="$1" command="$2"
  if [[ "$host" == "localhost" || "$host" == "$P0_HOST" ]]; then
    bash -lc "$command"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$command"
  fi
}

nic_bytes() {
  local host="$1"
  remote_shell "$host" "awk '{sum += \$1} END {print sum}' /sys/class/net/${NIC}/statistics/rx_bytes /sys/class/net/${NIC}/statistics/tx_bytes"
}

start_gpu_monitor() {
  local host="$1" out="$2"
  if [[ "$host" == "localhost" || "$host" == "$P0_HOST" ]]; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$out" 2>&1 &
    echo "$!"
  else
    remote_shell "$host" "nohup nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > '$out' 2>&1 </dev/null & echo \$!"
  fi
}

stop_gpu_monitor() {
  local host="$1" pid="$2"
  if [[ -n "$pid" ]]; then
    if [[ "$host" == "localhost" || "$host" == "$P0_HOST" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    else
      remote_shell "$host" "kill '$pid' 2>/dev/null || true" || true
    fi
  fi
}

readarray -td, batch_values < <(printf '%s,' "$BATCHES")
readarray -td, method_values < <(printf '%s,' "$METHODS")

for method in "${method_values[@]}"; do
  [[ "$method" == "serial" || "$method" == "pipeline" ]] || {
    echo "Unsupported method: $method" >&2; exit 2;
  }
  for batch in "${batch_values[@]}"; do
    [[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid batch: $batch" >&2; exit 2; }
    for repeat in $(seq 1 "$REPEATS"); do
      run_id="${method}_b${batch}_r${repeat}"
      run_dir="$RESULT_DIR/runs/$run_id"
      mkdir -p "$run_dir"
      echo "[$run_id] starting complete BERT 3PC inference"

      before0="$(nic_bytes "$P0_HOST")"
      before1="$(nic_bytes "$P1_HOST")"
      before2="$(nic_bytes "$P2_HOST")"

      gpu0="$run_dir/gpu_p0.csv"
      gpu1="/tmp/hpmpc_${run_id}_gpu_p1.csv"
      gpu2="/tmp/hpmpc_${run_id}_gpu_p2.csv"
      pid0="$(start_gpu_monitor "$P0_HOST" "$gpu0")"
      pid1="$(start_gpu_monitor "$P1_HOST" "$gpu1")"
      pid2="$(start_gpu_monitor "$P2_HOST" "$gpu2")"

      start_ns="$(date +%s%N)"
      status=PASS
      "$RUNNER" --batch "$batch" --mode "$method" --result-dir "$run_dir" \
        >"$run_dir/runner.log" 2>&1 || status=FAIL
      end_ns="$(date +%s%N)"

      stop_gpu_monitor "$P0_HOST" "$pid0"
      stop_gpu_monitor "$P1_HOST" "$pid1"
      stop_gpu_monitor "$P2_HOST" "$pid2"
      scp -q "$P1_HOST:$gpu1" "$run_dir/gpu_p1.csv" || true
      scp -q "$P2_HOST:$gpu2" "$run_dir/gpu_p2.csv" || true
      remote_shell "$P1_HOST" "rm -f '$gpu1'" || true
      remote_shell "$P2_HOST" "rm -f '$gpu2'" || true

      after0="$(nic_bytes "$P0_HOST")"
      after1="$(nic_bytes "$P1_HOST")"
      after2="$(nic_bytes "$P2_HOST")"
      wall_ms="$(( (end_ns - start_ns) / 1000000 ))"

      metrics="$(python3 - "$run_dir" "$batch" "$wall_ms" "$LINK_GBPS" \
        "$before0" "$after0" "$before1" "$after1" "$before2" "$after2" <<'PY'
import pathlib, re, statistics, sys

run_dir = pathlib.Path(sys.argv[1])
batch, wall_ms = int(sys.argv[2]), float(sys.argv[3])
link_gbps = float(sys.argv[4])
pairs = [(float(sys.argv[i]), float(sys.argv[i + 1])) for i in (5, 7, 9)]

latency = wall_ms / batch
throughput = batch / (wall_ms / 1000.0) if wall_ms else 0.0
deltas = [max(0.0, end - begin) for begin, end in pairs]
# Maximum per-host NIC load is the conservative bottleneck utilization.
network = max(deltas) * 8.0 / ((wall_ms / 1000.0) * link_gbps * 1e9) * 100.0 if wall_ms else 0.0

sent_mb = 0.0
pattern = re.compile(r"Sending to other players:\s*([0-9.]+)\s*MB,\s*([0-9.]+)\s*MB")
for log in run_dir.glob("*.log"):
    for a, b in pattern.findall(log.read_text(errors="ignore")):
        sent_mb += float(a) + float(b)
communication_gb = sent_mb / 1000.0 / batch

gpu_values = []
for path in run_dir.glob("gpu_p*.csv"):
    for line in path.read_text(errors="ignore").splitlines():
        try:
            gpu_values.append(float(line.strip().split()[0]))
        except (ValueError, IndexError):
            pass
gpu_peak = max(gpu_values, default=float("nan"))
print(f"{latency:.6f},{throughput:.6f},{communication_gb:.6f},{network:.6f},{gpu_peak:.3f}")
PY
)"
      printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$SCOPE" "$method" "$batch" "$repeat" "$wall_ms" "$metrics" "$status" >> "$RAW"
      echo "[$run_id] $status wall=${wall_ms}ms metrics=$metrics"
      [[ "$status" == PASS ]] || exit 1
    done
  done
done

python3 - "$RAW" "$SUMMARY" <<'PY'
import csv, math, statistics, sys
from collections import defaultdict

raw, output = sys.argv[1:]
groups = defaultdict(list)
with open(raw, newline="") as handle:
    for row in csv.DictReader(handle):
        groups[(row["scope"], row["method"], int(row["batch"]))].append(row)

fields = ["latency_ms_per_sample", "throughput_samples_per_sec", "communication_gb", "network_percent", "gpu_peak_mb"]
header = ["scope", "method", "batch", "runs"] + [f"{name}_{suffix}" for name in fields for suffix in ("mean", "std")]
with open(output, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    for key in sorted(groups, key=lambda item: (item[1], item[2])):
        rows = groups[key]
        if len(rows) < 3 or any(row["status"] != "PASS" for row in rows):
            raise SystemExit(f"incomplete group: {key}")
        out = {"scope": key[0], "method": key[1], "batch": key[2], "runs": len(rows)}
        for name in fields:
            values = [float(row[name]) for row in rows]
            out[f"{name}_mean"] = f"{statistics.mean(values):.6f}"
            out[f"{name}_std"] = f"{statistics.stdev(values):.6f}"
        writer.writerow(out)
PY

echo "Raw results:     $RAW"
echo "Summary results: $SUMMARY"
