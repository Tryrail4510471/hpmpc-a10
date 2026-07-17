#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_PY="${ROOT}/experiments/perf_benchmark.py"
RUNNER="${ROOT}/experiments/run_bert_3pc.sh"
OUTPUT="${ROOT}/results/bandwidth_results.json"
RESULT_DIR=""
FID=203
BATCH=1
RUNS=3
TASK=mrpc
IFACE="${HPMPC_NIC:-eth0}"
BANDWIDTHS=(1gbit 10gbit none)
P0_HOST="${HPMPC_P0_HOST:-localhost}"
P1_HOST="${HPMPC_P1_HOST:-user@118.178.242.191}"
P2_HOST="${HPMPC_P2_HOST:-user@101.37.149.54}"
HOSTS=("$P0_HOST" "$P1_HOST" "$P2_HOST")
MODIFIED=0
OVERWRITE_OUTPUT=0

usage() {
  cat <<'EOF'
Usage: HPMPC_ALLOW_TC=1 experiments/bandwidth_sensitivity.sh [options]

  --bandwidths 1gbit 10gbit none   required three paper points
  --iface DEVICE                   default: HPMPC_NIC or eth0
  --fid 203                        full twelve-layer BERT only
  --batch N                        default: 1
  --runs 3                         exactly three runs per point
  --task mrpc|rte                  default: mrpc
  --runner PATH                    real three-party FID=203 runner
  --result-dir DIR                 raw per-point data
  --output FILE                    final evidence-backed JSON
  --overwrite-output               allow replacing an existing output JSON

The script shapes all three hosts. HPMPC_ALLOW_TC=1 and passwordless sudo are
mandatory. An EXIT/INT/TERM/HUP trap removes the TBF qdisc and records whether
restoration succeeded. Run from an out-of-band management session.
EOF
}

while (($#)); do
  case "$1" in
    --bandwidths)
      shift
      BANDWIDTHS=()
      while (($#)) && [[ "$1" != --* ]]; do BANDWIDTHS+=("$1"); shift; done
      ;;
    --iface) IFACE="$2"; shift 2 ;;
    --fid) FID="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --overwrite-output) OVERWRITE_OUTPUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$FID" == 203 ]] || { echo "BLOCKED: only FID=203 is full BERT" >&2; exit 2; }
[[ "$RUNS" == 3 ]] || { echo "BLOCKED: bandwidth paper points require exactly 3 runs" >&2; exit 2; }
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "--batch must be positive" >&2; exit 2; }
[[ "$TASK" == mrpc || "$TASK" == rte ]] || { echo "--task must be mrpc or rte" >&2; exit 2; }
[[ "$IFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "unsafe interface name: $IFACE" >&2; exit 2; }
[[ "${HPMPC_ALLOW_TC:-0}" == 1 ]] || {
  echo "Refusing to modify qdisc without HPMPC_ALLOW_TC=1" >&2
  exit 2
}
[[ -f "$PERF_PY" ]] || { echo "BLOCKED: missing $PERF_PY" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "BLOCKED: missing real FID=203 runner $RUNNER" >&2; exit 2; }
if [[ "$RUNNER" != *.py && ! -x "$RUNNER" ]]; then
  echo "BLOCKED: FID=203 runner is not executable: $RUNNER" >&2
  exit 2
fi
if [[ -e "$OUTPUT" && "$OVERWRITE_OUTPUT" != 1 ]]; then
  echo "BLOCKED: refusing to overwrite $OUTPUT; pass --overwrite-output" >&2
  exit 2
fi
if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="${ROOT}/results/bandwidth_runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

python3 - "${BANDWIDTHS[@]}" <<'PY'
import sys
values = sys.argv[1:]
if len(values) != 3 or set(values) != {"1gbit", "10gbit", "none"}:
    raise SystemExit("--bandwidths must contain exactly: 1gbit 10gbit none")
PY

mkdir -p "$(dirname "$OUTPUT")" "$RESULT_DIR"
python3 - "$OUTPUT" "$FID" "$BATCH" "$RUNS" "$TASK" "$IFACE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "hpmpc_fid203_bandwidth_v1",
    "status": "PENDING",
    "fid": int(sys.argv[2]),
    "layers": 12,
    "parties": 3,
    "batch": int(sys.argv[3]),
    "required_runs_per_point": int(sys.argv[4]),
    "task": sys.argv[5],
    "interface": sys.argv[6],
    "points": [],
    "network_restored": False,
    "qdisc_before": {},
    "qdisc_after": {},
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

remote_shell() {
  local host="$1" command="$2"
  if [[ "$host" == localhost || "$host" == "$P0_HOST" ]]; then
    bash -lc "$command"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$command"
  fi
}

qdisc_show() {
  remote_shell "$1" "sudo -n tc qdisc show dev '$IFACE'"
}

record_qdisc() {
  local phase="$1" host="$2" snapshot="$3"
  python3 - "$OUTPUT" "$phase" "$host" "$snapshot" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload[sys.argv[2]][sys.argv[3]] = sys.argv[4]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

restore_links() {
  local host snapshot failed=0
  for host in "${HOSTS[@]}"; do
    remote_shell "$host" "sudo -n tc qdisc del dev '$IFACE' root 2>/dev/null || true" || failed=1
  done
  for host in "${HOSTS[@]}"; do
    snapshot="$(qdisc_show "$host")" || { failed=1; continue; }
    record_qdisc qdisc_after "$host" "$snapshot" || failed=1
    if grep -Eq '^qdisc tbf .* root' <<<"$snapshot"; then
      echo "ERROR: TBF remains on $host:$IFACE after cleanup" >&2
      failed=1
    fi
  done
  return "$failed"
}

finalize_result() {
  local exit_code="$1" restored="$2"
  python3 - "$OUTPUT" "$exit_code" "$restored" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
exit_code = int(sys.argv[2])
restored = sys.argv[3] == "1"
payload["network_restored"] = restored
if exit_code == 0 and restored and payload.get("status") == "COMPLETE_PENDING_NETWORK_RESTORE":
    payload["status"] = "COMPLETE"
elif exit_code != 0 or not restored:
    payload["status"] = "BLOCKED"
    payload["reason"] = (
        "experiment command failed" if exit_code != 0 else "qdisc restoration could not be verified"
    )
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

on_exit() {
  local rc=$? restored=1
  trap - EXIT INT TERM HUP
  if ((MODIFIED)); then
    restore_links || restored=0
  fi
  finalize_result "$rc" "$restored" || true
  if ((rc == 0)) && [[ "$restored" != 1 ]]; then rc=3; fi
  exit "$rc"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap on_exit EXIT

# Refuse to replace an administrator-defined root qdisc. mq/noqueue are the
# expected kernel defaults on the A10 hosts; deleting TBF restores that default.
for host in "${HOSTS[@]}"; do
  snapshot="$(qdisc_show "$host")"
  record_qdisc qdisc_before "$host" "$snapshot"
  root_kind="$(awk '$1=="qdisc" && $4=="root" {print $2; exit}' <<<"$snapshot")"
  [[ "$root_kind" == mq || "$root_kind" == noqueue ]] || {
    echo "BLOCKED: $host:$IFACE has non-default root qdisc '$root_kind'; refusing to replace" >&2
    exit 2
  }
  remote_shell "$host" "sudo -n tc qdisc show dev '$IFACE' >/dev/null"
done

apply_rate() {
  local label="$1" host
  MODIFIED=1
  for host in "${HOSTS[@]}"; do
    case "$label" in
      none)
        remote_shell "$host" "sudo -n tc qdisc del dev '$IFACE' root 2>/dev/null || true"
        ;;
      1gbit|10gbit)
        remote_shell "$host" \
          "sudo -n tc qdisc replace dev '$IFACE' root tbf rate '$label' burst 4mb latency 400ms"
        ;;
      *) echo "unsupported bandwidth: $label" >&2; return 2 ;;
    esac
  done
  for host in "${HOSTS[@]}"; do
    if [[ "$label" == none ]]; then
      ! qdisc_show "$host" | grep -Eq '^qdisc tbf .* root'
    else
      qdisc_show "$host" | grep -Eq '^qdisc tbf .* root'
    fi
  done
}

append_point() {
  local label="$1" gbps="$2" point_json="$3"
  python3 - "$OUTPUT" "$label" "$gbps" "$point_json" <<'PY'
import json, pathlib, sys
output, label, gbps, point_path = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), pathlib.Path(sys.argv[4])
point = json.loads(point_path.read_text())
if point.get("status") != "COMPLETE":
    raise SystemExit(f"bandwidth point {label} is not COMPLETE: {point.get('status')}")
groups = point.get("groups", [])
if len(groups) != 1 or groups[0].get("status") != "COMPLETE" or len(groups[0].get("runs", [])) != 3:
    raise SystemExit(f"bandwidth point {label} lacks three complete real runs")
summary = groups[0]["summary"]
payload = json.loads(output.read_text())
payload["points"].append({
    "label": label,
    "bandwidth_gbps": gbps,
    "status": "COMPLETE",
    "runs": 3,
    "latency_ms": summary["latency_ms"],
    "latency_ms_per_sample": summary["latency_ms_per_sample"],
    "nic_util_pct": summary["nic_util_pct"],
    "performance_result": str(point_path.resolve()),
})
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

for label in "${BANDWIDTHS[@]}"; do
  case "$label" in
    1gbit) gbps=1 ;;
    10gbit) gbps=10 ;;
    none) gbps=15 ;;
  esac
  echo "Applying $label on all three parties"
  apply_rate "$label"
  point_dir="${RESULT_DIR}/${label}"
  point_json="${point_dir}/perf.json"
  mkdir -p "$point_dir"
  python3 "$PERF_PY" \
    --experiment-kind bandwidth \
    --fid 203 \
    --batch-sizes "$BATCH" \
    --methods serial \
    --runs 3 \
    --task "$TASK" \
    --runner "$RUNNER" \
    --link-gbps "$gbps" \
    --run-root "${point_dir}/raw" \
    --output "$point_json" \
    --stop-on-error
  append_point "$label" "$gbps" "$point_json"
done

python3 - "$OUTPUT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
if {point["bandwidth_gbps"] for point in payload["points"]} != {1.0, 10.0, 15.0}:
    raise SystemExit("bandwidth result is missing a required point")
payload["status"] = "COMPLETE_PENDING_NETWORK_RESTORE"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "Bandwidth evidence: $OUTPUT"
