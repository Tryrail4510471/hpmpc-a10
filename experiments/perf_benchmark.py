#!/usr/bin/env python3
"""Run and aggregate evidence-backed FID=203 performance experiments.

The external 3PC runner is responsible for launching all parties and collecting
their logs.  For every invocation it must write the requested metrics manifest::

  {
    "status": "COMPLETE", "fid": 203, "layers": 12, "parties": 3,
    "scope": "full-bert-e2e", "batch": 1, "method": "serial",
    "party_logs": ["p0.log", "p1.log", "p2.log"],
    "metrics": {
      "latency_ms": 0.0, "throughput_sps": 0.0, "comm_gb": 0.0,
      "nic_util_pct": 0.0, "gpu_mem_mb": 0.0
    }
  }

Metrics without this provenance are rejected.  OOM is retained as a measured
capacity result with its real log; it is never converted into a timing value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


BATCHES = (1, 4, 8, 16, 32, 64)
METHODS = ("serial", "pipeline")
METRICS = ("latency_ms", "throughput_sps", "comm_gb", "nic_util_pct", "gpu_mem_mb")
COMPLETE_STATUSES = {"PASS", "PASSED", "COMPLETE"}
OOM_PATTERN = re.compile(
    r"(?:out of memory|cuda error.*memory|cudaErrorMemoryAllocation|std::bad_alloc|\boom\b)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_artifact(manifest_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def as_int(payload: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name in payload:
            try:
                return int(payload[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"manifest field {name!r} is not an integer") from error
    return None


def validate_success_manifest(
    manifest_path: Path,
    expected_batch: int,
    expected_method: str,
    expected_fid: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("runner did not create a non-empty metrics manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics manifest root must be an object")
    if str(payload.get("status", "")).upper() not in COMPLETE_STATUSES:
        raise ValueError(f"manifest status is {payload.get('status')!r}, not COMPLETE/PASS")
    if as_int(payload, "fid", "function_identifier") != expected_fid:
        raise ValueError("metrics manifest does not prove FID=203")
    if as_int(payload, "layers", "encoder_layers", "num_hidden_layers") != 12:
        raise ValueError("metrics manifest does not prove all 12 encoder layers")
    if as_int(payload, "parties", "party_count") != 3:
        raise ValueError("metrics manifest does not prove a real three-party run")
    if as_int(payload, "batch", "batch_size") != expected_batch:
        raise ValueError("metrics manifest batch does not match invocation")
    if str(payload.get("method", "")) != expected_method:
        raise ValueError("metrics manifest method does not match invocation")
    if str(payload.get("scope", "")) != "full-bert-e2e":
        raise ValueError("metrics manifest scope must be full-bert-e2e")

    raw_logs = payload.get("party_logs")
    if isinstance(raw_logs, dict):
        raw_logs = list(raw_logs.values())
    if not isinstance(raw_logs, list) or len(raw_logs) < 3:
        raise ValueError("metrics manifest must list at least three party logs")
    log_paths = [resolve_artifact(manifest_path, value) for value in raw_logs]
    if len({path.resolve() for path in log_paths}) < 3:
        raise ValueError("party_logs must identify three distinct files")
    for path in log_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"party log is missing or empty: {path}")
        try:
            path.resolve().relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"party log must be collected inside this run directory: {path}") from error

    source = payload.get("metrics")
    if not isinstance(source, dict):
        source = payload
    metrics: dict[str, float] = {}
    for name in METRICS:
        try:
            value = float(source[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing or invalid metric {name}") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metric {name} must be finite and non-negative")
        metrics[name] = value
    if metrics["latency_ms"] <= 0 or metrics["throughput_sps"] <= 0:
        raise ValueError("latency and throughput must be positive")
    if metrics["nic_util_pct"] > 100.0:
        raise ValueError("nic_util_pct must be in [0,100]")

    evidence = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "party_logs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in log_paths
        ],
    }
    return metrics, evidence


def validate_oom_manifest(
    manifest_path: Path,
    expected_batch: int,
    expected_method: str,
    expected_fid: int,
) -> dict[str, Any]:
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("OOM is unproven because the runner wrote no manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("status", "")).upper() != "OOM":
        raise ValueError("OOM manifest must have status=OOM")
    if as_int(payload, "fid", "function_identifier") != expected_fid:
        raise ValueError("OOM manifest does not prove FID=203")
    if as_int(payload, "layers", "encoder_layers", "num_hidden_layers") != 12:
        raise ValueError("OOM manifest does not prove all 12 layers")
    if as_int(payload, "parties", "party_count") != 3:
        raise ValueError("OOM manifest does not prove a three-party attempt")
    if as_int(payload, "batch", "batch_size") != expected_batch:
        raise ValueError("OOM manifest batch does not match invocation")
    if str(payload.get("method", "")) != expected_method:
        raise ValueError("OOM manifest method does not match invocation")
    if str(payload.get("scope", "")) != "full-bert-e2e":
        raise ValueError("OOM manifest scope must be full-bert-e2e")
    raw_logs = payload.get("party_logs")
    if isinstance(raw_logs, dict):
        raw_logs = list(raw_logs.values())
    if not isinstance(raw_logs, list) or len(raw_logs) < 3:
        raise ValueError("OOM manifest must list three party logs")
    paths = [resolve_artifact(manifest_path, value) for value in raw_logs]
    if len({path.resolve() for path in paths}) < 3:
        raise ValueError("OOM party logs must be distinct")
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"OOM party log is missing or empty: {path}")
        try:
            path.resolve().relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"OOM party log must be inside this run directory: {path}") from error
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "party_logs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def runner_command(runner: Path) -> list[str]:
    if runner.suffix == ".py":
        return [sys.executable, str(runner)]
    return [str(runner)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=203)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--task", choices=("mrpc", "rte"), default="mrpc")
    parser.add_argument(
        "--experiment-kind", choices=("performance", "bandwidth"), default="performance"
    )
    parser.add_argument("--runner", type=Path, default=Path("experiments/run_bert_3pc.sh"))
    parser.add_argument("--output", type=Path, default=Path("results/perf_results.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--link-gbps", type=float, default=15.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        print(
            f"BLOCKED: refusing to overwrite existing result {args.output}; pass --overwrite",
            file=sys.stderr,
        )
        return 2
    requested_batches = list(dict.fromkeys(args.batch_sizes))
    requested_methods = list(dict.fromkeys(args.methods))
    result: dict[str, Any] = {
        "schema": "hpmpc_fid203_performance_v1",
        "status": "PENDING",
        "fid": args.fid,
        "layers": 12,
        "parties": 3,
        "scope": "full-bert-e2e",
        "task": args.task,
        "experiment_kind": args.experiment_kind,
        "required_runs_per_group": args.runs,
        "requested_batches": requested_batches,
        "requested_methods": requested_methods,
        "link_gbps": args.link_gbps,
        "groups": [],
    }
    if args.fid != 203:
        result.update(status="BLOCKED_WRONG_FID", reason="only FID=203 is full 12-layer BERT")
        write_json(args.output, result)
        return 2
    if args.experiment_kind == "performance":
        if args.runs != 5:
            result.update(status="BLOCKED_INVALID_RUN_COUNT", reason="paper performance requires exactly 5 runs")
            write_json(args.output, result)
            return 2
        if set(requested_batches) != set(BATCHES):
            result.update(
                status="BLOCKED_INCOMPLETE_BATCH_MATRIX",
                reason="paper performance requires batch sizes 1,4,8,16,32,64",
            )
            write_json(args.output, result)
            return 2
        if set(requested_methods) != set(METHODS):
            result.update(
                status="BLOCKED_INCOMPLETE_METHOD_MATRIX",
                reason="paper performance requires both serial and pipeline methods",
            )
            write_json(args.output, result)
            return 2
    else:
        if args.runs != 3 or len(requested_batches) != 1 or requested_methods != ["serial"]:
            result.update(
                status="BLOCKED_INVALID_BANDWIDTH_MATRIX",
                reason="each bandwidth point requires one batch, serial method, and exactly 3 runs",
            )
            write_json(args.output, result)
            return 2
    if not args.runner.is_file():
        result.update(
            status="BLOCKED_MISSING_FID203_RUNNER",
            reason=f"runner does not exist: {args.runner}",
        )
        write_json(args.output, result)
        return 2
    if args.runner.suffix != ".py" and not os.access(args.runner, os.X_OK):
        result.update(
            status="BLOCKED_NONEXECUTABLE_FID203_RUNNER",
            reason=f"runner is not executable: {args.runner}",
        )
        write_json(args.output, result)
        return 2
    if args.timeout_seconds < 0 or args.link_gbps <= 0:
        raise SystemExit("--timeout-seconds must be >=0 and --link-gbps must be positive")

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = args.run_root or args.output.parent / f"{args.output.stem}_runs" / session
    run_root.mkdir(parents=True, exist_ok=False)
    result["run_root"] = str(run_root.resolve())
    write_json(args.output, result)
    blocked = False
    observed_oom = False

    for method in requested_methods:
        for batch in requested_batches:
            group: dict[str, Any] = {
                "method": method,
                "batch": batch,
                "required_runs": args.runs,
                "status": "PENDING",
                "runs": [],
            }
            result["groups"].append(group)
            for repeat in range(1, args.runs + 1):
                run_id = f"{method}_b{batch}_r{repeat}"
                run_dir = run_root / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                manifest = run_dir / "metrics.json"
                stdout_path = run_dir / "runner.stdout.log"
                stderr_path = run_dir / "runner.stderr.log"
                command = runner_command(args.runner) + [
                    "--fid", str(args.fid),
                    "--batch", str(batch),
                    "--mode", method,
                    "--task", args.task,
                    "--link-gbps", str(args.link_gbps),
                    "--run-id", run_id,
                    "--result-dir", str(run_dir),
                    "--metrics-json", str(manifest),
                ]
                run: dict[str, Any] = {
                    "run_id": run_id,
                    "repeat": repeat,
                    "command": command,
                    "status": "RUNNING",
                    "stdout_log": str(stdout_path.resolve()),
                    "stderr_log": str(stderr_path.resolve()),
                }
                group["runs"].append(run)
                write_json(args.output, result)
                try:
                    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                        completed = subprocess.run(
                            command,
                            stdout=stdout,
                            stderr=stderr,
                            timeout=args.timeout_seconds or None,
                            check=False,
                        )
                    run["returncode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    run.update(status="BLOCKED_TIMEOUT", reason="runner exceeded timeout")
                    group["status"] = "BLOCKED"
                    blocked = True
                    break
                except OSError as error:
                    run.update(status="BLOCKED_RUNNER_EXEC_FAILED", reason=str(error))
                    group["status"] = "BLOCKED"
                    blocked = True
                    break

                combined_log = ""
                for path in (stdout_path, stderr_path):
                    combined_log += path.read_text(encoding="utf-8", errors="replace")
                manifest_payload: dict[str, Any] = {}
                if manifest.is_file() and manifest.stat().st_size:
                    try:
                        decoded = json.loads(manifest.read_text(encoding="utf-8"))
                        if isinstance(decoded, dict):
                            manifest_payload = decoded
                    except json.JSONDecodeError:
                        pass
                manifest_status = str(manifest_payload.get("status", "")).upper()
                is_oom = manifest_status == "OOM" or bool(OOM_PATTERN.search(combined_log))
                if is_oom:
                    try:
                        oom_evidence = validate_oom_manifest(manifest, batch, method, args.fid)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        run.update(status="BLOCKED_INVALID_OOM_EVIDENCE", reason=str(error))
                        group["status"] = "BLOCKED"
                        blocked = True
                        break
                    run.update(
                        status="OOM",
                        reason="runner reported an evidence-backed out-of-memory condition",
                        evidence=oom_evidence,
                        stdout_sha256=sha256_file(stdout_path),
                        stderr_sha256=sha256_file(stderr_path),
                    )
                    group["status"] = "OOM"
                    observed_oom = True
                    break
                if completed.returncode != 0:
                    run.update(status="BLOCKED_RUNNER_FAILED", reason=f"runner exit code {completed.returncode}")
                    group["status"] = "BLOCKED"
                    blocked = True
                    break
                try:
                    metrics, evidence = validate_success_manifest(
                        manifest, batch, method, args.fid
                    )
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    run.update(status="BLOCKED_INVALID_METRICS", reason=str(error))
                    group["status"] = "BLOCKED"
                    blocked = True
                    break
                run.update(
                    status="COMPLETE",
                    metrics=metrics,
                    evidence=evidence,
                    stdout_sha256=sha256_file(stdout_path),
                    stderr_sha256=sha256_file(stderr_path),
                )
                write_json(args.output, result)

            complete_runs = [run for run in group["runs"] if run["status"] == "COMPLETE"]
            if len(complete_runs) == args.runs:
                group["status"] = "COMPLETE"
                group["summary"] = {
                    name: summarize([run["metrics"][name] for run in complete_runs])
                    for name in METRICS
                }
                group["summary"]["latency_ms_per_sample"] = summarize(
                    [run["metrics"]["latency_ms"] / batch for run in complete_runs]
                )
                group["summary"]["comm_gb_per_sample"] = summarize(
                    [run["metrics"]["comm_gb"] / batch for run in complete_runs]
                )
            elif group["status"] == "PENDING":
                group["status"] = "BLOCKED_INCOMPLETE_RUNS"
                blocked = True
            write_json(args.output, result)
            if blocked and args.stop_on_error:
                result.update(status="BLOCKED", reason="at least one group lacks valid real-run evidence")
                write_json(args.output, result)
                return 2

    if blocked:
        result.update(status="BLOCKED", reason="at least one group lacks valid real-run evidence")
        exit_code = 2
    elif observed_oom:
        result.update(status="COMPLETE_WITH_OOM", reason="OOM groups are recorded without fabricated metrics")
        exit_code = 0
    else:
        result.update(status="COMPLETE")
        exit_code = 0
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
