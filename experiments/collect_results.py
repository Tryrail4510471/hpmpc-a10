#!/usr/bin/env python3
"""Truth-gated collection of FID=203 paper results.

The paper table is written only after MRPC/RTE accuracy, the complete
batch/method performance matrix (including evidence-backed OOM outcomes), and
all three bandwidth points pass provenance checks.  On missing/BLOCKED data the
existing table is left untouched and a machine-readable gate report is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


REQUIRED_BATCHES = (1, 4, 8, 16, 32, 64)
REQUIRED_METHODS = ("serial", "pipeline")
REQUIRED_BANDWIDTHS = (1.0, 10.0, 15.0)
METRICS = ("latency_ms", "throughput_sps", "comm_gb", "nic_util_pct", "gpu_mem_mb")
PUMA = {
    "protocol": "3PC RSS",
    "mrpc": "—",
    "rte": "70.0",
    "latency_seconds": "33.913",
    "communication_gb": "10.773",
    "nic": "—",
}


class GateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise GateError(f"{name}: missing or empty result file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GateError(f"{name}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{name}: JSON root must be an object")
    return payload


def finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{name}: expected a number, got {value!r}") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise GateError(f"{name}: value must be finite{' and non-negative' if nonnegative else ''}")
    return result


def verify_hashed_file(path_value: Any, digest_value: Any, name: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateError(f"{name}: evidence file is missing or empty: {path}")
    digest = str(digest_value or "")
    if len(digest) != 64 or sha256_file(path) != digest:
        raise GateError(f"{name}: SHA-256 evidence mismatch for {path}")
    return path


def validate_party_evidence(evidence: Any, name: str) -> None:
    if not isinstance(evidence, dict):
        raise GateError(f"{name}: missing evidence object")
    manifest = verify_hashed_file(
        evidence.get("manifest"), evidence.get("manifest_sha256"), f"{name}.manifest"
    )
    logs = evidence.get("party_logs")
    if not isinstance(logs, list) or len(logs) < 3:
        raise GateError(f"{name}: at least three hashed party logs are required")
    paths = []
    for index, item in enumerate(logs):
        if not isinstance(item, dict):
            raise GateError(f"{name}.party_logs[{index}]: expected object")
        paths.append(
            verify_hashed_file(item.get("path"), item.get("sha256"), f"{name}.party_logs[{index}]")
        )
    if len({path.resolve() for path in paths}) < 3:
        raise GateError(f"{name}: party logs are not three distinct files")
    if manifest.stat().st_size == 0:  # explicit for readability in gate reports
        raise GateError(f"{name}: empty manifest")


def validate_accuracy(payload: dict[str, Any], dataset: str) -> dict[str, Any]:
    name = f"{dataset}_accuracy"
    if payload.get("schema") != "hpmpc_fid203_accuracy_v1":
        raise GateError(f"{name}: unsupported schema")
    if payload.get("status") != "COMPLETE":
        raise GateError(f"{name}: status={payload.get('status')!r}; COMPLETE required")
    if int(payload.get("fid", -1)) != 203 or int(payload.get("layers", -1)) != 12:
        raise GateError(f"{name}: result is not full twelve-layer FID=203")
    if str(payload.get("dataset", "")).lower() != dataset:
        raise GateError(f"{name}: dataset mismatch")
    runs = payload.get("mpc_runs")
    if int(payload.get("required_runs", -1)) != 3 or not isinstance(runs, list) or len(runs) != 3:
        raise GateError(f"{name}: exactly three complete MPC runs are required")
    accuracies = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise GateError(f"{name}.run[{index}]: expected object")
        accuracies.append(finite_number(run.get("accuracy_percent"), f"{name}.run[{index}].accuracy"))
        validate_party_evidence(run.get("evidence"), f"{name}.run[{index}]")
        evidence = run["evidence"]
        verify_hashed_file(
            evidence.get("prediction_path"), evidence.get("prediction_sha256"),
            f"{name}.run[{index}].predictions",
        )
    mean = finite_number(payload.get("mean"), f"{name}.mean")
    std = finite_number(payload.get("std"), f"{name}.std", nonnegative=True)
    if not math.isclose(mean, statistics.mean(accuracies), abs_tol=1e-9):
        raise GateError(f"{name}: stored mean does not match run values")
    if not math.isclose(std, statistics.stdev(accuracies), abs_tol=1e-9):
        raise GateError(f"{name}: stored std does not match run values")
    plaintext = finite_number(payload.get("plaintext_acc"), f"{name}.plaintext")
    drop = finite_number(payload.get("drop"), f"{name}.drop")
    if not math.isclose(drop, plaintext - mean, abs_tol=1e-9):
        raise GateError(f"{name}: stored accuracy drop is inconsistent")
    return {
        "mean": mean,
        "std": std,
        "plaintext": plaintext,
        "drop": drop,
        "acceptance_passed": bool(payload.get("acceptance_passed")),
    }


def validate_complete_perf_run(run: Any, name: str) -> dict[str, float]:
    if not isinstance(run, dict) or run.get("status") != "COMPLETE":
        raise GateError(f"{name}: run is not COMPLETE")
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise GateError(f"{name}: missing metrics")
    values = {
        metric: finite_number(metrics.get(metric), f"{name}.{metric}", nonnegative=True)
        for metric in METRICS
    }
    if values["latency_ms"] <= 0 or values["throughput_sps"] <= 0:
        raise GateError(f"{name}: latency and throughput must be positive")
    if values["nic_util_pct"] > 100:
        raise GateError(f"{name}: nic_util_pct exceeds 100")
    validate_party_evidence(run.get("evidence"), name)
    return values


def validate_oom_run(run: Any, name: str) -> None:
    if not isinstance(run, dict) or run.get("status") != "OOM":
        raise GateError(f"{name}: missing explicit OOM run")
    validate_party_evidence(run.get("evidence"), name)


def validate_performance(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    if payload.get("schema") != "hpmpc_fid203_performance_v1":
        raise GateError("performance: unsupported schema")
    if payload.get("status") not in {"COMPLETE", "COMPLETE_WITH_OOM"}:
        raise GateError(f"performance: status={payload.get('status')!r}")
    if int(payload.get("fid", -1)) != 203 or int(payload.get("layers", -1)) != 12:
        raise GateError("performance: result is not full twelve-layer FID=203")
    if int(payload.get("parties", -1)) != 3 or payload.get("scope") != "full-bert-e2e":
        raise GateError("performance: real full-model 3PC provenance is missing")
    if payload.get("experiment_kind") != "performance" or int(payload.get("required_runs_per_group", -1)) != 5:
        raise GateError("performance: expected paper matrix with five runs per group")
    if set(payload.get("requested_batches", [])) != set(REQUIRED_BATCHES):
        raise GateError("performance: required batch matrix is incomplete")
    if set(payload.get("requested_methods", [])) != set(REQUIRED_METHODS):
        raise GateError("performance: serial/pipeline matrix is incomplete")
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise GateError("performance: groups must be a list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise GateError("performance: malformed group")
        key = (str(group.get("method")), int(group.get("batch", 0)))
        if key in indexed:
            raise GateError(f"performance: duplicate group {key}")
        indexed[key] = group
    required = {(method, batch) for method in REQUIRED_METHODS for batch in REQUIRED_BATCHES}
    if set(indexed) != required:
        raise GateError(f"performance: missing/unexpected groups: {sorted(required ^ set(indexed))}")

    for key, group in indexed.items():
        name = f"performance.{key[0]}.batch{key[1]}"
        runs = group.get("runs")
        if not isinstance(runs, list):
            raise GateError(f"{name}: runs must be a list")
        if group.get("status") == "COMPLETE":
            if len(runs) != 5:
                raise GateError(f"{name}: exactly five runs are required")
            values = [validate_complete_perf_run(run, f"{name}.run[{index}]") for index, run in enumerate(runs)]
            summary = group.get("summary")
            if not isinstance(summary, dict):
                raise GateError(f"{name}: missing summary")
            for metric in METRICS:
                observed_mean = finite_number(summary.get(metric, {}).get("mean"), f"{name}.{metric}.mean")
                observed_std = finite_number(summary.get(metric, {}).get("std"), f"{name}.{metric}.std")
                expected = [value[metric] for value in values]
                if not math.isclose(observed_mean, statistics.mean(expected), abs_tol=1e-9):
                    raise GateError(f"{name}: inconsistent {metric} mean")
                if not math.isclose(observed_std, statistics.stdev(expected), abs_tol=1e-9):
                    raise GateError(f"{name}: inconsistent {metric} std")
        elif group.get("status") == "OOM":
            oom_runs = [run for run in runs if isinstance(run, dict) and run.get("status") == "OOM"]
            if len(oom_runs) != 1:
                raise GateError(f"{name}: OOM group must contain one attempted OOM run")
            validate_oom_run(oom_runs[0], f"{name}.oom")
        else:
            raise GateError(f"{name}: status={group.get('status')!r} is BLOCKED/incomplete")
    for method in REQUIRED_METHODS:
        if indexed[(method, 1)].get("status") != "COMPLETE":
            raise GateError(f"performance: batch=1 {method} must complete for the paper table")
    return indexed


def validate_bandwidth_perf(path: Path, label: str) -> None:
    payload = load_json(path, f"bandwidth.{label}.performance_result")
    if payload.get("schema") != "hpmpc_fid203_performance_v1" or payload.get("experiment_kind") != "bandwidth":
        raise GateError(f"bandwidth.{label}: invalid nested performance schema")
    if payload.get("status") != "COMPLETE" or int(payload.get("required_runs_per_group", -1)) != 3:
        raise GateError(f"bandwidth.{label}: nested result is not three-run COMPLETE")
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("status") != "COMPLETE":
        raise GateError(f"bandwidth.{label}: nested result lacks one complete group")
    runs = groups[0].get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise GateError(f"bandwidth.{label}: nested result lacks three runs")
    for index, run in enumerate(runs):
        validate_complete_perf_run(run, f"bandwidth.{label}.run[{index}]")


def validate_bandwidth(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    if payload.get("schema") != "hpmpc_fid203_bandwidth_v1" or payload.get("status") != "COMPLETE":
        raise GateError(f"bandwidth: status/schema invalid ({payload.get('status')!r})")
    if int(payload.get("fid", -1)) != 203 or int(payload.get("layers", -1)) != 12:
        raise GateError("bandwidth: result is not full twelve-layer FID=203")
    if int(payload.get("parties", -1)) != 3 or int(payload.get("required_runs_per_point", -1)) != 3:
        raise GateError("bandwidth: three real parties/runs are required")
    if payload.get("network_restored") is not True:
        raise GateError("bandwidth: qdisc restoration was not verified")
    before = payload.get("qdisc_before")
    after = payload.get("qdisc_after")
    if not isinstance(before, dict) or len(before) != 3:
        raise GateError("bandwidth: missing three-host qdisc_before evidence")
    if not isinstance(after, dict) or len(after) != 3:
        raise GateError("bandwidth: missing three-host qdisc_after evidence")
    for host, snapshot in after.items():
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise GateError(f"bandwidth: empty restored qdisc snapshot for {host}")
        if any(line.startswith("qdisc tbf ") and " root" in line for line in snapshot.splitlines()):
            raise GateError(f"bandwidth: TBF still present after restoration on {host}")
    points = payload.get("points")
    if not isinstance(points, list):
        raise GateError("bandwidth: points must be a list")
    indexed: dict[float, dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            raise GateError("bandwidth: malformed point")
        bandwidth = finite_number(point.get("bandwidth_gbps"), "bandwidth.point", nonnegative=True)
        if bandwidth in indexed:
            raise GateError(f"bandwidth: duplicate point {bandwidth}")
        if point.get("status") != "COMPLETE" or int(point.get("runs", -1)) != 3:
            raise GateError(f"bandwidth.{bandwidth}: not three-run COMPLETE")
        for metric in ("latency_ms", "latency_ms_per_sample", "nic_util_pct"):
            stats = point.get(metric)
            if not isinstance(stats, dict):
                raise GateError(f"bandwidth.{bandwidth}: missing {metric}")
            finite_number(stats.get("mean"), f"bandwidth.{bandwidth}.{metric}.mean", nonnegative=True)
            finite_number(stats.get("std"), f"bandwidth.{bandwidth}.{metric}.std", nonnegative=True)
        nested = Path(str(point.get("performance_result", "")))
        validate_bandwidth_perf(nested, str(bandwidth))
        indexed[bandwidth] = point
    if set(indexed) != set(REQUIRED_BANDWIDTHS):
        raise GateError("bandwidth: required 1/10/15 Gbps points are incomplete")
    return indexed


def pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def group_stat(group: dict[str, Any], metric: str) -> tuple[float, float]:
    summary = group["summary"][metric]
    return float(summary["mean"]), float(summary["std"])


def build_table(
    mrpc: dict[str, Any],
    rte: dict[str, Any],
    performance: dict[tuple[str, int], dict[str, Any]],
    bandwidth: dict[float, dict[str, Any]],
) -> str:
    rows = []
    for method, label in (("serial", "本文（无流水线）"), ("pipeline", "本文（流水线）")):
        group = performance[(method, 1)]
        latency_ms = group_stat(group, "latency_ms")
        communication = group_stat(group, "comm_gb")
        nic = group_stat(group, "nic_util_pct")
        rows.append(
            f"| {label} | Trio 3PC | {pm(mrpc['mean'], mrpc['std'], 2)} | "
            f"{pm(rte['mean'], rte['std'], 2)} | "
            f"{pm(latency_ms[0] / 1000.0, latency_ms[1] / 1000.0)} | "
            f"{pm(*communication)} | {pm(*nic, digits=2)}% |"
        )
    acceptance = mrpc["acceptance_passed"] and rte["acceptance_passed"]
    lines = [
        "# FID=203 BERT-base 论文实验结果",
        "",
        f"**状态：{'COMPLETE' if acceptance else 'COMPLETE_MEASURED_ACCEPTANCE_FAILED'}**",
        "",
        "配置：3×NVIDIA A10 24GB，Ubuntu 22.04，15Gbps 内网，hpmpc Trio 半诚实3PC，"
        "FID=203，BERT-base 12层，DATTYPE=512，FRACTIONAL=20，PROCESS_NUM=1，SPLITROLES=0。",
        "",
        "| 方法 | 协议 | MRPC(%) | RTE(%) | 延迟(s) | 通信(GB) | NIC利用率 |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| PUMA | {PUMA['protocol']} | {PUMA['mrpc']} | {PUMA['rte']} | {PUMA['latency_seconds']} | {PUMA['communication_gb']} | {PUMA['nic']} |",
        *rows,
        "",
        "PUMA 数值为论文公开值（BERT-base/RTE，序列长度128），不是本项目实测值；PUMA未报告MRPC。",
        "",
        "## 精度门禁",
        "",
        f"- MRPC：明文 {mrpc['plaintext']:.2f}%，MPC {pm(mrpc['mean'], mrpc['std'], 2)}%，下降 {mrpc['drop']:.2f} pp。",
        f"- RTE：明文 {rte['plaintext']:.2f}%，MPC {pm(rte['mean'], rte['std'], 2)}%，下降 {rte['drop']:.2f} pp。",
        "",
        "## Batch性能矩阵",
        "",
        "| 调度 | Batch | 状态 | 延迟(ms/inference) | 吞吐(samples/s) | 通信(GB/inference) | GPU显存(MB) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for method in REQUIRED_METHODS:
        for batch in REQUIRED_BATCHES:
            group = performance[(method, batch)]
            if group["status"] == "OOM":
                lines.append(f"| {method} | {batch} | OOM（实测并留存日志） | — | — | — | — |")
            else:
                latency = group_stat(group, "latency_ms")
                throughput = group_stat(group, "throughput_sps")
                communication = group_stat(group, "comm_gb")
                gpu = group_stat(group, "gpu_mem_mb")
                lines.append(
                    f"| {method} | {batch} | COMPLETE | {pm(*latency)} | {pm(*throughput)} | "
                    f"{pm(*communication)} | {pm(*gpu, digits=1)} |"
                )
    lines.extend(
        [
            "",
            "## 带宽敏感性",
            "",
            "| 带宽(Gbps) | 延迟(ms/sample) | NIC利用率 | 运行数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for gbps in REQUIRED_BANDWIDTHS:
        point = bandwidth[gbps]
        latency = point["latency_ms_per_sample"]
        nic = point["nic_util_pct"]
        lines.append(
            f"| {gbps:g} | {pm(float(latency['mean']), float(latency['std']))} | "
            f"{pm(float(nic['mean']), float(nic['std']), 2)}% | 3 |"
        )
    lines.extend(
        [
            "",
            "所有本文数据均由带哈希的 FID=203 manifest、三方日志和原始输出通过完整性门禁后生成。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrpc", type=Path, default=Path("results/accuracy_mrpc.json"))
    parser.add_argument("--rte", type=Path, default=Path("results/accuracy_rte.json"))
    parser.add_argument("--performance", type=Path, default=Path("results/perf_results.json"))
    parser.add_argument("--bandwidth", type=Path, default=Path("results/bandwidth_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_table.md"))
    parser.add_argument("--gate-report", type=Path, default=Path("results/collection_gate.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = {
        "mrpc_accuracy": args.mrpc,
        "rte_accuracy": args.rte,
        "performance": args.performance,
        "bandwidth": args.bandwidth,
    }
    errors: list[str] = []
    validated: dict[str, Any] = {}
    for name, path in sources.items():
        try:
            payload = load_json(path, name)
            if name == "mrpc_accuracy":
                validated[name] = validate_accuracy(payload, "mrpc")
            elif name == "rte_accuracy":
                validated[name] = validate_accuracy(payload, "rte")
            elif name == "performance":
                validated[name] = validate_performance(payload)
            else:
                validated[name] = validate_bandwidth(payload)
        except (GateError, OSError, TypeError, ValueError) as error:
            errors.append(str(error))

    report: dict[str, Any] = {
        "schema": "hpmpc_fid203_collection_gate_v1",
        "status": "BLOCKED" if errors else "COMPLETE",
        "fid": 203,
        "required_keys": [
            "mrpc_accuracy", "rte_accuracy", "latency_batch1", "comm_gb_batch1",
            "bandwidth_1g", "bandwidth_10g", "bandwidth_15g",
        ],
        "sources": {name: str(path) for name, path in sources.items()},
        "source_sha256": {
            name: sha256_file(path)
            for name, path in sources.items()
            if path.is_file() and path.stat().st_size > 0
        },
        "errors": errors,
        "table_written": False,
    }
    args.gate_report.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        args.gate_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BLOCKED: refusing to generate/overwrite the paper table", file=__import__("sys").stderr)
        for error in errors:
            print(f"- {error}", file=__import__("sys").stderr)
        return 2

    table = build_table(
        validated["mrpc_accuracy"], validated["rte_accuracy"],
        validated["performance"], validated["bandwidth"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    report["table_written"] = True
    report["output"] = str(args.output)
    report["accuracy_acceptance_passed"] = bool(
        validated["mrpc_accuracy"]["acceptance_passed"]
        and validated["rte_accuracy"]["acceptance_passed"]
    )
    args.gate_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(table)
    return 0 if report["accuracy_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
