#!/usr/bin/env python3
"""Evaluate real FID=203 predictions on GLUE MRPC or RTE.

The script never launches MPC and never treats a bare prediction array as MPC
evidence.  Every run must have a manifest proving that it came from the full
three-party, twelve-layer FID=203 graph and pointing to three non-empty party
logs plus a complete prediction/logit file.  Missing or invalid evidence is
written as a BLOCKED result and returns a non-zero status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


TASKS: dict[str, dict[str, Any]] = {
    "mrpc": {
        "subset": "mrpc",
        "expected": 408,
        "fields": ("sentence1", "sentence2"),
        "model": "textattack/bert-base-uncased-MRPC",
        "max_drop_pp": 3.0,
    },
    "rte": {
        "subset": "rte",
        "expected": 277,
        "fields": ("sentence1", "sentence2"),
        "model": "textattack/bert-base-uncased-RTE",
        "max_drop_pp": 3.0,
    },
}
COMPLETE_STATUSES = {"PASS", "PASSED", "COMPLETE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def nonempty_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{description} is missing or empty: {path}")
    return path


def _manifest_integer(payload: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name in payload:
            try:
                return int(payload[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"manifest field {name!r} is not an integer") from error
    return None


def validate_run_manifest(
    manifest_path: Path,
    task_name: str,
    expected_examples: int,
    fid: int,
) -> tuple[Path, dict[str, Any]]:
    nonempty_file(manifest_path, "MPC run manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest root must be an object")
    status = str(payload.get("status", "")).upper()
    if status not in COMPLETE_STATUSES:
        raise ValueError(f"{manifest_path}: status={status or 'MISSING'}, expected COMPLETE/PASS")
    observed_fid = _manifest_integer(payload, "fid", "function_identifier")
    if observed_fid != fid:
        raise ValueError(f"{manifest_path}: FID={observed_fid!r}, required FID={fid}")
    layers = _manifest_integer(payload, "layers", "encoder_layers", "num_hidden_layers")
    if layers != 12:
        raise ValueError(f"{manifest_path}: layers={layers!r}, required full 12-layer encoder")
    parties = _manifest_integer(payload, "parties", "party_count")
    if parties != 3:
        raise ValueError(f"{manifest_path}: parties={parties!r}, required real 3PC")
    if str(payload.get("dataset", "")).lower() != task_name:
        raise ValueError(f"{manifest_path}: dataset does not match {task_name}")
    if str(payload.get("split", "validation")).lower() != "validation":
        raise ValueError(f"{manifest_path}: only the validation split is accepted")
    examples = _manifest_integer(payload, "examples", "sample_count")
    if examples != expected_examples:
        raise ValueError(
            f"{manifest_path}: examples={examples!r}, expected complete set of {expected_examples}"
        )

    prediction_value = next(
        (
            payload[name]
            for name in ("predictions_path", "prediction_path", "logits_path", "output_path")
            if payload.get(name)
        ),
        None,
    )
    if prediction_value is None:
        raise ValueError(f"{manifest_path}: missing predictions_path/logits_path")
    prediction_path = nonempty_file(
        resolve_artifact(manifest_path, prediction_value), "MPC prediction artifact"
    )

    raw_logs = payload.get("party_logs")
    if isinstance(raw_logs, dict):
        raw_logs = list(raw_logs.values())
    if not isinstance(raw_logs, list) or len(raw_logs) < 3:
        raise ValueError(f"{manifest_path}: party_logs must contain at least three logs")
    logs = [
        nonempty_file(resolve_artifact(manifest_path, value), "party log")
        for value in raw_logs
    ]
    if len({path.resolve() for path in logs}) < 3:
        raise ValueError(f"{manifest_path}: party_logs must identify three distinct files")

    evidence = {
        "run_id": str(payload.get("run_id", manifest_path.stem)),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "party_logs": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in logs
        ],
    }
    return prediction_path, evidence


def load_prediction_array(path: Path, expected: int):
    import numpy as np

    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "logits" if "logits" in archive else "predictions"
            if key not in archive:
                raise ValueError(f"{path}: expected logits or predictions array")
            value = archive[key]
    elif path.suffix in {".json", ".jsonl"}:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            value = np.asarray([row.get("logits", row.get("prediction")) for row in rows])
        else:
            payload = json.loads(path.read_text())
            value = np.asarray(payload.get("logits", payload.get("predictions", payload)))
    else:
        raise ValueError(f"unsupported MPC prediction format: {path}")
    value = np.asarray(value)
    if value.ndim == 2:
        if value.shape != (expected, 2):
            raise ValueError(f"{path}: logits must have shape [{expected},2], got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: logits contain NaN/Inf")
        value = value.argmax(axis=1)
    elif value.ndim != 1 or value.shape[0] != expected:
        raise ValueError(f"{path}: labels must have shape [{expected}], got {value.shape}")
    value = value.astype(np.int64, copy=False)
    if not np.isin(value, [0, 1]).all():
        raise ValueError(f"{path}: predictions must be binary labels")
    return value


def metrics(labels, predictions, include_f1: bool) -> dict[str, float]:
    import numpy as np

    accuracy = float(np.mean(labels == predictions) * 100.0)
    result = {"accuracy_percent": accuracy}
    if include_f1:
        true_positive = int(np.sum((labels == 1) & (predictions == 1)))
        false_positive = int(np.sum((labels == 0) & (predictions == 1)))
        false_negative = int(np.sum((labels == 1) & (predictions == 0)))
        denominator = 2 * true_positive + false_positive + false_negative
        result["f1_percent"] = 0.0 if denominator == 0 else 200.0 * true_positive / denominator
    return result


def batch_ranges(length: int, size: int) -> Iterable[tuple[int, int]]:
    for begin in range(0, length, size):
        yield begin, min(begin + size, length)


def plaintext_predictions(
    dataset: Any,
    task: dict[str, Any],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
    local_files_only: bool,
):
    import numpy as np

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("plaintext inference requires torch and transformers") from error
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=local_files_only
    ).eval().to(device)
    logits: list[Any] = []
    left_name, right_name = task["fields"]
    with torch.inference_mode():
        for begin, end in batch_ranges(len(dataset), batch_size):
            encoded = tokenizer(
                dataset[begin:end][left_name],
                dataset[begin:end][right_name],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits.append(model(**encoded).logits.detach().cpu().numpy())
    joined = np.concatenate(logits, axis=0).astype(np.float32, copy=False)
    return joined.argmax(axis=1).astype(np.int64), joined


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", required=True, type=str.lower, choices=tuple(TASKS))
    result.add_argument("--split", default="validation", choices=("validation",))
    result.add_argument("--fid", type=int, default=203)
    result.add_argument("--runs", type=int, default=3)
    result.add_argument("--mpc-run-manifest", type=Path, action="append", default=[])
    result.add_argument(
        "--mpc-predictions", type=Path, action="append", default=[],
        help="Rejected without manifests; retained only to catch legacy unsafe invocations.",
    )
    result.add_argument("--model")
    result.add_argument("--plaintext-predictions", type=Path)
    result.add_argument("--save-plaintext-logits", type=Path)
    result.add_argument("--output", "--output-json", dest="output", type=Path)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--max-length", type=int, default=128)
    result.add_argument("--device", default="auto")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output or Path(f"results/accuracy_{args.dataset}.json")
    if output.exists() and not args.overwrite:
        print(f"BLOCKED: refusing to overwrite existing result {output}; pass --overwrite", file=sys.stderr)
        return 2
    base: dict[str, Any] = {
        "schema": "hpmpc_fid203_accuracy_v1",
        "dataset": args.dataset,
        "split": args.split,
        "fid": args.fid,
        "layers": 12,
        "required_runs": args.runs,
    }
    if args.fid != 203:
        write_result(output, {**base, "status": "BLOCKED_WRONG_FID", "reason": "only FID=203 is full BERT"})
        return 2
    if args.runs != 3:
        write_result(output, {**base, "status": "BLOCKED_INVALID_RUN_COUNT", "reason": "paper accuracy requires exactly 3 runs"})
        return 2
    if args.mpc_predictions:
        write_result(
            output,
            {
                **base,
                "status": "BLOCKED_UNPROVEN_MPC_OUTPUT",
                "reason": "bare --mpc-predictions are rejected; provide one FID=203 run manifest per run",
            },
        )
        return 2

    manifests = list(args.mpc_run_manifest)
    if not manifests:
        manifests = sorted(Path("results/accuracy/runs").glob(f"{args.dataset}_fid203_run*.manifest.json"))
    if len(manifests) != args.runs:
        write_result(
            output,
            {
                **base,
                "status": "BLOCKED_MISSING_FID203_RUNS",
                "reason": f"found {len(manifests)} manifests; exactly {args.runs} are required",
                "manifests": [str(path) for path in manifests],
            },
        )
        return 2

    task = TASKS[args.dataset]
    try:
        validated = [
            validate_run_manifest(path, args.dataset, task["expected"], args.fid)
            for path in manifests
        ]
        run_ids = [evidence["run_id"] for _, evidence in validated]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("MPC run manifests must have distinct run_id values")
        prediction_paths = [str(path.resolve()) for path, _ in validated]
        if len(set(prediction_paths)) != len(prediction_paths):
            raise ValueError("each MPC run must provide a distinct prediction artifact")
        all_log_paths = [
            item["path"]
            for _, evidence in validated
            for item in evidence["party_logs"]
        ]
        if len(set(all_log_paths)) != len(all_log_paths):
            raise ValueError("party logs may not be reused across independent MPC runs")
    except (OSError, ValueError) as error:
        write_result(output, {**base, "status": "BLOCKED_INVALID_MPC_EVIDENCE", "reason": str(error)})
        return 2

    try:
        import numpy as np
        from datasets import load_dataset

        dataset = load_dataset("nyu-mll/glue", task["subset"], split=args.split)
        if len(dataset) != task["expected"]:
            raise RuntimeError(
                f"{args.dataset} validation has {len(dataset)} rows; expected {task['expected']}"
            )
        labels = np.asarray(dataset["label"], dtype=np.int64)
        if args.plaintext_predictions:
            plain = load_prediction_array(args.plaintext_predictions, len(dataset))
            plaintext_logits = None
            model_name = "external-plaintext-predictions"
        else:
            model_name = args.model or task["model"]
            plain, plaintext_logits = plaintext_predictions(
                dataset, task, model_name, args.batch_size, args.max_length,
                args.device, args.local_files_only,
            )
            if args.save_plaintext_logits:
                args.save_plaintext_logits.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.save_plaintext_logits, plaintext_logits, allow_pickle=False)

        include_f1 = args.dataset == "mrpc"
        plain_metrics = metrics(labels, plain, include_f1)
        runs = []
        for prediction_path, evidence in validated:
            prediction = load_prediction_array(prediction_path, len(dataset))
            run_metrics = metrics(labels, prediction, include_f1)
            runs.append({**run_metrics, "evidence": evidence})
        accuracies = np.asarray([run["accuracy_percent"] for run in runs], dtype=np.float64)
        drops = plain_metrics["accuracy_percent"] - accuracies
    except Exception as error:  # dependency/data/model failures are recorded, never backfilled
        write_result(output, {**base, "status": "BLOCKED_EVALUATION_FAILED", "reason": str(error)})
        return 2

    mean = float(accuracies.mean())
    std = float(accuracies.std(ddof=1))
    mean_drop = float(plain_metrics["accuracy_percent"] - mean)
    maximum_drop = float(drops.max())
    acceptance = bool(math.isfinite(maximum_drop) and maximum_drop < task["max_drop_pp"])
    result = {
        **base,
        "status": "COMPLETE",
        "examples": len(dataset),
        "dataset_source": "nyu-mll/glue",
        "model": model_name,
        "max_length": args.max_length,
        "plaintext": plain_metrics,
        "mpc_runs": runs,
        "mpc_acc": accuracies.tolist(),
        "plaintext_acc": plain_metrics["accuracy_percent"],
        "mean": mean,
        "std": std,
        "drop": mean_drop,
        "accuracy_drop_max_pp": maximum_drop,
        "allowed_drop_pp": task["max_drop_pp"],
        "acceptance_passed": acceptance,
    }
    write_result(output, result)
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
