#!/usr/bin/env python3
"""Verify a PROMPT-D full BERT export without trusting its manifest.

The default mode performs three checks:

1. the manifest is the canonical 12-layer MRPC/RTE schema;
2. every counted-Q20 file has the declared header, size and SHA-256;
3. every payload is bit-exact with freshly quantized Hugging Face weights and
   its Q12.20 round-trip error is below the requested tolerance.

Use ``--integrity-only`` on a machine without PyTorch/Transformers.  It still
checks all 203 tensor files and all hashes, but cannot establish provenance
against the source checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    from export_bert_full import (
        COUNT_HEADER,
        FRACTIONAL,
        MANIFEST_FORMAT,
        SCALE,
        TASKS,
        _load_model,
        _parse_tasks,
        _resolve_state_dict,
        _validate_encoder_config,
        build_specs,
        quantize_q20,
        sha256_file,
        verify_layer0_ffn_bit_exact,
    )
except ImportError:  # Support ``python -m nn.Pygeon.verify_export``.
    from .export_bert_full import (  # type: ignore[no-redef]
        COUNT_HEADER,
        FRACTIONAL,
        MANIFEST_FORMAT,
        SCALE,
        TASKS,
        _load_model,
        _parse_tasks,
        _resolve_state_dict,
        _validate_encoder_config,
        build_specs,
        quantize_q20,
        sha256_file,
        verify_layer0_ffn_bit_exact,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return document


def _safe_tensor_path(root: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError(f"unsafe tensor filename in manifest: {filename!r}")
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"tensor path escapes export directory: {filename!r}") from error
    return path


def _validate_manifest_schema(manifest: dict[str, Any]) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"unsupported manifest format {manifest.get('format')!r}; expected {MANIFEST_FORMAT!r}"
        )
    if manifest.get("fractional") != FRACTIONAL or manifest.get("scale") != SCALE:
        raise ValueError(
            f"fixed-point mismatch: fractional={manifest.get('fractional')}, "
            f"scale={manifest.get('scale')}"
        )
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("manifest tasks must be a non-empty list")
    tasks = tuple(str(task).lower() for task in raw_tasks)
    if len(tasks) != len(set(tasks)) or any(task not in TASKS for task in tasks):
        raise ValueError(f"invalid manifest tasks: {raw_tasks!r}")

    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("manifest models must be an object")
    for source_role in ("encoder",) + tasks:
        if not isinstance(models.get(source_role), str) or not models[source_role]:
            raise ValueError(f"manifest has no valid model for source_role={source_role}")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("manifest parameters must be a list")
    expected_specs = build_specs(tasks)
    if len(parameters) != len(expected_specs):
        raise ValueError(
            f"manifest has {len(parameters)} tensor entries; expected {len(expected_specs)}"
        )
    if manifest.get("tensor_file_count") != len(expected_specs):
        raise ValueError("manifest tensor_file_count does not match the canonical schema")
    if manifest.get("directory_entry_count_with_manifest") != len(expected_specs) + 1:
        raise ValueError("manifest directory_entry_count_with_manifest is inconsistent")

    total_count = 0
    total_bytes = 0
    for ordinal, (entry, spec) in enumerate(zip(parameters, expected_specs)):
        if not isinstance(entry, dict):
            raise ValueError(f"parameters[{ordinal}] must be an object")
        expected_fields = {
            "ordinal": ordinal,
            "logical_name": spec.logical_name,
            "filename": spec.filename,
            "shape": list(spec.shape),
            "state_dict_key": spec.state_dict_key,
            "source_role": spec.source_role,
            "count": spec.count,
            "dtype": "int32_le",
            "fractional": FRACTIONAL,
            "file_size_bytes": COUNT_HEADER.size + spec.count * 4,
            "source_model": models[spec.source_role],
        }
        for key, expected in expected_fields.items():
            if entry.get(key) != expected:
                raise ValueError(
                    f"parameters[{ordinal}].{key}={entry.get(key)!r}; expected {expected!r}"
                )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"parameters[{ordinal}].sha256 is not a SHA-256 hex digest")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError(f"parameters[{ordinal}].sha256 is not hexadecimal") from error
        total_count += spec.count
        total_bytes += COUNT_HEADER.size + spec.count * 4

    if manifest.get("total_parameter_count") != total_count:
        raise ValueError("manifest total_parameter_count does not equal the tensor sum")
    if manifest.get("total_tensor_bytes_including_headers") != total_bytes:
        raise ValueError("manifest total_tensor_bytes_including_headers does not equal the tensor sum")
    return tasks, parameters


def _verify_entry_integrity(root: Path, entry: dict[str, Any]) -> None:
    path = _safe_tensor_path(root, entry["filename"])
    if not path.is_file():
        raise ValueError(f"missing tensor file: {path}")
    expected_size = int(entry["file_size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: size={actual_size}, expected={expected_size}")
    with path.open("rb") as stream:
        raw_header = stream.read(COUNT_HEADER.size)
    if len(raw_header) != COUNT_HEADER.size:
        raise ValueError(f"{path}: missing element-count header")
    count = COUNT_HEADER.unpack(raw_header)[0]
    if count != entry["count"]:
        raise ValueError(f"{path}: header count={count}, expected={entry['count']}")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"{path}: sha256={digest}, expected={entry['sha256']}")


def _verify_integrity(root: Path, parameters: Sequence[dict[str, Any]]) -> None:
    for index, entry in enumerate(parameters, start=1):
        _verify_entry_integrity(root, entry)
        print(
            f"integrity {index:3d}/{len(parameters)} {entry['filename']:45s} PASS",
            flush=True,
        )


def _read_q20(path: Path, count: int) -> Any:
    import numpy as np

    with path.open("rb") as stream:
        raw_header = stream.read(COUNT_HEADER.size)
        if len(raw_header) != COUNT_HEADER.size or COUNT_HEADER.unpack(raw_header)[0] != count:
            raise ValueError(f"{path}: invalid counted-Q20 header")
        values = np.fromfile(stream, dtype="<i4", count=count)
        trailing = stream.read(1)
    if values.size != count or trailing:
        raise ValueError(f"{path}: invalid counted-Q20 payload length")
    return values


def _verify_source_group(
    root: Path,
    source_role: str,
    model_id: str,
    entries: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, int]:
    import numpy as np

    print(f"loading verification source_role={source_role} model={model_id}", flush=True)
    model = _load_model(model_id, source_role, args)
    try:
        if source_role == "encoder":
            _validate_encoder_config(getattr(model, "bert", model))
        elif getattr(model.config, "num_labels", None) != 2:
            raise ValueError(f"{source_role} classifier does not have two labels")
        state = _resolve_state_dict(model, source_role)
        global_max_error = 0.0
        verified = 0
        for entry in entries:
            key = entry["state_dict_key"]
            if key not in state:
                raise KeyError(f"{model_id}: missing state-dict key {key!r}")
            original = state[key].detach().cpu().float().numpy()
            expected_shape = tuple(entry["shape"])
            if tuple(original.shape) != expected_shape:
                raise ValueError(
                    f"{entry['logical_name']}: source shape={tuple(original.shape)}, "
                    f"expected={expected_shape}"
                )
            stored = _read_q20(root / entry["filename"], int(entry["count"])).reshape(expected_shape)
            expected = quantize_q20(original)
            if not np.array_equal(stored, expected):
                mismatches = np.flatnonzero(stored.reshape(-1) != expected.reshape(-1))
                first = int(mismatches[0])
                raise AssertionError(
                    f"{entry['filename']}: not bit-exact with source quantization; "
                    f"first element={first}, stored={int(stored.reshape(-1)[first])}, "
                    f"expected={int(expected.reshape(-1)[first])}"
                )
            restored = stored.astype(np.float64) / SCALE
            error = float(np.max(np.abs(restored - original.astype(np.float64))))
            if not np.isfinite(error) or error >= args.tolerance:
                raise AssertionError(
                    f"{entry['filename']}: max_abs_error={error:.9g} is not below {args.tolerance}"
                )
            global_max_error = max(global_max_error, error)
            verified += 1
            print(
                f"source    {entry['filename']:45s} max_abs_error={error:.9g} PASS",
                flush=True,
            )
        return global_max_error, verified
    finally:
        del model


def _run_self_test() -> None:
    """Exercise counted-header, size and hash checks using only stdlib."""
    values = (0, 1, -1, 1048576, -2147483648, 2147483647)
    with tempfile.TemporaryDirectory(prefix="verify-export-self-test-") as temporary:
        root = Path(temporary)
        path = root / "fixture.bin"
        path.write_bytes(COUNT_HEADER.pack(len(values)) + struct.pack(f"<{len(values)}i", *values))
        entry = {
            "filename": path.name,
            "count": len(values),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        _verify_entry_integrity(root, entry)
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 0x01
        path.write_bytes(payload)
        try:
            _verify_entry_integrity(root, entry)
        except ValueError as error:
            if "sha256" not in str(error):
                raise
        else:
            raise AssertionError("self-test failed to detect payload corruption")
    print("SELF-TEST PASS: counted header, size and SHA-256 corruption detection")


def build_parser() -> argparse.ArgumentParser:
    default_manifest = Path(__file__).resolve().parent / "weights/manifest.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", help="override the revision stored in the manifest")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="skip Hugging Face provenance and numerical comparison",
    )
    parser.add_argument("--encoder-model", help="override manifest encoder checkpoint")
    parser.add_argument("--mrpc-model", help="override manifest MRPC checkpoint")
    parser.add_argument("--rte-model", help="override manifest RTE checkpoint")
    parser.add_argument("--reference-layer0-ffn", type=Path)
    parser.add_argument("--self-test", action="store_true", help="run a dependency-free verifier smoke test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.self_test:
        _run_self_test()
        return 0
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_files_only = bool(args.local_files_only or args.offline)

    manifest_path = args.manifest.expanduser().resolve()
    root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    tasks, parameters = _validate_manifest_schema(manifest)
    _verify_integrity(root, parameters)

    reference = args.reference_layer0_ffn
    if reference is None:
        compatibility = manifest.get("layer0_ffn_legacy_compatibility", {})
        stored_reference = compatibility.get("reference") if isinstance(compatibility, dict) else None
        if stored_reference and Path(stored_reference).expanduser().is_file():
            reference = Path(stored_reference)
    if reference is not None:
        result = verify_layer0_ffn_bit_exact(root, reference.expanduser().resolve())
        print(f"legacy layer0 FFN bit-exact compatibility: {result['status'].upper()}")

    if args.integrity_only:
        print(
            f"PASS integrity-only: {len(parameters)} files, "
            f"{manifest['total_parameter_count']:,} parameters"
        )
        return 0

    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("manifest models must be an object")
    overrides = {
        "encoder": args.encoder_model,
        "mrpc": args.mrpc_model,
        "rte": args.rte_model,
    }
    if args.revision is None:
        args.revision = manifest.get("revision")
    global_max_error = 0.0
    verified = 0
    for source_role in ("encoder",) + tasks:
        model_id = overrides[source_role] or models.get(source_role)
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"manifest has no valid model for source_role={source_role}")
        role_entries = [entry for entry in parameters if entry["source_role"] == source_role]
        error, count = _verify_source_group(root, source_role, model_id, role_entries, args)
        global_max_error = max(global_max_error, error)
        verified += count

    if verified != len(parameters):
        raise AssertionError(f"verified {verified} tensors, expected {len(parameters)}")
    print(
        f"PASS source verification: {verified} tensors are bit-exact, "
        f"global_max_abs_error={global_max_error:.9g} < {args.tolerance}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileExistsError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
