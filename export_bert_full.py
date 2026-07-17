#!/usr/bin/env python3
"""Export a complete 12-layer BERT model as PIGEON counted-Q20 files.

Each tensor is stored in its own little-endian binary file::

    int32 element_count
    int32 q12_20_values[element_count]

The quantizer and row-major tensor order intentionally match
``export_bert_ffn.py``.  For the default ``google-bert/bert-base-uncased``
encoder, the six layer-0 FFN payloads can therefore be concatenated and
compared byte-for-byte with the existing combined layer-0 FFN export.

One shared encoder plus two task heads is the layout requested by PROMPT-D.
For task-accurate evaluation, export MRPC and RTE as separate bundles, using
the corresponding fine-tuned checkpoint as both ``--encoder-model`` and the
single selected task model; one encoder cannot represent two independently
fine-tuned checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


FRACTIONAL = 20
SCALE = 1 << FRACTIONAL
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
COUNT_HEADER = struct.Struct("<i")
FORMAT_NAME = "pigeon_counted_q20_tensor_v1"
MANIFEST_FORMAT = "pigeon_bert_full_q20_manifest_v1"
DEFAULT_ENCODER_MODEL = "google-bert/bert-base-uncased"
DEFAULT_MRPC_MODEL = "textattack/bert-base-uncased-MRPC"
DEFAULT_RTE_MODEL = "textattack/bert-base-uncased-RTE"
TASKS = ("mrpc", "rte")


@dataclass(frozen=True)
class TensorSpec:
    """One exported tensor and the state-dict entry that supplies it."""

    logical_name: str
    filename: str
    shape: tuple[int, ...]
    state_dict_key: str
    source_role: str

    @property
    def count(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def _encoder_spec(
    logical_name: str,
    filename: str,
    shape: tuple[int, ...],
    state_dict_key: str,
) -> TensorSpec:
    return TensorSpec(logical_name, filename, shape, state_dict_key, "encoder")


def build_specs(tasks: Sequence[str] = TASKS) -> tuple[TensorSpec, ...]:
    """Return the canonical PROMPT-D inference-order export schema."""
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unsupported tasks: {', '.join(unknown)}")

    specs: list[TensorSpec] = [
        _encoder_spec(
            "bert.embeddings.word_embeddings.weight",
            "embeddings_word.bin",
            (30522, 768),
            "embeddings.word_embeddings.weight",
        ),
        _encoder_spec(
            "bert.embeddings.position_embeddings.weight",
            "embeddings_position.bin",
            (512, 768),
            "embeddings.position_embeddings.weight",
        ),
        _encoder_spec(
            "bert.embeddings.token_type_embeddings.weight",
            "embeddings_token_type.bin",
            (2, 768),
            "embeddings.token_type_embeddings.weight",
        ),
        _encoder_spec(
            "bert.embeddings.LayerNorm.weight",
            "embeddings_ln_weight.bin",
            (768,),
            "embeddings.LayerNorm.weight",
        ),
        _encoder_spec(
            "bert.embeddings.LayerNorm.bias",
            "embeddings_ln_bias.bin",
            (768,),
            "embeddings.LayerNorm.bias",
        ),
    ]

    per_layer = (
        ("q_weight", "attention.self.query.weight", (768, 768)),
        ("q_bias", "attention.self.query.bias", (768,)),
        ("k_weight", "attention.self.key.weight", (768, 768)),
        ("k_bias", "attention.self.key.bias", (768,)),
        ("v_weight", "attention.self.value.weight", (768, 768)),
        ("v_bias", "attention.self.value.bias", (768,)),
        ("attention_output_weight", "attention.output.dense.weight", (768, 768)),
        ("attention_output_bias", "attention.output.dense.bias", (768,)),
        ("attention_ln_weight", "attention.output.LayerNorm.weight", (768,)),
        ("attention_ln_bias", "attention.output.LayerNorm.bias", (768,)),
        ("ffn_intermediate_weight", "intermediate.dense.weight", (3072, 768)),
        ("ffn_intermediate_bias", "intermediate.dense.bias", (3072,)),
        ("ffn_output_weight", "output.dense.weight", (768, 3072)),
        ("ffn_output_bias", "output.dense.bias", (768,)),
        ("ffn_ln_weight", "output.LayerNorm.weight", (768,)),
        ("ffn_ln_bias", "output.LayerNorm.bias", (768,)),
    )
    for layer_index in range(12):
        prefix = f"encoder.layer.{layer_index}"
        for file_stem, suffix, shape in per_layer:
            specs.append(
                _encoder_spec(
                    f"bert.{prefix}.{suffix}",
                    f"layer{layer_index}_{file_stem}.bin",
                    shape,
                    f"{prefix}.{suffix}",
                )
            )

    specs.extend(
        (
            _encoder_spec(
                "bert.pooler.dense.weight",
                "pooler_weight.bin",
                (768, 768),
                "pooler.dense.weight",
            ),
            _encoder_spec(
                "bert.pooler.dense.bias",
                "pooler_bias.bin",
                (768,),
                "pooler.dense.bias",
            ),
        )
    )
    for task in tasks:
        specs.extend(
            (
                TensorSpec(
                    f"classifier.{task}.weight",
                    f"classifier_{task}_weight.bin",
                    (2, 768),
                    "classifier.weight",
                    task,
                ),
                TensorSpec(
                    f"classifier.{task}.bias",
                    f"classifier_{task}_bias.bin",
                    (2,),
                    "classifier.bias",
                    task,
                ),
            )
        )

    filenames = [spec.filename for spec in specs]
    if len(filenames) != len(set(filenames)):
        raise AssertionError("export schema contains duplicate filenames")
    return tuple(specs)


def quantize_q20(value: Any) -> Any:
    """Use the exact float32 -> float64 -> np.rint path of export_bert_ffn."""
    import numpy as np

    float32_value = np.asarray(value, dtype=np.float32)
    scaled = np.rint(np.asarray(float32_value, dtype=np.float64) * SCALE)
    return np.clip(scaled, INT32_MIN, INT32_MAX).astype("<i4")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_counted_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw_header = stream.read(COUNT_HEADER.size)
    if len(raw_header) != COUNT_HEADER.size:
        raise ValueError(f"{path}: missing 4-byte PIGEON element-count header")
    count = COUNT_HEADER.unpack(raw_header)[0]
    if count < 0:
        raise ValueError(f"{path}: invalid negative element count {count}")
    return count, path.stat().st_size


def write_counted_q20(path: Path, fixed: Any, overwrite: bool) -> dict[str, Any]:
    """Atomically write one contiguous int32 tensor in counted-Q20 format."""
    import numpy as np

    payload = np.ascontiguousarray(fixed, dtype="<i4").reshape(-1, order="C")
    count = int(payload.size)
    if count > INT32_MAX:
        raise OverflowError(f"{path}: {count} elements do not fit the int32 header")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(COUNT_HEADER.pack(count))
            stream.write(payload.tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    expected_size = COUNT_HEADER.size + count * 4
    actual_count, actual_size = read_counted_header(path)
    if actual_count != count or actual_size != expected_size:
        raise IOError(
            f"{path}: post-write validation failed: count={actual_count}/{count}, "
            f"bytes={actual_size}/{expected_size}"
        )
    return {
        "count": count,
        "file_size_bytes": expected_size,
        "sha256": sha256_file(path),
    }


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only or args.offline),
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision is not None:
        kwargs["revision"] = args.revision
    return kwargs


def _load_model(model_id: str, source_role: str, args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
    except ImportError as error:
        raise RuntimeError(
            "export requires transformers, torch and numpy; activate the project venv first"
        ) from error

    loader = AutoModel if source_role == "encoder" else AutoModelForSequenceClassification
    model = loader.from_pretrained(model_id, **_model_kwargs(args))
    model.eval()
    return model


def _validate_encoder_config(model: Any) -> None:
    config = model.config
    observed = {
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "type_vocab_size": getattr(config, "type_vocab_size", None),
    }
    expected = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
    }
    mismatches = [f"{key}={observed[key]!r} (expected {value})" for key, value in expected.items() if observed[key] != value]
    if mismatches:
        raise ValueError("checkpoint is not BERT-base-uncased: " + ", ".join(mismatches))


def _resolve_state_dict(model: Any, source_role: str) -> dict[str, Any]:
    if source_role == "encoder":
        # AutoModel returns BertModel for BERT checkpoints.  This guard also
        # supports a caller supplying an already wrapped model in tests.
        base_model = getattr(model, "bert", model)
        return base_model.state_dict()
    return model.state_dict()


def _export_source_group(
    source_role: str,
    model_id: str,
    specs: Sequence[TensorSpec],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    print(f"loading source_role={source_role} model={model_id}", flush=True)
    model = _load_model(model_id, source_role, args)
    try:
        if source_role == "encoder":
            _validate_encoder_config(getattr(model, "bert", model))
        else:
            num_labels = getattr(model.config, "num_labels", None)
            if num_labels != 2:
                raise ValueError(f"{source_role} classifier has num_labels={num_labels}, expected 2")
        state = _resolve_state_dict(model, source_role)
        entries: list[dict[str, Any]] = []
        for ordinal, spec in enumerate(specs):
            if spec.state_dict_key not in state:
                available = ", ".join(sorted(state)[:8])
                raise KeyError(
                    f"{model_id}: missing {spec.state_dict_key!r}; first available keys: {available}"
                )
            # Explicit float32 conversion preserves export_bert_ffn.py's input
            # semantics and avoids depending on the model wrapper's dtype.
            value = state[spec.state_dict_key].detach().cpu().float().numpy()
            if tuple(value.shape) != spec.shape:
                raise ValueError(
                    f"{spec.logical_name}: expected shape {spec.shape}, got {tuple(value.shape)}"
                )
            fixed = quantize_q20(value)
            metadata = write_counted_q20(output_dir / spec.filename, fixed, args.overwrite)
            entries.append(
                {
                    "ordinal": ordinal,
                    **asdict(spec),
                    "shape": list(spec.shape),
                    "dtype": "int32_le",
                    "fractional": FRACTIONAL,
                    "source_model": model_id,
                    **metadata,
                }
            )
            print(
                f"  {spec.filename:45s} shape={str(spec.shape):18s} "
                f"count={spec.count:,}",
                flush=True,
            )
        return entries
    finally:
        del model
        gc.collect()


def _layer0_ffn_files() -> tuple[str, ...]:
    return (
        "layer0_ffn_intermediate_weight.bin",
        "layer0_ffn_intermediate_bias.bin",
        "layer0_ffn_output_weight.bin",
        "layer0_ffn_output_bias.bin",
        "layer0_ffn_ln_weight.bin",
        "layer0_ffn_ln_bias.bin",
    )


def verify_layer0_ffn_bit_exact(output_dir: Path, reference: Path) -> dict[str, Any]:
    """Compare concatenated independent payloads to the legacy FFN stream."""
    with reference.open("rb") as stream:
        raw_header = stream.read(COUNT_HEADER.size)
        if len(raw_header) != COUNT_HEADER.size:
            raise ValueError(f"{reference}: missing element-count header")
        reference_count = COUNT_HEADER.unpack(raw_header)[0]
        reference_payload = stream.read()

    payloads: list[bytes] = []
    total_count = 0
    for filename in _layer0_ffn_files():
        path = output_dir / filename
        with path.open("rb") as stream:
            raw_header = stream.read(COUNT_HEADER.size)
            if len(raw_header) != COUNT_HEADER.size:
                raise ValueError(f"{path}: missing element-count header")
            count = COUNT_HEADER.unpack(raw_header)[0]
            payload = stream.read()
        if len(payload) != count * 4:
            raise ValueError(f"{path}: payload has {len(payload)} bytes, expected {count * 4}")
        total_count += count
        payloads.append(payload)
    combined = b"".join(payloads)
    if total_count != reference_count or combined != reference_payload:
        first_difference = next(
            (index for index, (left, right) in enumerate(zip(combined, reference_payload)) if left != right),
            min(len(combined), len(reference_payload)),
        )
        raise AssertionError(
            "layer0 FFN export is not bit-exact with the legacy stream: "
            f"count={total_count}/{reference_count}, bytes={len(combined)}/{len(reference_payload)}, "
            f"first_differing_byte={first_difference}"
        )
    return {
        "status": "pass",
        "reference": str(reference),
        "reference_sha256": sha256_file(reference),
        "element_count": total_count,
        "payload_sha256": hashlib.sha256(combined).hexdigest(),
    }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_tasks(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return TASKS
    return (raw,)


def _schema_summary(specs: Sequence[TensorSpec]) -> dict[str, Any]:
    total_count = sum(spec.count for spec in specs)
    return {
        "format": MANIFEST_FORMAT,
        "fractional": FRACTIONAL,
        "tensor_file_count": len(specs),
        "directory_entry_count_with_manifest": len(specs) + 1,
        "total_parameter_count": total_count,
        "total_tensor_bytes_including_headers": total_count * 4 + len(specs) * COUNT_HEADER.size,
        "first_files": [asdict(spec) for spec in specs[:5]],
        "last_files": [asdict(spec) for spec in specs[-4:]],
    }


def build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parent / "weights"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--mrpc-model", default=DEFAULT_MRPC_MODEL)
    parser.add_argument("--rte-model", default=DEFAULT_RTE_MODEL)
    parser.add_argument("--tasks", choices=("both",) + TASKS, default="both")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", help="optional Hugging Face checkpoint revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force Transformers/Hugging Face offline mode and cached files only",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reference-layer0-ffn",
        type=Path,
        help="legacy combined FFN file whose payload must match bit-for-bit",
    )
    parser.add_argument("--skip-layer0-compat", action="store_true")
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="print the complete expected schema summary without importing ML dependencies",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = _parse_tasks(args.tasks)
    specs = build_specs(tasks)
    if args.print_schema:
        print(json.dumps(_schema_summary(specs), indent=2))
        return 0

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    managed_paths = [args.output_dir / spec.filename for spec in specs] + [manifest_path]
    existing = [path for path in managed_paths if path.exists()]
    if existing and not args.overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        suffix = " ..." if len(existing) > 3 else ""
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} managed path(s): {preview}{suffix}; "
            "pass --overwrite"
        )

    model_ids = {
        "encoder": args.encoder_model,
        "mrpc": args.mrpc_model,
        "rte": args.rte_model,
    }
    entries: list[dict[str, Any]] = []
    for source_role in ("encoder",) + tasks:
        role_specs = [spec for spec in specs if spec.source_role == source_role]
        group_entries = _export_source_group(
            source_role,
            model_ids[source_role],
            role_specs,
            args.output_dir,
            args,
        )
        # Ordinal is global inference/file order, not source-local order.
        entry_by_filename = {entry["filename"]: entry for entry in group_entries}
        entries.extend(entry_by_filename[spec.filename] for spec in role_specs)
    by_filename = {entry["filename"]: entry for entry in entries}
    entries = [by_filename[spec.filename] for spec in specs]
    for ordinal, entry in enumerate(entries):
        entry["ordinal"] = ordinal

    compatibility: dict[str, Any] = {"status": "not_checked"}
    reference = args.reference_layer0_ffn
    if reference is None and args.encoder_model == DEFAULT_ENCODER_MODEL:
        candidate = Path(__file__).resolve().parent / "models/pretrained/bert_base_uncased_layer0_ffn_q20.bin"
        if candidate.exists():
            reference = candidate
    if not args.skip_layer0_compat and reference is not None:
        compatibility = verify_layer0_ffn_bit_exact(args.output_dir, reference.expanduser().resolve())
        print(f"layer0 FFN legacy compatibility: {compatibility['status'].upper()}")

    total_count = sum(entry["count"] for entry in entries)
    manifest = {
        "format": MANIFEST_FORMAT,
        "tensor_format": FORMAT_NAME,
        "description": "BERT-base 12-layer inference parameters in PIGEON Q12.20 format",
        "fractional": FRACTIONAL,
        "scale": SCALE,
        "endianness": "little",
        "header": "int32_le element_count",
        "payload": "C-order contiguous int32_le Q12.20 values",
        "tasks": list(tasks),
        "models": {role: model_ids[role] for role in ("encoder",) + tasks},
        "bundle_semantics": "one shared encoder/pooler plus the listed task classifier heads",
        "task_accuracy_note": (
            "MRPC and RTE fine-tuned checkpoints have different encoder weights; for task-accurate "
            "evaluation export one task per directory with its task checkpoint also passed as "
            "--encoder-model"
        ),
        "revision": args.revision,
        "tensor_file_count": len(entries),
        "directory_entry_count_with_manifest": len(entries) + 1,
        "total_parameter_count": total_count,
        "total_tensor_bytes_including_headers": sum(entry["file_size_bytes"] for entry in entries),
        "layer0_ffn_legacy_compatibility": compatibility,
        "parameters": entries,
    }
    _atomic_write_json(manifest_path, manifest)
    print(
        f"PASS exported {len(entries)} tensor files, {total_count:,} parameters, "
        f"{manifest['total_tensor_bytes_including_headers'] / (1024 ** 2):.2f} MiB"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileExistsError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
