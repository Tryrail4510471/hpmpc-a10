#!/usr/bin/env python3
"""Generate a deterministic full-BERT reference for the FID=203 harness.

The public token/type/mask files use a four-byte little-endian element-count
header.  Token IDs are uint32; the key-padding mask is uint8 with 1=keep and
0=padding.  Native HuggingFace eval-mode logits, the pooler result, embeddings,
and all twelve encoder boundaries are saved for progressive MPC diagnosis.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Sequence


COUNT = struct.Struct("<I")
MODELS = {
    "mrpc": "textattack/bert-base-uncased-MRPC",
    "rte": "textattack/bert-base-uncased-RTE",
}
DEFAULT_TEXT = {
    "mrpc": (
        "The company released its quarterly results on Tuesday.",
        "On Tuesday, the firm published its quarterly financial results.",
    ),
    "rte": (
        "A soccer player is running across a field while spectators watch.",
        "An athlete is outdoors.",
    ),
}


def _write_counted(path: Path, values: Any, dtype: str) -> None:
    import numpy as np

    payload = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COUNT.pack(int(payload.size)))
        stream.write(payload.tobytes(order="C"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one full BERT classification reference.")
    parser.add_argument("--task", choices=tuple(MODELS), default="mrpc")
    parser.add_argument("--model")
    parser.add_argument("--text-a")
    parser.add_argument("--text-b")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.sequence_length <= 128:
        raise SystemExit("--sequence-length must be in [1,128]")

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = args.model or MODELS[args.task]
    text_a, text_b = DEFAULT_TEXT[args.task]
    text_a = args.text_a or text_a
    text_b = args.text_b or text_b
    output_dir = args.output_dir or (
        Path(__file__).resolve().parents[2] / "tests/data/bert_base_reference" / args.task
    )
    load_kwargs: dict[str, Any] = {"local_files_only": bool(args.offline)}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision is not None:
        load_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, attn_implementation="eager", **load_kwargs
    ).eval()
    encoded = tokenizer(
        text_a,
        text_b,
        max_length=args.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(torch.int64)
    attention_mask = encoded["attention_mask"].to(torch.uint8)
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(token_ids)).to(torch.int64)

    with torch.inference_mode():
        bert_output = model.bert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = model.classifier(model.dropout(bert_output.pooler_output))
        embeddings = model.bert.embeddings(
            input_ids=token_ids,
            token_type_ids=token_type_ids,
        )

    hidden_states = tuple(value.detach().cpu().numpy().astype(np.float32) for value in bert_output.hidden_states)
    if len(hidden_states) != 13:
        raise AssertionError(f"expected embeddings plus 12 layer outputs, got {len(hidden_states)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_counted(output_dir / "token_ids_u32.bin", token_ids.cpu().numpy(), "<u4")
    _write_counted(output_dir / "token_type_ids_u32.bin", token_type_ids.cpu().numpy(), "<u4")
    _write_counted(output_dir / "attention_mask_u8.bin", attention_mask.cpu().numpy(), "u1")
    np.save(output_dir / "logits.npy", logits.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    np.save(
        output_dir / "pooler_output.npy",
        bert_output.pooler_output.detach().cpu().numpy().astype(np.float32),
        allow_pickle=False,
    )
    np.save(output_dir / "embeddings.npy", embeddings.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    np.savez(output_dir / "encoder_boundaries.npz", **{
        f"boundary_{index:02d}": value for index, value in enumerate(hidden_states)
    })

    metadata = {
        "task": args.task,
        "model": model_name,
        "revision": args.revision,
        "text_a": text_a,
        "text_b": text_b,
        "batch": 1,
        "sequence_length": args.sequence_length,
        "valid_tokens": int(attention_mask.sum().item()),
        "logits": logits.detach().cpu().tolist(),
        "prediction": int(logits.argmax(dim=-1).item()),
        "layernorm_eps": float(model.config.layer_norm_eps),
        "files": {
            "token_ids": "token_ids_u32.bin",
            "token_type_ids": "token_type_ids_u32.bin",
            "attention_mask": "attention_mask_u8.bin",
            "logits": "logits.npy",
            "pooler_output": "pooler_output.npy",
            "embeddings": "embeddings.npy",
            "encoder_boundaries": "encoder_boundaries.npz",
        },
    }
    (output_dir / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
