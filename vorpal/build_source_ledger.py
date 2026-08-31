#!/usr/bin/env python3
"""Build the exact evaluator-only canonical BF16 source ledger for VORPAL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_BLOCKS = 400
EXPECTED_VALUES = 1 << 18


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    construction = load_json(args.construction_manifest)
    selection = load_json(args.selection_manifest)
    blocks = construction.get("blocks")
    selected = selection.get("plte_blocks")
    if not isinstance(blocks, list) or len(blocks) != EXPECTED_BLOCKS:
        raise ValueError("construction manifest must contain exactly 400 blocks")
    if not isinstance(selected, list) or len(selected) != EXPECTED_BLOCKS:
        raise ValueError("selection manifest must contain exactly 400 PLTE blocks")
    selected_by_id = {str(row["id"]): row for row in selected}
    if len(selected_by_id) != EXPECTED_BLOCKS:
        raise ValueError("selection block IDs are not unique")
    checkpoint = selection.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("repo") or not checkpoint.get("revision"):
        raise ValueError("selection manifest omits immutable checkpoint identity")

    ledger_rows = []
    seen_ids: set[str] = set()
    for ordinal, block in enumerate(blocks):
        if int(block.get("ordinal", -1)) != ordinal:
            raise ValueError(f"construction block {ordinal} has a noncanonical ordinal")
        block_id = str(block["id"])
        if block_id in seen_ids or block_id not in selected_by_id:
            raise ValueError(f"unknown or repeated construction block ID {block_id!r}")
        seen_ids.add(block_id)
        source_meta = selected_by_id[block_id]
        for field in ("tensor", "role", "block_index"):
            if block[field] != source_meta[field]:
                raise ValueError(f"{block_id}: construction/selection {field} mismatch")
        if int(source_meta.get("source_values", -1)) != EXPECTED_VALUES:
            raise ValueError(f"{block_id}: unexpected source value count")
        source_value = str(block["source_path"])
        source_path = Path(source_value)
        resolved = source_path if source_path.is_absolute() else args.source_root / source_path
        if not resolved.is_file() or resolved.stat().st_size != EXPECTED_VALUES * 2:
            raise ValueError(f"{block_id}: missing or malformed BF16 source {resolved}")
        actual_hash = sha256_path(resolved)
        expected_hash = str(block["source_sha256"]).lower()
        if actual_hash != expected_hash:
            raise ValueError(f"{block_id}: BF16 source SHA256 mismatch")
        ledger_rows.append(
            {
                "canonical_block_ordinal": ordinal,
                "id": block_id,
                "tensor": str(source_meta["tensor"]),
                "role": str(source_meta["role"]),
                "layer": source_meta.get("layer"),
                "block_index": int(source_meta["block_index"]),
                "path": source_value,
                "sha256": actual_hash,
            }
        )
    if seen_ids != set(selected_by_id):
        raise AssertionError("construction and selection block sets differ")

    result = {
        "format": "canonical BF16 source ledger v1",
        "evaluator_only": True,
        "checkpoint": {
            "repo": str(checkpoint["repo"]),
            "revision": str(checkpoint["revision"]),
        },
        "selection_manifest_sha256": sha256_path(args.selection_manifest),
        "construction_manifest_sha256": sha256_path(args.construction_manifest),
        "blocks": ledger_rows,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "blocks": len(ledger_rows),
                "checkpoint": result["checkpoint"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
