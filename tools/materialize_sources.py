#!/usr/bin/env python3
"""Fetch the immutable Qwen BF16 blocks required by the published PLTE ledger.

The raw source bytes are intentionally gitignored. Downloads are pinned to the
checkpoint revision hard-coded in the evidenced fetcher. Use --include-routers
to additionally fetch all 48 router blocks for router-codec reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
FETCHER = PLTE / "agent_root_fetch_qwen_block.py"
BLOCK_BYTES = 2 * (1 << 18)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetched_name(tensor: str, block_index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tensor)
    return f"{safe}.block{block_index}.bf16.bin"


def fetch_block(tensor: str, block_index: int, temporary: Path) -> bytes:
    subprocess.run(
        [
            sys.executable,
            str(FETCHER),
            tensor,
            "--block-index",
            str(block_index),
            "--output-dir",
            str(temporary),
        ],
        check=True,
    )
    payload = (temporary / fetched_name(tensor, block_index)).read_bytes()
    if len(payload) != BLOCK_BYTES:
        raise AssertionError(f"short source block for {tensor} block {block_index}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-routers", action="store_true")
    args = parser.parse_args()

    evidence = json.loads(
        (PLTE / "agent_root_polar_escape_evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if evidence["checkpoint"] != {
        "repo": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    }:
        raise AssertionError("unexpected checkpoint provenance")

    entries_by_source: dict[Path, list[dict]] = {}
    for entry in evidence["reports"]:
        source = PLTE / entry["source"]
        entries_by_source.setdefault(source, []).append(entry)

    downloaded: dict[tuple[str, int], bytes] = {}
    with tempfile.TemporaryDirectory(prefix="plte-source-fetch-") as directory:
        temporary = Path(directory)
        for entries in entries_by_source.values():
            for entry in entries:
                key = (entry["canonical_tensor"], int(entry["canonical_block_index"]))
                if key not in downloaded:
                    downloaded[key] = fetch_block(*key, temporary)

        for source, entries in entries_by_source.items():
            unique_local = {
                int(entry["source_local_block_index"]): entry for entry in entries
            }
            max_index = max(unique_local)
            payload = bytearray((max_index + 1) * BLOCK_BYTES)
            for local_index, entry in unique_local.items():
                key = (entry["canonical_tensor"], int(entry["canonical_block_index"]))
                block = downloaded[key]
                expected = entry["source_block_sha256"]
                if sha256(block) != expected:
                    raise AssertionError(
                        f"upstream block hash mismatch for {key}: {sha256(block)} != {expected}"
                    )
                start = local_index * BLOCK_BYTES
                payload[start : start + BLOCK_BYTES] = block
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)

        if args.include_routers:
            router_dir = PLTE / "qwen_weight_cache" / "range_blocks"
            router_dir.mkdir(parents=True, exist_ok=True)
            for layer in range(48):
                tensor = f"model.layers.{layer}.mlp.gate.weight"
                block = fetch_block(tensor, 0, temporary)
                target = router_dir / fetched_name(tensor, 0)
                target.write_bytes(block)

    # Re-read every evidence slice from its final on-disk path.
    for source, entries in entries_by_source.items():
        payload = source.read_bytes()
        for entry in entries:
            local_index = int(entry["source_local_block_index"])
            start = local_index * BLOCK_BYTES
            block = payload[start : start + BLOCK_BYTES]
            if sha256(block) != entry["source_block_sha256"]:
                raise AssertionError(f"materialized source verification failed: {source}")

    print(
        json.dumps(
            {
                "checkpoint": evidence["checkpoint"],
                "evidence_source_files": len(entries_by_source),
                "unique_blocks_downloaded": len(downloaded),
                "router_blocks_downloaded": 48 if args.include_routers else 0,
                "raw_weight_bytes_are_gitignored": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
