#!/usr/bin/env python3
"""Fetch one immutable 2^18-value BF16 block by safetensors HTTP range."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import urllib.request
from pathlib import Path


REPO = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
N = 1 << 18


def read_url(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "polar-lattice-ptq-block-audit/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        if byte_range is not None and getattr(response, "status", None) != 206:
            raise RuntimeError(f"range not honored: {response.status}")
        return response.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tensor")
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("qwen_weight_cache/range_blocks"))
    args = ap.parse_args()

    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    index = json.loads(read_url(base + "model.safetensors.index.json").decode())
    shard = index["weight_map"][args.tensor]
    shard_url = base + shard + "?download=true"
    prefix = read_url(shard_url, (0, 7))
    header_length = struct.unpack("<Q", prefix)[0]
    header = json.loads(
        read_url(shard_url, (8, 8 + header_length - 1)).decode().rstrip(" \t\r\n\0")
    )
    metadata = header[args.tensor]
    if metadata["dtype"] != "BF16":
        raise ValueError(metadata["dtype"])
    shape = [int(x) for x in metadata["shape"]]
    values = math.prod(shape)
    begin_value = args.block_index * N
    end_value = begin_value + N
    if end_value > values:
        raise ValueError((args.block_index, end_value, values))
    tensor_start = int(metadata["data_offsets"][0])
    absolute_start = 8 + header_length + tensor_start + 2 * begin_value
    absolute_end = absolute_start + 2 * N - 1
    raw = read_url(shard_url, (absolute_start, absolute_end))
    if len(raw) != 2 * N:
        raise RuntimeError((len(raw), 2 * N))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tensor)
    output = args.output_dir / f"{safe}.block{args.block_index}.bf16.bin"
    output.write_bytes(raw)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "tensor": args.tensor,
        "shape": shape,
        "dtype": "BF16",
        "block_index": args.block_index,
        "block_values": N,
        "shard": shard,
        "absolute_byte_range_in_shard": [absolute_start, absolute_end],
        "output": str(output),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
