#!/usr/bin/env python3
"""Assemble the self-contained, physically charged WFOUTR01 bundle."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import os
from pathlib import Path

import outer_decode as dec


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = Path(str(path) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    containers = parser.add_mutually_exclusive_group(required=True)
    containers.add_argument("--containers", type=Path)
    containers.add_argument("--container-dir", type=Path)
    parser.add_argument(
        "--container-pattern",
        default="wf-{index:03d}.polar.bin",
        help="Python format string used with --container-dir",
    )
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    side = dec.read_side_file(args.side)
    mask = args.raw_mask.read_bytes()
    expected_mask_bytes = dec.BASE_MASK_LEVELS * (
        (dec.NORMATIVE_BLOCK_LENGTH + 7) // 8
    )
    if len(mask) != expected_mask_bytes:
        raise ValueError(
            f"raw mask contains {len(mask)} bytes, expected {expected_mask_bytes}"
        )
    compressed_side = lzma.compress(
        side.blob, format=lzma.FORMAT_XZ, preset=9
    )
    compressed_mask = bz2.compress(mask, compresslevel=9)
    # Test both compressed members with the exact bounded decoders used by the
    # independent consumer before writing anything.
    if dec.decompress_lzma_xz_exact(compressed_side, len(side.blob)) != side.blob:
        raise AssertionError("LZMA side round trip mismatch")
    if dec.decompress_bz2_exact(compressed_mask, len(mask)) != mask:
        raise AssertionError("BZ2 mask round trip mismatch")

    if args.containers is not None:
        with args.containers.open("rb") as handle:
            frames = dec.read_all_frames(handle, side.chunk_count)
    else:
        parsed = []
        for index in range(side.chunk_count):
            path = args.container_dir / args.container_pattern.format(index=index)
            with path.open("rb") as handle:
                frame = dec.read_container_frame(
                    handle, index, dec.NORMATIVE_BLOCK_LENGTH
                )
                if handle.read(1):
                    raise ValueError(f"individual container {path} has trailing bytes")
            parsed.append(frame)
        frames = tuple(parsed)

    header = dec.OUTER_HEADER.pack(
        dec.OUTER_MAGIC,
        dec.OUTER_VERSION,
        dec.OUTER_HEADER.size,
        dec.SIDE_CODEC_LZMA_XZ,
        len(side.blob),
        len(compressed_side),
        dec.MASK_CODEC_BZ2,
        len(mask),
        len(compressed_mask),
        hashlib.sha256(side.blob).digest(),
        hashlib.sha256(compressed_side).digest(),
        hashlib.sha256(mask).digest(),
        hashlib.sha256(compressed_mask).digest(),
    )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    temporary = Path(str(args.output) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        handle.write(header)
        handle.write(compressed_side)
        handle.write(compressed_mask)
        for frame in frames:
            handle.write(frame.literal)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)

    # Source-free independent reparse, including both decompressor EOFs and
    # physical EOF after the side-declared chunk count.
    with args.output.open("rb") as handle:
        check = dec.read_bundle_prelude(handle)
        checked_frames = dec.read_all_frames(handle, check.side.chunk_count)
    if check.side.blob != side.blob or check.raw_mask != mask:
        raise AssertionError("post-write bundle prelude mismatch")
    if tuple(frame.sha256 for frame in checked_frames) != tuple(
        frame.sha256 for frame in frames
    ):
        raise AssertionError("post-write container sequence mismatch")
    panel_values = side.label_count * side.group_values
    container_bytes = sum(len(frame.literal) for frame in frames)
    output_bytes = args.output.stat().st_size
    expected_bytes = (
        len(header) + len(compressed_side) + len(compressed_mask) + container_bytes
    )
    if output_bytes != expected_bytes:
        raise AssertionError("physical bundle size mismatch")
    receipt = {
        "format": "continuous reverse-waterfilled PLTE self-contained bundle pack v1",
        "status": "passed",
        "source_free_reparse_passed": True,
        "bundle_path": str(args.output),
        "bundle_bytes": output_bytes,
        "bundle_sha256": sha256_path(args.output),
        "physical_all_in_bpw": output_bytes * 8.0 / panel_values,
        "panel_values": panel_values,
        "header_bytes": len(header),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "side": {
            "codec": "LZMA-XZ preset 9",
            "raw_bytes": len(side.blob),
            "compressed_bytes": len(compressed_side),
            "raw_sha256": hashlib.sha256(side.blob).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed_side).hexdigest(),
        },
        "mask": {
            "codec": "BZ2 level 9",
            "raw_bytes": len(mask),
            "compressed_bytes": len(compressed_mask),
            "raw_sha256": hashlib.sha256(mask).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed_mask).hexdigest(),
        },
        "containers": {
            "count": len(frames),
            "bytes": container_bytes,
            "ordered_sha256": [frame.sha256 for frame in frames],
            "all_arithmetic_padding_zero": True,
            "all_sparse_tail_padding_zero": True,
            "exact_eof": True,
        },
    }
    atomic_json(args.receipt, receipt, args.overwrite)
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "containers"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
