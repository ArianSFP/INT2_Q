#!/usr/bin/env python3
"""Pack isolated WFOUTR fixed-route side-codec experiment v2."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
from pathlib import Path

import fixed_route_codec as route_codec


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = Path(str(path) + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--container-dir", type=Path, required=True)
    parser.add_argument("--container-pattern", default="wf-{index:03d}.polar.bin")
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = route_codec.load_v1_outer()
    dependency_bindings = route_codec.dependency_bindings(
        base,
        "fixed_route_packer_wrapper_sha256",
        Path(__file__),
    )
    side = base.read_side_file(args.side)
    encoded_side = route_codec.encode_side_payload(base, side.blob)
    mask = args.raw_mask.read_bytes()
    expected_mask_bytes = base.BASE_MASK_LEVELS * (
        (base.NORMATIVE_BLOCK_LENGTH + 7) // 8
    )
    if len(mask) != expected_mask_bytes:
        raise ValueError(f"raw mask is {len(mask)} bytes, expected {expected_mask_bytes}")
    compressed_mask = bz2.compress(mask, compresslevel=9)
    if base.decompress_bz2_exact(compressed_mask, len(mask)) != mask:
        raise AssertionError("BZ2 mask round trip failed")

    frames = []
    for index in range(side.chunk_count):
        path = args.container_dir / args.container_pattern.format(index=index)
        with path.open("rb") as handle:
            frame = base.read_container_frame(
                handle, index, base.NORMATIVE_BLOCK_LENGTH
            )
            if handle.read(1):
                raise ValueError(f"individual container {path} has trailing bytes")
        frames.append(frame)

    payload = encoded_side["payload"]
    header = base.OUTER_HEADER.pack(
        base.OUTER_MAGIC,
        base.OUTER_VERSION,
        base.OUTER_HEADER.size,
        route_codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400,
        len(side.blob),
        len(payload),
        base.MASK_CODEC_BZ2,
        len(mask),
        len(compressed_mask),
        hashlib.sha256(side.blob).digest(),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(mask).digest(),
        hashlib.sha256(compressed_mask).digest(),
    )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    temporary = Path(str(args.output) + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        handle.write(header)
        handle.write(payload)
        handle.write(compressed_mask)
        for frame in frames:
            handle.write(frame.literal)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)

    with args.output.open("rb") as handle:
        checked = route_codec.read_bundle_prelude_v2(base, handle)
        checked_frames = base.read_all_frames(handle, checked.side.chunk_count)
    if checked.side.blob != side.blob or checked.raw_mask != mask:
        raise AssertionError("source-free fixed-route prelude mismatch")
    if [frame.sha256 for frame in checked_frames] != [frame.sha256 for frame in frames]:
        raise AssertionError("post-write container sequence mismatch")
    container_bytes = sum(len(frame.literal) for frame in frames)
    bundle_bytes = args.output.stat().st_size
    prelude_bytes = len(header) + len(payload) + len(compressed_mask)
    if bundle_bytes != prelude_bytes + container_bytes:
        raise AssertionError("physical bundle partition mismatch")
    panel_values = side.label_count * side.group_values
    receipt = {
        "format": "continuous PLTE WFOUTR fixed-route bundle experiment v2",
        "status": "passed",
        "experimental_not_v1": True,
        "dependency_bindings": dependency_bindings,
        "source_free_reparse_passed": True,
        "bundle_path": str(args.output),
        "bundle_bytes": bundle_bytes,
        "bundle_sha256": sha256_path(args.output),
        "physical_all_in_bpw": bundle_bytes * 8.0 / panel_values,
        "panel_values": panel_values,
        "header_bytes": len(header),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "physical_prelude_bytes": prelude_bytes,
        "side": {
            "codec_id": route_codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400,
            "codec": "XZ(canonical all-A64 WFPLTE01) + fixed LSB-first route400",
            "raw_bytes": len(side.blob),
            "raw_sha256": hashlib.sha256(side.blob).hexdigest(),
            "canonical_raw_bytes": len(encoded_side["canonical"]),
            "canonical_raw_sha256": hashlib.sha256(encoded_side["canonical"]).hexdigest(),
            "canonical_xz_bytes": len(encoded_side["canonical_xz"]),
            "canonical_xz_sha256": hashlib.sha256(encoded_side["canonical_xz"]).hexdigest(),
            "route_bits": route_codec.ROUTE_BITS,
            "route_bytes": len(encoded_side["route"]),
            "route_sha256": hashlib.sha256(encoded_side["route"]).hexdigest(),
            "compressed_bytes": len(payload),
            "compressed_sha256": hashlib.sha256(payload).hexdigest(),
            "literal_side_reconstructed_hash_verified": True,
            "canonical_profile_offsets_verified": True,
            "alphabet_domain": [64, 128],
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
    print(json.dumps({key: value for key, value in receipt.items() if key != "containers"}, indent=2))


if __name__ == "__main__":
    main()
