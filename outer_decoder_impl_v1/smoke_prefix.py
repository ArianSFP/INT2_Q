#!/usr/bin/env python3
"""Non-claim prefix smoke while the full 400-container run is still encoding."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import lzma
from pathlib import Path

import outer_decode as dec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--container-dir", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--clean-decoder", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    side = dec.read_side_file(args.side)
    if (
        not 1 <= args.chunks <= side.chunk_count
        or args.start_chunk < 0
        or args.start_chunk + args.chunks > side.chunk_count
    ):
        raise ValueError("invalid smoke prefix length")
    raw_mask = args.raw_mask.read_bytes()
    compressed_side = lzma.compress(side.blob, format=lzma.FORMAT_XZ, preset=9)
    compressed_mask = bz2.compress(raw_mask, compresslevel=9)
    header = dec.OUTER_HEADER.pack(
        dec.OUTER_MAGIC,
        dec.OUTER_VERSION,
        dec.OUTER_HEADER.size,
        dec.SIDE_CODEC_LZMA_XZ,
        len(side.blob),
        len(compressed_side),
        dec.MASK_CODEC_BZ2,
        len(raw_mask),
        len(compressed_mask),
        hashlib.sha256(side.blob).digest(),
        hashlib.sha256(compressed_side).digest(),
        hashlib.sha256(raw_mask).digest(),
        hashlib.sha256(compressed_mask).digest(),
    )
    literal_frames = []
    indices = range(args.start_chunk, args.start_chunk + args.chunks)
    for index in indices:
        path = args.container_dir / f"wf-{index:03d}.polar.bin"
        with path.open("rb") as handle:
            frame = dec.read_container_frame(handle, index, dec.NORMATIVE_BLOCK_LENGTH)
            if handle.read(1):
                raise ValueError(f"individual smoke container {path} has trailing bytes")
        literal_frames.append(frame)
    prefix = header + compressed_side + compressed_mask + b"".join(
        frame.literal for frame in literal_frames
    )
    handle = io.BytesIO(prefix)
    prelude = dec.read_bundle_prelude(handle)
    frames = tuple(
        dec.read_container_frame(handle, index, dec.NORMATIVE_BLOCK_LENGTH)
        for index in indices
    )
    if handle.read(1):
        raise AssertionError("prefix smoke did not consume its physical EOF")
    if prelude.side.blob != side.blob or prelude.raw_mask != raw_mask:
        raise AssertionError("prefix prelude round trip failed")
    rows = []
    tasks = list(
        zip(
            side.profiles[args.start_chunk : args.start_chunk + args.chunks],
            frames,
            strict=True,
        )
    )
    for result in dec.decode_tasks(
        tasks, args.clean_decoder, prelude.raw_mask, args.workers
    ):
        reconstruction = result.pop("reconstruction")
        result["normalized_reconstruction_f64_sha256"] = hashlib.sha256(
            reconstruction.astype("<f8", copy=False).tobytes()
        ).hexdigest()
        rows.append(result)
    output = {
        "format": "non-claim outer bundle prefix smoke v1",
        "status": "passed",
        "claim_grade": False,
        "reason_nonclaim": "physical prefix intentionally omits side-declared trailing containers",
        "chunks": args.chunks,
        "start_chunk": args.start_chunk,
        "workers": args.workers,
        "prefix_bytes": len(prefix),
        "header_bytes": len(header),
        "compressed_side_bytes": len(compressed_side),
        "compressed_mask_bytes": len(compressed_mask),
        "raw_mask_sha256": hashlib.sha256(raw_mask).hexdigest(),
        "rows": rows,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
