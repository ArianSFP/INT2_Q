#!/usr/bin/env python3
"""Source-free one-chunk branch test for A128 or embedded sparse tails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import outer_decode as dec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--clean-decoder", type=Path, required=True)
    parser.add_argument("--expected-frequency-sha256")
    parser.add_argument("--expected-indices-sha256")
    args = parser.parse_args()
    side = dec.read_side_file(args.side)
    index = args.chunk_index
    if not 0 <= index < side.chunk_count:
        raise ValueError("chunk index is out of range")
    with args.container.open("rb") as handle:
        frame = dec.read_container_frame(handle, index, dec.NORMATIVE_BLOCK_LENGTH)
        if handle.read(1):
            raise ValueError("branch-test container has trailing bytes")
    raw_mask = args.raw_mask.read_bytes()
    result = next(
        dec.decode_tasks(
            [(side.profiles[index], frame)],
            args.clean_decoder,
            raw_mask,
            workers=1,
        )
    )
    normalized = np.asarray(result.pop("reconstruction"), dtype=np.float64)
    begin = index * side.groups_per_chunk
    end = begin + side.groups_per_chunk
    members = side.stable_order[begin:end]
    if members.size != side.groups_per_chunk or np.unique(members).size != members.size:
        raise AssertionError("branch-test stable membership is incomplete or repeated")
    scattered = normalized.reshape(side.groups_per_chunk, side.group_values).copy()
    scattered *= side.qscales[members, None]
    if (
        args.expected_frequency_sha256
        and result["frequency_u16_sha256"] != args.expected_frequency_sha256
    ):
        raise AssertionError(
            "frequency hash disagrees with prior exact clean decode: "
            f"{result['frequency_u16_sha256']} != {args.expected_frequency_sha256}"
        )
    if (
        args.expected_indices_sha256
        and result["reconstruction_indices_i16_sha256"]
        != args.expected_indices_sha256
    ):
        raise AssertionError(
            "index hash disagrees with prior exact clean decode: "
            f"{result['reconstruction_indices_i16_sha256']} != {args.expected_indices_sha256}"
        )
    output = {
        "format": "source-free outer decoder branch smoke v1",
        "status": "passed",
        "chunk_index": index,
        "container_bytes": len(frame.literal),
        "container_sha256": frame.sha256,
        "logical_bits": frame.logical_bits,
        "arithmetic_padding_bits": frame.arithmetic_padding_bits,
        "tail_padding_bits": frame.tail_padding_bits,
        "alphabet_size_from_literal_side": side.profiles[index].alphabet_size,
        "test_distortion_binary64": side.profiles[index].distortion,
        "eta_binary64": side.profiles[index].eta,
        "escape_count": int(frame.escape_positions.size),
        "escape_positions": frame.escape_positions.astype(int).tolist(),
        "escape_values_u16": frame.escape_values_u16.astype(int).tolist(),
        "escape_positions_i32_sha256": hashlib.sha256(
            frame.escape_positions.astype("<i4", copy=False).tobytes()
        ).hexdigest(),
        "escape_values_u16_sha256": hashlib.sha256(
            frame.escape_values_u16.astype("<u2", copy=False).tobytes()
        ).hexdigest(),
        "stable_member_ordinals_i64_sha256": hashlib.sha256(
            members.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "normalized_reconstruction_f64_sha256": hashlib.sha256(
            normalized.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "scattered_raw_reconstruction_f64_sha256": hashlib.sha256(
            scattered.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "prior_frequency_hash_asserted": bool(args.expected_frequency_sha256),
        "prior_indices_hash_asserted": bool(args.expected_indices_sha256),
        **result,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
