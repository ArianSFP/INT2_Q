#!/usr/bin/env python3
"""Emit and round-trip the literal decoder side stream for waterfilled PLTE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


MAGIC = b"WFPLTE01"
VERSION = 1
HEADER = struct.Struct("<8sIIIIIIdI")
PROFILE = struct.Struct("<ddB")
LABEL_BITS = 6


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pack_labels(values: list[int]) -> bytes:
    output = bytearray((len(values) * LABEL_BITS + 7) // 8)
    bit_offset = 0
    for signed in values:
        if signed < -32 or signed > 31:
            raise ValueError(f"six-bit label out of range: {signed}")
        value = signed & 0x3F
        byte_index = bit_offset >> 3
        shift = bit_offset & 7
        word = value << shift
        output[byte_index] |= word & 0xFF
        if shift > 2:
            output[byte_index + 1] |= (word >> 8) & 0xFF
        bit_offset += LABEL_BITS
    return bytes(output)


def unpack_labels(payload: bytes, count: int) -> list[int]:
    values: list[int] = []
    bit_offset = 0
    for _ in range(count):
        byte_index = bit_offset >> 3
        shift = bit_offset & 7
        word = payload[byte_index]
        if byte_index + 1 < len(payload):
            word |= payload[byte_index + 1] << 8
        value = (word >> shift) & 0x3F
        values.append(value - 64 if value & 0x20 else value)
        bit_offset += LABEL_BITS
    used_bits = count * LABEL_BITS
    if used_bits % 8 and payload[-1] >> (used_bits % 8):
        raise AssertionError("nonzero label padding")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    blocks = manifest["blocks"]
    chunks = manifest["chunks"]
    group_values = int(manifest["parameters"]["group_values"])
    groups_per_block = int(manifest["parameters"]["groups_per_polar_block"])
    labels = [int(label) for block in blocks for label in block["labels_i8"]]
    scales = np.asarray(
        [float(block["serialized_rms_fp32"]) for block in blocks], dtype="<f4"
    )
    lut = np.exp2(np.arange(-32, 32, dtype=np.float64) / 16.0).astype("<f8")
    lambda_variance = float(manifest["parameters"]["lambda_variance"])
    header = HEADER.pack(
        MAGIC,
        VERSION,
        len(blocks),
        groups_per_block,
        group_values,
        len(chunks),
        len(labels),
        lambda_variance,
        len(lut),
    )
    profile_blob = bytearray()
    for chunk in chunks:
        alphabet_size = int(chunk.get("alphabet_size", 128 if float(chunk["nominal_rate"]) >= 2.75 else 64))
        alphabet_code = {64: 0, 128: 1, 256: 2}.get(alphabet_size)
        if alphabet_code is None:
            raise ValueError(f"unsupported alphabet size {alphabet_size}")
        profile_blob.extend(
            PROFILE.pack(float(chunk["test_distortion"]), float(chunk["eta"]), alphabet_code)
        )
    blob = header + lut.tobytes() + scales.tobytes() + pack_labels(labels) + bytes(profile_blob)
    args.output.write_bytes(blob)

    # Independent parse and exact semantic reconstruction.
    offset = 0
    unpacked = HEADER.unpack_from(blob, offset)
    offset += HEADER.size
    magic, version, block_count, gpb, gv, chunk_count, label_count, decoded_lambda, lut_count = unpacked
    if (magic, version, block_count, gpb, gv, chunk_count, label_count, lut_count) != (
        MAGIC,
        VERSION,
        len(blocks),
        groups_per_block,
        group_values,
        len(chunks),
        len(labels),
        64,
    ):
        raise AssertionError("side header round-trip mismatch")
    decoded_lut = np.frombuffer(blob, dtype="<f8", count=lut_count, offset=offset).copy()
    offset += lut_count * 8
    decoded_scales = np.frombuffer(blob, dtype="<f4", count=block_count, offset=offset).copy()
    offset += block_count * 4
    label_bytes = (label_count * LABEL_BITS + 7) // 8
    decoded_labels = unpack_labels(blob[offset : offset + label_bytes], label_count)
    offset += label_bytes
    decoded_profiles = []
    for _ in range(chunk_count):
        d, eta, code = PROFILE.unpack_from(blob, offset)
        offset += PROFILE.size
        decoded_profiles.append((d, eta, {0: 64, 1: 128, 2: 256}[code]))
    if offset != len(blob):
        raise AssertionError("side parser did not consume exact EOF")
    if decoded_lambda != lambda_variance or not np.array_equal(decoded_lut, lut):
        raise AssertionError("lambda/LUT mismatch")
    if not np.array_equal(decoded_scales, scales) or decoded_labels != labels:
        raise AssertionError("scale/label round-trip mismatch")

    qscale = np.empty(label_count, dtype=np.float64)
    for index, label in enumerate(decoded_labels):
        block_index = index // groups_per_block
        qscale[index] = float(decoded_scales[block_index]) * float(decoded_lut[label + 32])
    order = np.lexsort((np.arange(label_count, dtype=np.int64), np.square(qscale)))
    expected_order = np.asarray(
        [int(member["canonical_group_ordinal"]) for chunk in chunks for member in chunk["members"]],
        dtype=np.int64,
    )
    if not np.array_equal(order, expected_order):
        raise AssertionError("side-only stable membership reconstruction mismatch")
    for expected, decoded in zip(chunks, decoded_profiles, strict=True):
        expected_alphabet = int(expected.get("alphabet_size", 128 if float(expected["nominal_rate"]) >= 2.75 else 64))
        if struct.pack("<d", float(expected["test_distortion"])) != struct.pack("<d", decoded[0]):
            raise AssertionError("D profile mismatch")
        if struct.pack("<d", float(expected["eta"])) != struct.pack("<d", decoded[1]):
            raise AssertionError("eta profile mismatch")
        if decoded[2] != expected_alphabet:
            raise AssertionError("alphabet profile mismatch")

    panel_values = len(blocks) * groups_per_block * group_values
    receipt = {
        "format": "continuous reverse-waterfilled PLTE side receipt v1",
        "status": "exact round-trip passed",
        "side_path": str(args.output),
        "side_bytes": len(blob),
        "side_bits": len(blob) * 8,
        "side_bpw_over_panel": len(blob) * 8 / panel_values,
        "side_sha256": sha256(blob),
        "header_bytes": HEADER.size,
        "exp2_lut_bytes": len(lut) * 8,
        "block_scale_bytes": len(scales) * 4,
        "packed_label_bytes": label_bytes,
        "profile_bytes": len(profile_blob),
        "stable_membership_reconstructed_from_side_only": True,
        "exact_eof": True,
        "profile_binary64_roundtrip": True,
        "note": "selection/tensor geometry and the pinned polar mask are global decoder assets, not encoded in this panel side stream",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
