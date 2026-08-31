#!/usr/bin/env python3
"""Small parser/adversarial tests that require no encoder artifacts."""

from __future__ import annotations

import io
import struct
import unittest

import numpy as np

import outer_decode as dec


def pack_labels(values: np.ndarray) -> bytes:
    output = bytearray((values.size * 6 + 7) // 8)
    offset = 0
    for signed in values.tolist():
        value = int(signed) & 0x3F
        byte = offset >> 3
        shift = offset & 7
        word = value << shift
        output[byte] |= word & 0xFF
        if shift > 2:
            output[byte + 1] |= (word >> 8) & 0xFF
        offset += 6
    return bytes(output)


def one_chunk_side() -> bytes:
    block_count = 1
    labels = np.arange(-32, 32, dtype=np.int8).repeat(2)
    lut = np.exp2(np.arange(-32, 32, dtype=np.float64) / 16.0).astype("<f8")
    scales = np.asarray([0.25], dtype="<f4")
    header = dec.HEADER.pack(
        dec.MAGIC,
        dec.VERSION,
        block_count,
        dec.NORMATIVE_GROUPS_PER_CHUNK,
        dec.NORMATIVE_GROUP_VALUES,
        1,
        labels.size,
        1.0e-5,
        64,
    )
    profile = dec.PROFILE.pack(0.29, 0.5989929996555583, 0)
    return header + lut.tobytes() + scales.tobytes() + pack_labels(labels) + profile


class WireTests(unittest.TestCase):
    def test_side_exact_and_stable(self) -> None:
        side = dec.parse_side(one_chunk_side())
        self.assertEqual(side.chunk_count, 1)
        self.assertEqual(side.label_count, 128)
        self.assertTrue(np.array_equal(np.sort(side.stable_order), np.arange(128)))

    def test_side_rejects_trailing_byte(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected"):
            dec.parse_side(one_chunk_side() + b"\0")

    def test_container_padding_and_eof(self) -> None:
        good = struct.pack("<If", 1, 1.0) + b"\x80"
        frame = dec.read_container_frame(io.BytesIO(good), 0, 1 << 18)
        self.assertEqual(frame.logical_bits, 1)
        self.assertEqual(frame.arithmetic_padding_bits, 7)
        bad = struct.pack("<If", 1, 1.0) + b"\x81"
        with self.assertRaisesRegex(ValueError, "arithmetic padding"):
            dec.read_container_frame(io.BytesIO(bad), 0, 1 << 18)
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            dec.read_all_frames(io.BytesIO(good + b"x"), 1)

    def test_sparse_tail_padding(self) -> None:
        header_word = (1 << 20) | 0
        # One all-zero 34-bit record takes five bytes and has six zero pad bits.
        good = struct.pack("<If", header_word, 1.0) + b"\0" * 5
        frame = dec.read_container_frame(io.BytesIO(good), 0, 1 << 18)
        self.assertEqual(frame.tail_padding_bits, 6)
        bad = struct.pack("<If", header_word, 1.0) + b"\0\0\0\0\1"
        with self.assertRaisesRegex(ValueError, "sparse-tail padding"):
            dec.read_container_frame(io.BytesIO(bad), 0, 1 << 18)


if __name__ == "__main__":
    unittest.main()
