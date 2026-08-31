#!/usr/bin/env python3
"""Adversarial tests for isolated fixed-route side codec v2."""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import math
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fixed_route_codec as codec
import evaluate_sources_fixed_route_v2 as fixed_evaluator
import select_continuous_adaptive_fixed_route_v2 as fixed_selector


BASE = codec.load_v1_outer()


def synthetic_side(codes: list[int] | None = None) -> bytes:
    codes = codes if codes is not None else [index & 1 for index in range(400)]
    if len(codes) != 400:
        raise ValueError("test side requires 400 codes")
    block_count = 400
    groups = BASE.NORMATIVE_GROUPS_PER_CHUNK
    label_count = block_count * groups
    lut = np.exp2(np.arange(-32, 32, dtype=np.float64) / 16.0).astype("<f8")
    scales = np.ones(block_count, dtype="<f4")
    labels = bytes((label_count * BASE.LABEL_BITS + 7) // 8)
    profiles = b"".join(BASE.PROFILE.pack(1.0, 1.0, code) for code in codes)
    header = BASE.HEADER.pack(
        BASE.MAGIC,
        BASE.VERSION,
        block_count,
        groups,
        BASE.NORMATIVE_GROUP_VALUES,
        400,
        label_count,
        1.0,
        len(lut),
    )
    blob = header + lut.tobytes() + scales.tobytes() + labels + profiles
    BASE.parse_side(blob)
    return blob


def bundle_prefix(literal: bytes) -> tuple[bytes, bytes]:
    encoded = codec.encode_side_payload(BASE, literal)
    mask = bytes(BASE.BASE_MASK_LEVELS * ((BASE.NORMATIVE_BLOCK_LENGTH + 7) // 8))
    cmask = bz2.compress(mask, compresslevel=9)
    payload = encoded["payload"]
    header = BASE.OUTER_HEADER.pack(
        BASE.OUTER_MAGIC,
        BASE.OUTER_VERSION,
        BASE.OUTER_HEADER.size,
        codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400,
        len(literal),
        len(payload),
        BASE.MASK_CODEC_BZ2,
        len(mask),
        len(cmask),
        hashlib.sha256(literal).digest(),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(mask).digest(),
        hashlib.sha256(cmask).digest(),
    )
    return header + payload + cmask, mask


class FixedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.literal = synthetic_side()
        cls.encoded = codec.encode_side_payload(BASE, cls.literal)

    def test_exact_roundtrip_and_all_a64_canonical(self) -> None:
        canonical = self.encoded["canonical"]
        offsets = codec.profile_code_offsets(BASE, canonical)
        self.assertEqual(len(offsets), 400)
        self.assertTrue(all(canonical[offset] == 0 for offset in offsets))
        self.assertEqual(len(self.encoded["route"]), 50)
        self.assertEqual(
            codec.reconstruct_literal_side(BASE, canonical, self.encoded["route"]),
            self.literal,
        )

    def test_rejects_a256_literal(self) -> None:
        codes = [0] * 400
        codes[137] = 2
        with self.assertRaises(ValueError):
            codec.route_from_literal_side(BASE, synthetic_side(codes))

    def test_route_requires_exact_400_bits(self) -> None:
        canonical = self.encoded["canonical"]
        with self.assertRaises(ValueError):
            codec.reconstruct_literal_side(BASE, canonical, bytes(49))
        with self.assertRaises(ValueError):
            codec.reconstruct_literal_side(BASE, canonical, bytes(51))

    def test_canonical_profile_must_be_a64(self) -> None:
        corrupt = bytearray(self.encoded["canonical"])
        corrupt[codec.profile_code_offsets(BASE, corrupt)[9]] = 1
        with self.assertRaises(ValueError):
            codec.reconstruct_literal_side(BASE, bytes(corrupt), self.encoded["route"])

    def test_reconstructed_raw_hash_is_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            codec.decode_side_payload(
                BASE,
                self.encoded["payload"],
                len(self.literal),
                bytes(32),
            )

    def test_xz_trailing_byte_is_rejected_before_route(self) -> None:
        payload = (
            self.encoded["canonical_xz"]
            + b"\0"
            + self.encoded["route"]
        )
        with self.assertRaises(ValueError):
            codec.decode_side_payload(
                BASE,
                payload,
                len(self.literal),
                hashlib.sha256(self.literal).digest(),
            )

    def test_xz_truncation_is_rejected(self) -> None:
        payload = self.encoded["canonical_xz"][:-1] + self.encoded["route"]
        with self.assertRaises((ValueError, EOFError)):
            codec.decode_side_payload(
                BASE,
                payload,
                len(self.literal),
                hashlib.sha256(self.literal).digest(),
            )

    def test_profile_offset_header_tamper_is_rejected(self) -> None:
        corrupt = bytearray(self.literal)
        # chunk_count is the sixth u32 after the 8-byte magic.
        struct.pack_into("<I", corrupt, 8 + 4 * 4, 399)
        with self.assertRaises(ValueError):
            codec.profile_code_offsets(BASE, bytes(corrupt))

    def test_bundle_prelude_source_free_parse(self) -> None:
        prefix, mask = bundle_prefix(self.literal)
        handle = io.BytesIO(prefix)
        prelude = codec.read_bundle_prelude_v2(BASE, handle)
        self.assertEqual(prelude.side.blob, self.literal)
        self.assertEqual(prelude.raw_mask, mask)
        self.assertEqual(handle.read(), b"")

    def test_bundle_rejects_payload_hash_tamper(self) -> None:
        prefix, _ = bundle_prefix(self.literal)
        corrupt = bytearray(prefix)
        corrupt[BASE.OUTER_HEADER.size + 5] ^= 1
        with self.assertRaises(ValueError):
            codec.read_bundle_prelude_v2(BASE, io.BytesIO(corrupt))

    def test_bundle_rejects_wrong_codec_id(self) -> None:
        prefix, _ = bundle_prefix(self.literal)
        corrupt = bytearray(prefix)
        # side_codec is the fourth u32 after 8-byte magic.
        struct.pack_into("<I", corrupt, 8 + 3 * 4, BASE.SIDE_CODEC_LZMA_XZ)
        with self.assertRaises(ValueError):
            codec.read_bundle_prelude_v2(BASE, io.BytesIO(corrupt))

    def test_decoder_dependency_binding_tamper_each_hash(self) -> None:
        wrapper = Path(__file__).with_name("outer_decode_fixed_route_v2.py")
        expected = codec.dependency_bindings(
            BASE, "fixed_route_decoder_wrapper_sha256", wrapper
        )
        codec.validate_dependency_bindings(dict(expected), expected)
        for key in expected:
            with self.subTest(key=key):
                corrupt = dict(expected)
                corrupt[key] = "0" * 64
                with self.assertRaises(ValueError):
                    codec.validate_dependency_bindings(corrupt, expected)

    def test_evaluator_dependency_binding_tamper_each_hash(self) -> None:
        delegated = fixed_evaluator.load_v1_evaluator()
        expected = fixed_evaluator.evaluation_dependency_bindings(delegated)
        codec.validate_dependency_bindings(dict(expected), expected)
        for key in expected:
            with self.subTest(key=key):
                corrupt = dict(expected)
                corrupt[key] = "f" * 64
                with self.assertRaises(ValueError):
                    codec.validate_dependency_bindings(corrupt, expected)

    def test_unaudited_v1_outer_override_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "outer_decode.py"
            trusted = Path(BASE.__file__)
            corrupt.write_bytes(trusted.read_bytes() + b"\n# tamper\n")
            environment = dict(os.environ)
            environment["WFOUTR_V1_OUTER_DECODE"] = str(corrupt)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import fixed_route_codec as c; c.load_v1_outer()",
                ],
                cwd=Path(__file__).parent,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unaudited v1 outer decoder", completed.stderr)

    def test_unaudited_selector_core_fails_closed(self) -> None:
        selector = fixed_selector.load_selector()
        expected = fixed_selector.selector_dependency_bindings(selector)
        fixed_selector.validate_selector_dependency_bindings(dict(expected), expected)
        for key in expected:
            with self.subTest(selector_dependency=key):
                corrupt = dict(expected)
                corrupt[key] = "b" * 64
                with self.assertRaises(ValueError):
                    fixed_selector.validate_selector_dependency_bindings(
                        corrupt, expected
                    )
        with tempfile.TemporaryDirectory() as directory:
            corrupt_core = Path(directory) / "select_continuous_adaptive.py"
            corrupt_core.write_bytes(
                Path(selector.__file__).read_bytes() + b"\n# tamper\n"
            )
            with self.assertRaisesRegex(ValueError, "unaudited selector core"):
                fixed_selector.load_selector(corrupt_core)

    def test_selector_receipt_records_exact_dependencies(self) -> None:
        selector = fixed_selector.load_selector()
        expected = fixed_selector.selector_dependency_bindings(selector)
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "selection.receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": (
                            "continuous PLTE exact adaptive selection receipt "
                            "fixed-route v2"
                        ),
                        "status": "passed",
                        "physical_selection_mode": "fixed-route",
                    }
                ),
                encoding="utf-8",
            )
            fixed_selector.bind_selection_receipt(receipt_path, expected)
            rebound = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(rebound["selector_dependency_bindings"], expected)
            checksum = receipt_path.with_name("selection.receipt.sha256").read_text(
                encoding="ascii"
            )
            self.assertEqual(
                checksum,
                f"{fixed_selector.sha256_path(receipt_path)}  "
                "selection.receipt.json\n",
            )

    def test_isolated_packer_roundtrip_with_400_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            side = root / "side.bin"
            side.write_bytes(self.literal)
            mask = root / "mask.raw"
            mask.write_bytes(
                bytes(
                    BASE.BASE_MASK_LEVELS
                    * ((BASE.NORMATIVE_BLOCK_LENGTH + 7) // 8)
                )
            )
            containers = root / "containers"
            containers.mkdir()
            frame = struct.pack("<If", 0, 1.0)
            for index in range(400):
                (containers / f"wf-{index:03d}.polar.bin").write_bytes(frame)
            bundle = root / "test.wfouter"
            receipt = root / "pack.receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("pack_bundle_fixed_route_v2.py")),
                    "--side",
                    str(side),
                    "--container-dir",
                    str(containers),
                    "--raw-mask",
                    str(mask),
                    "--output",
                    str(bundle),
                    "--receipt",
                    str(receipt),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            row = json.loads(receipt.read_text())
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["side"]["route_bits"], 400)
            self.assertEqual(row["side"]["route_bytes"], 50)
            self.assertEqual(row["containers"]["count"], 400)
            self.assertEqual(row["bundle_bytes"], bundle.stat().st_size)
            expected_dependencies = codec.dependency_bindings(
                BASE,
                "fixed_route_packer_wrapper_sha256",
                Path(__file__).with_name("pack_bundle_fixed_route_v2.py"),
            )
            codec.validate_dependency_bindings(
                row["dependency_bindings"], expected_dependencies
            )
            for key in expected_dependencies:
                with self.subTest(packer_dependency=key):
                    corrupt = dict(row["dependency_bindings"])
                    corrupt[key] = "a" * 64
                    with self.assertRaises(ValueError):
                        codec.validate_dependency_bindings(
                            corrupt, expected_dependencies
                        )


if __name__ == "__main__":
    unittest.main()
