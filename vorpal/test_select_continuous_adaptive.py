#!/usr/bin/env python3
"""Synthetic tests for the exact adaptive selector core."""

from __future__ import annotations

import itertools
import hashlib
import json
import lzma
import struct
import sys
import tempfile
import textwrap
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from select_continuous_adaptive import (
    Choice,
    State,
    choose_best_physical_state,
    choose_best_state,
    compress_signature_sides,
    compress_fixed_route_sides,
    fixed_route_payload,
    objective_log,
    pareto_dp,
    parse_container,
    signature_pareto_dp,
    side_blob_for_signature,
    source_energy_matches_canonical,
    stage_selection,
)


def choice(chunk: int, name: str, byte_delta: int, saving: str) -> Choice:
    base_bytes = 100 + chunk
    return Choice(
        chunk_index=chunk,
        option_id=name,
        kind="base" if name == "base" else ("a128" if name == "a128" else "tail"),
        alphabet_size=128 if name == "a128" else 64,
        container_path=Path(name),
        container_bytes=base_bytes + byte_delta,
        container_sha256="0" * 64,
        raw_sse=Decimal("100") - Decimal(saving),
        base_raw_sse=Decimal("100"),
        savings=Decimal(saving),
        escape_count=0,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offset_binary64(value: float, ulps: int) -> float:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits + ulps))[0]


def encoder_report(container: Path, source_sha: str, alphabet: int, gap: float) -> dict:
    trial = {
        "source": {"block_bf16_sha256": source_sha},
        "literal_container_bytes": container.stat().st_size,
        "literal_container_sha256": digest(container),
        "gap_db": gap,
    }
    for field in (
        "arithmetic_roundtrip_bits_match",
        "online_causal_arithmetic_bits_match",
        "causal_decoder_frequencies_match",
        "causal_decoder_frozen_bits_match",
        "reconstruction_indices_match",
        "tail_escape_records_roundtrip",
        "tail_escape_padding_is_zero",
        "container_header_roundtrip",
    ):
        trial[field] = True
    return {
        "parameters": {
            "block_length": 1 << 18,
            "alphabet_size": alphabet,
            "container_cap_bytes": 0,
            "test_channel_distortion": 1.0,
            "eta": 1.0,
        },
        "trials": [trial],
    }


class ParetoTests(unittest.TestCase):
    def test_dp_matches_exhaustive_objective(self) -> None:
        groups = [
            [choice(0, "base", 0, "0"), choice(0, "a128", 10, "30"), choice(0, "tail-5", 5, "20")],
            [choice(1, "base", 0, "0"), choice(1, "a128", 12, "40"), choice(1, "tail-4", 4, "8")],
            [choice(2, "base", 0, "0"), choice(2, "a128", -2, "-1"), choice(2, "tail-7", 7, "16")],
        ]
        base_bytes = sum(group[0].container_bytes for group in groups)
        base_sse = Decimal("1000")
        panel_values = 1000
        prelude = 5
        frontier = pareto_dp(groups)
        selected, score = choose_best_state(
            frontier,
            base_raw_sse=base_sse,
            base_container_bytes=base_bytes,
            constant_proxy_bytes=prelude,
            panel_values=panel_values,
            max_bpw=Decimal("100"),
        )
        exhaustive = []
        for rows in itertools.product(*groups):
            delta = sum(row.container_bytes - group[0].container_bytes for row, group in zip(rows, groups, strict=True))
            saving = sum((row.savings for row in rows), Decimal(0))
            candidate_score = objective_log(base_sse - saving, base_bytes + delta + prelude, panel_values)
            exhaustive.append((candidate_score, base_bytes + delta + prelude, tuple(row.option_id for row in rows), delta, saving))
        expected = min(exhaustive, key=lambda row: (row[0], row[1], row[2]))
        self.assertEqual(score, expected[0])
        self.assertEqual(selected.choice_ids, expected[2])
        self.assertEqual(selected.byte_delta, expected[3])
        self.assertEqual(selected.savings, expected[4])

    def test_signature_dp_and_xz_rerank_match_exhaustive(self) -> None:
        groups = [
            [choice(0, "base", 0, "0"), choice(0, "a128", 2, "3"), choice(0, "tail-1", 1, "1")],
            [choice(1, "base", 0, "0"), choice(1, "a128", 3, "4"), choice(1, "tail-2", 2, "2")],
            [choice(2, "base", 0, "0"), choice(2, "a128", -1, "-1"), choice(2, "tail-3", 3, "5")],
        ]
        chunks = [{"alphabet_size": 64} for _ in groups]
        # Three fixed-width profile records with their alphabet bytes at 16.
        original_side = bytes(17 * len(groups))
        frontier, _ = signature_pareto_dp(groups, max_states=10000)
        compressed = compress_signature_sides(
            frontier,
            original_side=original_side,
            chunks=chunks,
            triggered=list(range(len(groups))),
            workers=2,
            max_signatures=100,
        )
        base_bytes = sum(group[0].container_bytes for group in groups)
        base_sse = Decimal("1000")
        selected, score, _ = choose_best_physical_state(
            frontier,
            side_compression=compressed,
            mask_compressed_bytes=11,
            base_raw_sse=base_sse,
            base_container_bytes=base_bytes,
            panel_values=10000,
            max_bpw=Decimal("100"),
        )
        exhaustive = []
        for rows in itertools.product(*groups):
            signature = sum(
                (1 << i) if row.alphabet_size == 128 else 0
                for i, row in enumerate(rows)
            )
            delta = sum(
                row.container_bytes - group[0].container_bytes
                for row, group in zip(rows, groups, strict=True)
            )
            saving = sum((row.savings for row in rows), Decimal(0))
            total = (
                base_bytes
                + delta
                + 168
                + compressed[signature]["compressed_bytes"]
                + 11
            )
            exhaustive.append(
                (
                    objective_log(base_sse - saving, total, 10000),
                    total,
                    tuple(row.option_id for row in rows),
                )
            )
        expected = min(exhaustive, key=lambda row: (row[0], row[1], row[2]))
        self.assertEqual((score, selected.choice_ids), (expected[0], expected[2]))

    def test_signature_dp_preserves_cross_signature_dominated_state(self) -> None:
        groups = [[
            choice(0, "base", 0, "0"),
            choice(0, "a128", 5, "-1"),
        ]]
        self.assertEqual(len(pareto_dp(groups)), 1)
        exact, _ = signature_pareto_dp(groups, max_states=10)
        self.assertEqual({state.alphabet_mask for state in exact}, {0, 1})

    def test_fixed_route_compresses_canonical_side_once_and_is_equivalent(self) -> None:
        chunks = [{"alphabet_size": 64} for _ in range(400)]
        raw = bytes(17 * 400)
        states = [
            State(0, Decimal(0), ("base", "base"), 0),
            State(1, Decimal(1), ("a128", "a128"), 3),
        ]
        with mock.patch(
            "select_continuous_adaptive.lzma.compress", wraps=lzma.compress
        ) as compressed:
            rows = compress_fixed_route_sides(
                states,
                original_side=raw,
                chunks=chunks,
                triggered=[0, 1],
            )
        self.assertEqual(compressed.call_count, 1)
        self.assertEqual(
            rows[0]["canonical_xz_sha256"], rows[3]["canonical_xz_sha256"]
        )
        for signature in (0, 3):
            selected_raw = side_blob_for_signature(
                raw, chunks, [0, 1], signature
            )
            direct = fixed_route_payload(selected_raw, chunks)
            for key in rows[signature]:
                self.assertEqual(rows[signature][key], direct[key])

    def test_pareto_prunes_equal_or_worse_savings(self) -> None:
        groups = [[
            choice(0, "base", 0, "0"),
            choice(0, "tail-bad", 5, "0"),
            choice(0, "a128", 7, "3"),
            choice(0, "tail-good", 8, "2"),
        ]]
        frontier = pareto_dp(groups)
        self.assertEqual([(row.byte_delta, row.savings) for row in frontier], [(0, Decimal(0)), (7, Decimal(3))])

    def test_negative_byte_delta_is_retained(self) -> None:
        frontier = pareto_dp([[
            choice(0, "base", 0, "0"),
            choice(0, "a128", -3, "-1"),
        ]])
        self.assertEqual([row.byte_delta for row in frontier], [-3, 0])

    def test_strict_rate_budget_excludes_equality(self) -> None:
        frontier = pareto_dp([[
            choice(0, "base", 0, "0"),
            choice(0, "a128", 1, "90"),
        ]])
        # Base total is 100 + 25 prelude = 125 bytes = exactly 1 bpw
        # over 1000 values, so strict max=1 excludes every state.
        with self.assertRaises(AssertionError):
            choose_best_state(
                frontier,
                base_raw_sse=Decimal("100"),
                base_container_bytes=100,
                constant_proxy_bytes=25,
                panel_values=1000,
                max_bpw=Decimal("1"),
            )


class SourceEnergyTests(unittest.TestCase):
    def test_four_ulps_is_inside_and_five_ulps_is_outside(self) -> None:
        canonical = Decimal(repr(100.0))
        inside = Decimal(repr(offset_binary64(100.0, 4)))
        outside = Decimal(repr(offset_binary64(100.0, 5)))
        absolute, relative, ulps = source_energy_matches_canonical(
            canonical, inside
        )
        self.assertGreater(absolute, 0)
        self.assertGreater(relative, 0)
        self.assertEqual(ulps, 4)
        with self.assertRaisesRegex(AssertionError, "exceeds canonical-source gate"):
            source_energy_matches_canonical(canonical, outside)

    def test_nonfinite_and_type_tampering_fail_closed(self) -> None:
        valid = Decimal("100.0")
        for invalid in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(nonfinite=str(invalid)):
                with self.assertRaises(AssertionError):
                    source_energy_matches_canonical(valid, invalid)
        for invalid in ("100.0", 100, 100.0, True, None):
            with self.subTest(type=type(invalid).__name__):
                with self.assertRaisesRegex(
                    AssertionError, "literal JSON decimal number"
                ):
                    source_energy_matches_canonical(valid, invalid)


class ContainerTests(unittest.TestCase):
    def test_self_delimiting_length_and_tail_padding(self) -> None:
        logical_bits = 13
        escapes = 2
        header = logical_bits | (escapes << 20)
        payload = bytes((logical_bits + 7) // 8)
        tail = bytes((34 * escapes + 7) // 8)
        blob = struct.pack("<If", header, 1.0) + payload + tail
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.polar.bin"
            path.write_bytes(blob)
            parsed = parse_container(path, expected_escape_count=escapes)
            self.assertEqual(parsed["container_bytes"], len(blob))
            path.write_bytes(blob + b"\0")
            with self.assertRaises(AssertionError):
                parse_container(path)

    def test_nonzero_tail_padding_fails(self) -> None:
        logical_bits = 8
        escapes = 1
        header = logical_bits | (escapes << 20)
        blob = struct.pack("<If", header, 1.0) + b"\0" + bytes(4) + b"\x01"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-padding.polar.bin"
            path.write_bytes(blob)
            with self.assertRaises(AssertionError):
                parse_container(path)


class EndToEndTests(unittest.TestCase):
    def test_a128_override_side_regeneration_and_400_staged_copies(self) -> None:
        """Exercise the write-once boundary with one decisive A128 option."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            candidate_dir = root / "candidates"
            base_dir.mkdir()
            candidate_dir.mkdir()
            chunks = []
            run_rows = []
            for index in range(400):
                normalized_source = root / f"wf-{index:03d}.bf16.bin"
                normalized_source.write_bytes(struct.pack("<I", index))
                source_sha = digest(normalized_source)
                base_alphabet = 128 if index == 399 else 64
                chunks.append(
                    {
                        "chunk_index": index,
                        "alphabet_size": base_alphabet,
                        "test_distortion": 1.0,
                        "eta": 1.0,
                        "normalized_source": str(normalized_source),
                        "normalized_source_sha256": source_sha,
                    }
                )
                container = base_dir / f"wf-{index:03d}.polar.bin"
                container.write_bytes(struct.pack("<If", 0, 1.0))
                report = encoder_report(
                    container,
                    source_sha,
                    base_alphabet,
                    0.2 if index in (0, 399) else 0.0,
                )
                (base_dir / f"wf-{index:03d}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                run_rows.append({"chunk_index": index, "container_bytes": 8})

            manifest = {
                "format": "synthetic",
                "strict_ptq": True,
                "training_or_retraining": False,
                "parameters": {"block_values": 1 << 18},
                "census": {"values": 400 * (1 << 18)},
                "ideal_projection": {"source_energy": 10000.0},
                "chunks": chunks,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            run_receipt = {
                "status": "complete",
                "all_internal_roundtrips_passed": True,
                "chunks": 400,
                "manifest_sha256": digest(manifest_path),
                "failures": [],
                "actual_container_bytes": 3200,
                "rows": run_rows,
            }
            (base_dir / "run.receipt.json").write_text(
                json.dumps(run_receipt), encoding="utf-8"
            )

            a128_container = candidate_dir / "wf-000-a128.polar.bin"
            a128_container.write_bytes(struct.pack("<If", 8, 1.0) + b"\0")
            a128_report = candidate_dir / "wf-000-a128.json"
            a128_report.write_text(
                json.dumps(
                    encoder_report(
                        a128_container,
                        chunks[0]["normalized_source_sha256"],
                        128,
                        0.0,
                    )
                ),
                encoding="utf-8",
            )
            raw_mask = root / "mask.raw"
            raw_mask.write_bytes(bytes(range(64)))
            a128_energy = offset_binary64(1000.0, 4)
            a128_decode = candidate_dir / "wf-000-a128.decode.json"
            a128_decode.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "chunk_index": 0,
                        "container_sha256": digest(a128_container),
                        "raw_source_energy": a128_energy,
                        "raw_sse": 1.0,
                        "adaptive_input_binding": {
                            "manifest_sha256": digest(manifest_path),
                            "metadata_sha256": digest(a128_report),
                            "container_sha256": digest(a128_container),
                            "scorer_sha256": "6" * 64,
                            "decoder_sha256": "7" * 64,
                            "raw_mask_sha256": digest(raw_mask),
                            "normalized_source_sha256": chunks[0][
                                "normalized_source_sha256"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base_container = base_dir / "wf-000.polar.bin"
            tail_container = candidate_dir / "wf-000-k1.polar.bin"
            tail_container.write_bytes(
                struct.pack("<I", 1 << 20) + base_container.read_bytes()[4:] + bytes(5)
            )
            candidate = {
                "format": "continuous PLTE all-base adaptive candidate receipt v3",
                "status": "complete",
                "strict_ptq": True,
                "training_or_retraining": False,
                "implementation_sha256": "1" * 64,
                "pinned_runner_core_sha256": "2" * 64,
                "pinned_repacker_core_sha256": "3" * 64,
                "manifest_sha256": digest(manifest_path),
                "base_receipt_sha256": digest(base_dir / "run.receipt.json"),
                "base_receipt_status": "complete",
                "encoder_sha256": "4" * 64,
                "repacker_sha256": "5" * 64,
                "scorer_sha256": "6" * 64,
                "decoder_sha256": "7" * 64,
                "raw_mask_sha256": digest(raw_mask),
                "base_reports_scanned": 400,
                "trigger_predicate_universe": "all 400 canonical validated base gaps",
                "triggered_base_alphabet_counts": {"64": 1, "128": 1},
                "row_schema": {
                    "base_alphabet_size": "required: 64 or 128",
                    "base": "required and explicitly carries alphabet_size",
                    "upgrade": "A64: required A128 object; A128: null",
                    "tails": "required prefixes against the base alphabet",
                },
                "scanned_chunk_indices": list(range(400)),
                "trigger_gap_db_strictly_greater_than": 0.10,
                "triggered_chunk_indices": [0, 399],
                "tail_prefixes": [1],
                "tail_ranking": "raw-gain",
                "failures": [],
                "rows": [
                    {
                        "chunk_index": 0,
                        "trigger_gap_db": 0.2,
                        "base_alphabet_size": 64,
                        "available_option_kinds": [
                            "base",
                            "alphabet-upgrade",
                            "tail",
                        ],
                        "base": {
                            "alphabet_size": 64,
                            "report": str(base_dir / "wf-000.json"),
                            "container": str(base_container),
                            "container_bytes": 8,
                            "container_sha256": digest(base_container),
                            "raw_source_energy": 1000.0,
                            "raw_sse": 100.0,
                        },
                        "upgrade": {
                            "kind": "alphabet-upgrade",
                            "from_alphabet_size": 64,
                            "to_alphabet_size": 128,
                            "independent_decode_passed": True,
                            "report": str(a128_report),
                            "container": str(a128_container),
                            "decode": str(a128_decode),
                            "container_bytes": 9,
                            "container_sha256": digest(a128_container),
                            "raw_source_energy": a128_energy,
                            "raw_sse": 1.0,
                        },
                        "tails": [
                            {
                                "escape_count": 1,
                                "container_path": str(tail_container),
                                "container_bytes": 13,
                                "container_sha256": digest(tail_container),
                                "payload_unchanged": True,
                                "independent_physical_reparse_passed": True,
                                "parsed_tail_applied_for_scoring": True,
                                "raw_gain_identity_passed": True,
                                "raw_source_energy": 1000.0,
                                "raw_sse": 50.0,
                            }
                        ],
                    }
                ],
            }
            base_399 = base_dir / "wf-399.polar.bin"
            tail_399 = candidate_dir / "wf-399-k1.polar.bin"
            tail_399.write_bytes(
                struct.pack("<I", 1 << 20) + base_399.read_bytes()[4:] + bytes(5)
            )
            candidate["rows"].append(
                {
                    "chunk_index": 399,
                    "trigger_gap_db": 0.2,
                    "base_alphabet_size": 128,
                    "available_option_kinds": ["base", "tail"],
                    "base": {
                        "alphabet_size": 128,
                        "report": str(base_dir / "wf-399.json"),
                        "container": str(base_399),
                        "container_bytes": 8,
                        "container_sha256": digest(base_399),
                        "raw_source_energy": 1000.0,
                        "raw_sse": 200.0,
                        "independent_clean_decode_passed": True,
                    },
                    "upgrade": None,
                    "tails": [
                        {
                            "escape_count": 1,
                            "container_path": str(tail_399),
                            "container_bytes": 13,
                            "container_sha256": digest(tail_399),
                            "payload_unchanged": True,
                            "independent_physical_reparse_passed": True,
                            "parsed_tail_applied_for_scoring": True,
                            "raw_gain_identity_passed": True,
                            "raw_source_energy": 1000.0,
                            "raw_sse": 10.0,
                        }
                    ],
                }
            )
            # Large, arbitrary validated prefixes exercise the A128-outlier
            # insurance schedule.  They are intentionally unprofitable here;
            # the exact global rate/SSE objective must retain and reject them,
            # not a hard-coded prefix whitelist.
            candidate["tail_prefixes"] = [1, 240, 480]
            for row, index, raw_sse in (
                (candidate["rows"][0], 0, 100.0),
                (candidate["rows"][1], 399, 200.0),
            ):
                base = base_dir / f"wf-{index:03d}.polar.bin"
                for k in (240, 480):
                    tail = candidate_dir / f"wf-{index:03d}-k{k}.polar.bin"
                    tail.write_bytes(
                        struct.pack("<I", k << 20)
                        + base.read_bytes()[4:]
                        + bytes((34 * k + 7) // 8)
                    )
                    row["tails"].append(
                        {
                            "escape_count": k,
                            "container_path": str(tail),
                            "container_bytes": tail.stat().st_size,
                            "container_sha256": digest(tail),
                            "payload_unchanged": True,
                            "independent_physical_reparse_passed": True,
                            "parsed_tail_applied_for_scoring": True,
                            "raw_gain_identity_passed": True,
                            "raw_source_energy": 1000.0,
                            "raw_sse": raw_sse,
                        }
                    )
            candidate_path = candidate_dir / "candidate.receipt.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            fake_packer = root / "fake_packer.py"
            fake_packer.write_text(
                textwrap.dedent(
                    """
                    import argparse, hashlib, json
                    from pathlib import Path
                    p = argparse.ArgumentParser()
                    p.add_argument('--manifest', type=Path, required=True)
                    p.add_argument('--output', type=Path, required=True)
                    p.add_argument('--receipt', type=Path, required=True)
                    a = p.parse_args()
                    manifest = json.loads(a.manifest.read_text())
                    blob = bytearray(17 * len(manifest['chunks']))
                    codes = {64: 0, 128: 1, 256: 2}
                    for index, chunk in enumerate(manifest['chunks']):
                        blob[index * 17 + 16] = codes[int(chunk['alphabet_size'])]
                    blob = bytes(blob)
                    a.output.write_bytes(blob)
                    receipt = {
                        'status': 'exact round-trip passed',
                        'exact_eof': True,
                        'side_bytes': len(blob),
                        'side_sha256': hashlib.sha256(blob).hexdigest(),
                    }
                    a.receipt.write_text(json.dumps(receipt))
                    """
                ),
                encoding="utf-8",
            )
            fake_bundle_packer = root / "fake_bundle_packer.py"
            fake_bundle_packer.write_text(
                textwrap.dedent(
                    """
                    import argparse, bz2, hashlib, json, lzma
                    from pathlib import Path
                    p = argparse.ArgumentParser()
                    p.add_argument('--side', type=Path, required=True)
                    p.add_argument('--container-dir', type=Path, required=True)
                    p.add_argument('--raw-mask', type=Path, required=True)
                    p.add_argument('--output', type=Path, required=True)
                    p.add_argument('--receipt', type=Path, required=True)
                    a = p.parse_args()
                    side = a.side.read_bytes()
                    mask = a.raw_mask.read_bytes()
                    canonical = bytearray(side)
                    route = bytearray(50)
                    for index in range(400):
                        offset = index * 17 + 16
                        if canonical[offset]:
                            route[index >> 3] |= 1 << (index & 7)
                        canonical[offset] = 0
                    canonical_xz = lzma.compress(bytes(canonical), format=lzma.FORMAT_XZ, preset=9)
                    cside = canonical_xz + bytes(route)
                    cmask = bz2.compress(mask, compresslevel=9)
                    frames = [(a.container_dir / f'wf-{i:03d}.polar.bin').read_bytes() for i in range(400)]
                    blob = bytes(168) + cside + cmask + b''.join(frames)
                    a.output.write_bytes(blob)
                    h = lambda value: hashlib.sha256(value).hexdigest()
                    receipt = {
                        'status': 'passed',
                        'source_free_reparse_passed': True,
                        'bundle_bytes': len(blob),
                        'bundle_sha256': h(blob),
                        'header_bytes': 168,
                        'side': {'raw_bytes': len(side), 'compressed_bytes': len(cside), 'canonical_xz_bytes': len(canonical_xz), 'route_bytes': len(route), 'raw_sha256': h(side), 'compressed_sha256': h(cside)},
                        'mask': {'raw_bytes': len(mask), 'compressed_bytes': len(cmask), 'raw_sha256': h(mask), 'compressed_sha256': h(cmask)},
                        'containers': {'count': len(frames), 'bytes': sum(map(len, frames)), 'ordered_sha256': [h(x) for x in frames], 'exact_eof': True},
                    }
                    a.receipt.write_text(json.dumps(receipt))
                    """
                ),
                encoding="utf-8",
            )
            output = root / "selected"
            args = SimpleNamespace(
                manifest=manifest_path,
                base_dir=base_dir,
                candidate_receipt=candidate_path,
                base_total_raw_sse=Decimal("1000"),
                total_raw_energy=Decimal("10000"),
                packer=fake_packer,
                bundle_packer=fake_bundle_packer,
                raw_mask=raw_mask,
                python=Path(sys.executable),
                output_dir=output,
                max_bpw=Decimal("2.5"),
                physical_selection="fixed-route",
                max_exact_states=10000,
                max_compression_signatures=100,
                compression_workers=2,
            )
            receipt_path = stage_selection(args)
            receipt = json.loads(receipt_path.read_text())
            selected_manifest = json.loads((output / "selected.manifest.json").read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(
                receipt["dp"]["selected_choice_ids"], ["upgrade-a128", "tail-k1"]
            )
            self.assertEqual(selected_manifest["chunks"][0]["alphabet_size"], 128)
            self.assertTrue(all(
                selected_manifest["chunks"][i]["alphabet_size"] == 64
                for i in range(1, 399)
            ))
            self.assertEqual(selected_manifest["chunks"][399]["alphabet_size"], 128)
            self.assertEqual(len(list((output / "containers").glob("wf-*.polar.bin"))), 400)
            self.assertEqual((output / "side.bin").read_bytes()[16], 1)
            self.assertEqual((output / "side.bin").read_bytes()[399 * 17 + 16], 1)
            self.assertTrue((output / "selected.wfouter").is_file())
            self.assertEqual(
                receipt["accounting"]["physical_bundle_bytes"],
                (output / "selected.wfouter").stat().st_size,
            )
            energy_audit = receipt["validation"]["source_energy_validation"]
            self.assertEqual(energy_audit["upgrade_comparisons"], 1)
            self.assertEqual(energy_audit["nonexact_upgrade_comparisons"], 1)
            self.assertEqual(energy_audit["maximum_observed_binary64_ulps"], 4)
            self.assertEqual(energy_audit["tail_exact_comparisons"], 6)

            # Literal-booleans are part of the security boundary: a truthy
            # string from a partial/legacy runner must fail closed.
            bad_candidate = json.loads(json.dumps(candidate))
            bad_candidate["rows"][0]["tails"][0][
                "independent_physical_reparse_passed"
            ] = "true"
            bad_candidate_path = candidate_dir / "bad-flags.receipt.json"
            bad_candidate_path.write_text(json.dumps(bad_candidate), encoding="utf-8")
            bad_args = SimpleNamespace(**vars(args))
            bad_args.candidate_receipt = bad_candidate_path
            bad_args.output_dir = root / "bad-flags-selected"
            with self.assertRaisesRegex(AssertionError, "tail candidate binding mismatch"):
                stage_selection(bad_args)

            # A V2 alias is never silently accepted after schema convergence.
            bad_schema = json.loads(json.dumps(candidate))
            bad_schema["format"] = "continuous PLTE adaptive candidate receipt v2"
            bad_schema_path = candidate_dir / "bad-schema.receipt.json"
            bad_schema_path.write_text(json.dumps(bad_schema), encoding="utf-8")
            bad_schema_args = SimpleNamespace(**vars(args))
            bad_schema_args.candidate_receipt = bad_schema_path
            bad_schema_args.output_dir = root / "bad-schema-selected"
            with self.assertRaisesRegex(AssertionError, "incomplete or unbound"):
                stage_selection(bad_schema_args)

if __name__ == "__main__":
    unittest.main()
