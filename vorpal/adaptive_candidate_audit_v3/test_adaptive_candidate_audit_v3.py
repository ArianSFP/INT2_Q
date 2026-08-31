#!/usr/bin/env python3
"""Regression tests for the all-base adaptive V3 audit path."""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

import repack_tail_prefixes as repack
import run_adaptive_candidates as runner


TAIL_GRID = [1, 3, 7, 15, 30, 60, 120, 240, 480]


def expect_failure(exception, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__} from {function.__name__}")


def test_trigger_includes_high_gap_a64_and_a128() -> None:
    validated = []
    expected = [5, 350, 399]
    for index in range(400):
        validated.append(
            {
                "chunk": {
                    "chunk_index": index,
                    "alphabet_size": 64 if index < 339 else 128,
                },
                "trial": {"gap_db": 0.25 if index in expected else 0.05},
            }
        )
    triggered = runner.trigger_all_validated(validated, 0.12)
    assert [int(row["chunk"]["chunk_index"]) for row in triggered] == expected


def test_a128_high_gap_is_tail_only() -> None:
    with tempfile.TemporaryDirectory(prefix="adaptive-v3-a128-") as temporary:
        root = Path(temporary)
        base_report = root / "wf-350.json"
        base_container = root / "wf-350.polar.bin"
        base_report.write_text("{}", encoding="utf-8")
        base_container.write_bytes(b"eightbyt")
        base_row = {
            "chunk": {"chunk_index": 350, "alphabet_size": 128},
            "trial": {"gap_db": 0.31},
            "report_path": base_report,
            "container_path": base_container,
        }
        fake_tail_report = {
            "raw_source_energy": 17.0,
            "base_raw_sse": 0.5,
            "base_decoded_with_clean_decoder": True,
            "rows": [
                {
                    "escape_count": 1,
                    "container_path": str(root / "wf-350-k1.polar.bin"),
                }
            ],
        }
        original = runner.run_tail_candidates
        runner.run_tail_candidates = lambda args, manifest, row: fake_tail_report
        try:
            produced = runner.generate_one_v3(object(), {}, base_row)
        finally:
            runner.run_tail_candidates = original
        assert produced["base_alphabet_size"] == 128
        assert produced["upgrade"] is None
        assert produced["available_option_kinds"] == ["base", "tail"]
        assert produced["base"]["independent_clean_decode_passed"] is True
        runner.validate_v3_row_schema(produced, 128)

        fake_tail_report["base_decoded_with_clean_decoder"] = "true"
        runner.run_tail_candidates = lambda args, manifest, row: fake_tail_report
        try:
            expect_failure(
                AssertionError, runner.generate_one_v3, object(), {}, base_row
            )
        finally:
            runner.run_tail_candidates = original


def test_a64_schema_has_optional_upgrade_object() -> None:
    old = {
        "chunk_index": 5,
        "trigger_gap_db": 0.2,
        "base": {
            "report": "base.json",
            "container": "base.bin",
            "container_bytes": 10,
            "container_sha256": "a" * 64,
            "raw_source_energy": 1.0,
            "raw_sse": 0.1,
        },
        "a128": {
            "report": "a128.json",
            "container": "a128.bin",
            "decode": "a128.decode.json",
            "container_bytes": 11,
            "container_sha256": "b" * 64,
            "raw_source_energy": 1.0,
            "raw_sse": 0.09,
            "independent_decode_passed": True,
        },
        "tails": [{"escape_count": 1}],
    }
    produced = runner.normalize_v2_a64_row(old)
    assert produced["base_alphabet_size"] == 64
    assert produced["upgrade"]["from_alphabet_size"] == 64
    assert produced["upgrade"]["to_alphabet_size"] == 128
    runner.validate_v3_row_schema(produced, 64)
    old["a128"]["independent_decode_passed"] = "true"
    expect_failure(AssertionError, runner.normalize_v2_a64_row, old)


def test_arbitrary_tail_k_and_header_limit() -> None:
    logical_bits = 13
    payload = bytes((0b10101010, 0b10101000))
    for k in (*TAIL_GRID, repack.MAX_ESCAPE_RECORDS):
        positions = np.arange(k, dtype=np.int32)
        values = np.arange(k, dtype=np.uint16)
        tail = repack.pack_escape_records(positions, values)
        word = logical_bits | (k << repack.LOGICAL_LENGTH_BITS)
        frame = struct.pack("<If", word, 1.0) + payload + tail
        parsed = repack.parse_container_bytes(frame)
        assert parsed[0] == logical_bits
        assert parsed[2] == payload
        assert parsed[3].size == k
        assert np.array_equal(parsed[3], positions)
        assert np.array_equal(parsed[4], values)
        assert len(frame) == 8 + 2 + (34 * k + 7) // 8

    too_many = repack.MAX_ESCAPE_RECORDS + 1
    expect_failure(
        ValueError,
        repack.pack_escape_records,
        np.arange(too_many, dtype=np.int32),
        np.arange(too_many, dtype=np.uint16),
    )


def test_cli_default_tail_grid() -> None:
    old_argv = sys.argv
    try:
        sys.argv = [
            "runner",
            "--manifest",
            "manifest",
            "--base-dir",
            "base",
            "--output-dir",
            "out",
            "--encoder",
            "encoder",
            "--repacker",
            "repacker",
            "--scorer",
            "scorer",
            "--decoder",
            "decoder",
            "--raw-mask",
            "mask",
            "--repo",
            "repo",
            "--polar-repo",
            "polar",
        ]
        args = runner.parse_args()
    finally:
        sys.argv = old_argv
    assert args.trigger_gap_db == 0.10
    assert args.tail_ks == TAIL_GRID


def main() -> None:
    tests = (
        test_trigger_includes_high_gap_a64_and_a128,
        test_a128_high_gap_is_tail_only,
        test_a64_schema_has_optional_upgrade_object,
        test_arbitrary_tail_k_and_header_limit,
        test_cli_default_tail_grid,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} adaptive V3 regressions")


if __name__ == "__main__":
    main()
