#!/usr/bin/env python3
"""Isolated regression tests for the hardened adaptive candidate copies."""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import repack_tail_prefixes as repack
import run_adaptive_candidates as runner


def expect_failure(exception, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__} from {function.__name__}")


def synthetic_manifest() -> dict:
    chunks = []
    for index in range(400):
        chunks.append(
            {
                "chunk_index": index,
                "alphabet_size": 64,
                "test_distortion": 0.29,
                "eta": 0.5989929996555583,
                "members": [{} for _ in range(128)],
            }
        )
    return {"chunks": chunks}


def test_partial_base_tree_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="adaptive-partial-") as temporary:
        root = Path(temporary)
        base_dir = root / "base"
        base_dir.mkdir()
        repo = root / "repo"
        repo.mkdir()
        manifest_path = root / "manifest.json"
        manifest = synthetic_manifest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        encoder = root / "encoder.py"
        encoder.write_text("# frozen test encoder\n", encoding="utf-8")
        receipt_path = base_dir / "run.receipt.json"
        args = SimpleNamespace(
            manifest=manifest_path,
            encoder=encoder,
            base_dir=base_dir,
            base_receipt=receipt_path,
            repo=repo,
        )

        # An in-progress receipt is rejected before a candidate directory can
        # be involved.
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "manifest_sha256": runner.sha256_path(manifest_path),
                    "encoder_sha256": runner.sha256_path(encoder),
                    "chunks": 276,
                    "all_internal_roundtrips_passed": False,
                    "failures": [],
                    "rows": [],
                }
            ),
            encoding="utf-8",
        )
        expect_failure(RuntimeError, runner.validate_complete_base_run, args, manifest)

        # A forged "complete" receipt with 400 canonical rows but no physical
        # base tree also fails. This exercises the exact regression that the
        # original scanner silently skipped.
        rows = []
        for index in range(400):
            report, container = runner.base_artifacts(base_dir, index)
            rows.append(
                {
                    "chunk_index": index,
                    "report": str(report),
                    "container": str(container),
                    "container_bytes": 8,
                    "relative_mse": 0.1,
                }
            )
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "manifest_sha256": runner.sha256_path(manifest_path),
                    "encoder_sha256": runner.sha256_path(encoder),
                    "chunks": 400,
                    "all_internal_roundtrips_passed": True,
                    "failures": [],
                    "rows": rows,
                    "actual_container_bytes": 3200,
                    "actual_container_bpw_before_outer_side": 0.0,
                }
            ),
            encoding="utf-8",
        )
        expect_failure(FileNotFoundError, runner.validate_complete_base_run, args, manifest)


def test_trigger_is_exact_all_400_predicate() -> None:
    validated = []
    expected = [3, 117, 399]
    for index in range(400):
        validated.append(
            {
                "chunk": {"chunk_index": index, "alphabet_size": 64},
                "trial": {"gap_db": 0.25 if index in expected else 0.05},
            }
        )
    actual = runner.triggered_rows(validated, 0.12)
    assert [int(row["chunk"]["chunk_index"]) for row in actual] == expected

    # A high-gap A128 input cannot be silently omitted by an A64-only runner.
    validated[117]["chunk"]["alphabet_size"] = 128
    expect_failure(RuntimeError, runner.triggered_rows, validated, 0.12)


def test_exact_tail_frame_and_padding_fail_closed() -> None:
    logical_bits = 10
    payload = bytes((0b10101010, 0b11000000))
    positions = np.asarray([2, 101, (1 << 18) - 1], dtype=np.int32)
    values = np.asarray([0x0001, 0x7FC0, 0xFFFF], dtype=np.uint16)
    tail = repack.pack_escape_records(positions, values)
    word = logical_bits | (positions.size << repack.LOGICAL_LENGTH_BITS)
    frame = struct.pack("<If", word, 1.5) + payload + tail
    parsed = repack.parse_container_bytes(frame)
    assert parsed[0] == logical_bits
    assert parsed[2] == payload
    assert np.array_equal(parsed[3], positions)
    assert np.array_equal(parsed[4], values)
    assert len(frame) == 8 + math.ceil(logical_bits / 8) + math.ceil(34 * 3 / 8)

    bad_arithmetic = bytearray(frame)
    bad_arithmetic[8 + len(payload) - 1] |= 1
    expect_failure(ValueError, repack.parse_container_bytes, bytes(bad_arithmetic))

    bad_tail = bytearray(frame)
    bad_tail[-1] |= 1
    expect_failure(ValueError, repack.parse_container_bytes, bytes(bad_tail))

    expect_failure(
        ValueError,
        repack.pack_escape_records,
        np.asarray([2, 2], dtype=np.int32),
        np.asarray([1, 2], dtype=np.uint16),
    )


def test_raw_coordinate_gain_identity() -> None:
    raw = np.asarray([2.0, -1.0, 0.1, 4.0], dtype=np.float64)
    qscale = np.asarray([2.0, 0.5, 3.0, 0.25], dtype=np.float64)
    base = np.asarray([0.5, -3.0, 0.0, 12.0], dtype=np.float64)
    normalized = np.asarray([1.0, -2.0, 0.02, 16.0], dtype=np.float64)
    base_error = np.square(raw - qscale * base)
    escaped_error = np.square(raw - qscale * normalized)
    gains = base_error - escaped_error
    ranking = np.argsort(-gains, kind="stable")
    for k in (1, 2, 3):
        chosen = ranking[:k]
        reconstructed = base.copy()
        reconstructed[chosen] = normalized[chosen]
        direct = np.sum(base_error) - np.sum(np.square(raw - qscale * reconstructed))
        analytic = np.sum(gains[chosen])
        assert math.isclose(float(direct), float(analytic), rel_tol=1e-14, abs_tol=1e-14)


def main() -> None:
    tests = (
        test_partial_base_tree_fails,
        test_trigger_is_exact_all_400_predicate,
        test_exact_tail_frame_and_padding_fail_closed,
        test_raw_coordinate_gain_identity,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} adaptive audit regressions")


if __name__ == "__main__":
    main()
