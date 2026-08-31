#!/usr/bin/env python3
"""Unit and subprocess smoke tests for ``run_base_clean_scores.py``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import run_base_clean_scores as scorer


ENCODER_HASH = "ab" * 32
MANIFEST_HASH = "cd" * 32
SCRIPT_HASH = "de" * 32
DECODER_HASH = "ef" * 32
MASK_HASH = "12" * 32
SOURCE_DIGEST = "34" * 32
NORMALIZED_DIGEST = "56" * 32


def write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def synthetic_job(root: Path) -> tuple[dict, scorer.BaseJob, Path, Path]:
    base = root / "base"
    base.mkdir()
    normalized = root / "normalized.bf16.bin"
    normalized_hash = write(normalized, b"\0" * (scorer.BLOCK_VALUES * 2))
    container = base / "wf-000.polar.bin"
    container_payload = struct.pack("<If", 8, 1.0) + b"\x80"
    container_hash = write(container, container_payload)
    chunk = {
        "chunk_index": 0,
        "normalized_source": str(normalized),
        "normalized_source_sha256": normalized_hash,
        "nominal_rate": 1.5,
        "test_distortion": 0.4,
        "eta": 0.6,
        "alphabet_size": 64,
    }
    trial = {
        "trial": 0,
        "source": {
            "path": str(normalized),
            "values": scorer.BLOCK_VALUES,
            "block_bf16_sha256": normalized_hash,
        },
        "relative_mse": 0.5,
        "literal_container_bytes": len(container_payload),
        "literal_container_sha256": container_hash,
        "arithmetic_logical_bits": 8,
        "arithmetic_payload_sha256": hashlib.sha256(b"\x80").hexdigest(),
        "tail_escape_count": 0,
        "passes_container_cap": True,
        **{field: True for field in scorer.ROUNDTRIP_FIELDS},
    }
    report = base / "wf-000.json"
    report.write_text(
        json.dumps(
            {
                "strict_ptq": True,
                "source_training_or_retraining": False,
                "implementation_sha256": ENCODER_HASH,
                "parameters": {
                    "block_length": scorer.BLOCK_VALUES,
                    "trials": 1,
                    "container_cap_bytes": 0,
                    "alphabet_size": 64,
                    "sigma_source": 3.0,
                    "test_channel_distortion": 0.4,
                    "eta": 0.6,
                },
                "trials": [trial],
            }
        ),
        encoding="utf-8",
    )
    state, job, error = scorer.inspect_base_job(base, chunk, ENCODER_HASH)
    if state != "ready" or job is None:
        raise AssertionError((state, error))
    return chunk, job, report, container


def score_payload(job: scorer.BaseJob, chunk: dict) -> dict:
    energy = 2.0
    sse = 0.4
    relative = sse / energy
    bpw = job.container.bytes * 8.0 / scorer.BLOCK_VALUES
    return {
        "format": scorer.CHUNK_SCORE_FORMAT,
        "status": "passed",
        "strict_ptq": True,
        "chunk_index": job.chunk_index,
        "nominal_rate": chunk["nominal_rate"],
        "actual_container_bpw": bpw,
        "container_bytes": job.container.bytes,
        "container_sha256": job.container.sha256,
        "logical_bits": job.logical_bits,
        "selected_symbols": 1,
        "frequency_u16_sha256": "78" * 32,
        "reconstruction_indices_sha256": "9a" * 32,
        "normalized_relative_mse": job.encoder_relative_mse,
        "encoder_relative_mse": job.encoder_relative_mse,
        "raw_source_energy": energy,
        "raw_sse": sse,
        "raw_relative_mse": relative,
        "raw_gap_at_actual_container_rate_db": 10.0
        * math.log10(relative / (2.0 ** (-2.0 * bpw))),
        "normalized_roundtrip_matches_at_1e_12": True,
        "tail_escape_count": 0,
        "tail_padding_zero": True,
        "cupy_version": "test",
        "gpu": "test GPU",
    }


class ScorerTests(unittest.TestCase):
    def test_output_inside_base_is_rejected_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            output = base / "forbidden_scores"
            args = argparse.Namespace(
                output_dir=output,
                base_dir=base,
                manifest=root / "manifest",
                run_receipt=base / "run.json",
                chunk_decoder=root / "chunk_decoder",
                decoder=root / "decoder",
                raw_mask=root / "mask",
                receipt=None,
            )
            with self.assertRaisesRegex(scorer.ValidationError, "disjoint"):
                scorer.execute(args)
            self.assertFalse(output.exists())

    def test_base_job_validation_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk, job, _, container = synthetic_job(root)
            self.assertEqual(job.logical_bits, 8)
            container.write_bytes(container.read_bytes() + b"x")
            state, _, error = scorer.inspect_base_job(root / "base", chunk, ENCODER_HASH)
            self.assertEqual(state, "invalid")
            self.assertIn("expected", str(error))

    def test_score_validation_rejects_forged_sse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chunk, job, _, _ = synthetic_job(Path(temporary))
            payload = score_payload(job, chunk)
            scorer.validate_score_payload(payload, job, chunk)
            payload["raw_sse"] = 0.5
            with self.assertRaisesRegex(scorer.ValidationError, "raw relative MSE"):
                scorer.validate_score_payload(payload, job, chunk)

    def test_subprocess_publish_resume_and_immutable_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk, job, report, container = synthetic_job(root)
            output = root / "scores"
            output.mkdir()
            fake = root / "fake_decoder.py"
            payload = score_payload(job, chunk)
            fake.write_text(
                "import argparse,json\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser(add_help=False);p.add_argument('--output');a,_=p.parse_known_args()\n"
                f"Path(a.output).write_text(json.dumps({payload!r}),encoding='utf-8')\n",
                encoding="utf-8",
            )
            dummy = root / "dummy"
            dummy.write_bytes(b"x")
            provenance = scorer.Provenance(
                MANIFEST_HASH,
                ENCODER_HASH,
                SCRIPT_HASH,
                DECODER_HASH,
                MASK_HASH,
                SOURCE_DIGEST,
                NORMALIZED_DIGEST,
            )
            args = argparse.Namespace(
                output_dir=output,
                python=Path(sys.executable),
                chunk_decoder=fake,
                decoder=dummy,
                raw_mask=dummy,
                manifest=dummy,
                repo=root,
            )
            before = (scorer.sha256_path(report), scorer.sha256_path(container))
            first = scorer.score_one(args, provenance, job, chunk)
            second = scorer.score_one(args, provenance, job, chunk)
            after = (scorer.sha256_path(report), scorer.sha256_path(container))
            self.assertEqual(first["status"], "decoded")
            self.assertEqual(second["status"], "resumed")
            self.assertEqual(before, after)
            envelope_path = output / "wf-000.clean.json"
            envelope = scorer.load_json(envelope_path)
            envelope["bindings"]["manifest_sha256"] = "00" * 32
            envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
            forged_hash = scorer.sha256_path(envelope_path)
            with self.assertRaisesRegex(scorer.ValidationError, "left untouched"):
                scorer.score_one(args, provenance, job, chunk)
            self.assertEqual(scorer.sha256_path(envelope_path), forged_hash)

    def test_math_fsum_aggregate_is_canonical(self) -> None:
        rows = [
            {
                "chunk_index": index,
                "raw_source_energy": energy,
                "raw_sse": sse,
                "container_bytes": 1,
            }
            for index, energy, sse in ((2, 1.0, 0.25), (0, 1e16, 1e15), (1, 1.0, 0.25))
        ]
        aggregate = scorer.aggregate_rows(rows)
        self.assertEqual(aggregate["raw_source_energy"], math.fsum((1e16, 1.0, 1.0)))
        self.assertEqual(aggregate["raw_sse"], math.fsum((1e15, 0.25, 0.25)))
        self.assertEqual(aggregate["summation"], "math.fsum over canonical chunk order")

    def test_final_run_receipt_must_be_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk, template, _, _ = synthetic_job(root)
            jobs = {index: replace(template, chunk_index=index) for index in range(scorer.BLOCKS)}
            chunks = [dict(chunk, chunk_index=index, alphabet_size=128 if index >= 339 else 64) for index in range(scorer.BLOCKS)]
            total_bytes = sum(job.container.bytes for job in jobs.values())
            rows = [
                {
                    "chunk_index": index,
                    "status": "resumed",
                    "report": str(jobs[index].report.path),
                    "container": str(jobs[index].container.path),
                    "container_bytes": jobs[index].container.bytes,
                    "relative_mse": jobs[index].encoder_relative_mse,
                }
                for index in range(scorer.BLOCKS)
            ]
            receipt = {
                "format": scorer.RUN_FORMAT,
                "status": "complete",
                "all_internal_roundtrips_passed": True,
                "manifest_sha256": MANIFEST_HASH,
                "encoder_sha256": ENCODER_HASH,
                "failures": [],
                "chunks": scorer.BLOCKS,
                "a64_chunks": 339,
                "a128_chunks": 61,
                "actual_container_bytes": total_bytes,
                "actual_container_bpw_before_outer_side": total_bytes * 8.0 / scorer.PANEL_VALUES,
                "rows": rows,
            }
            path = root / "run.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            scorer.validate_run_receipt(path, MANIFEST_HASH, ENCODER_HASH, jobs, chunks)
            receipt["rows"][0], receipt["rows"][1] = receipt["rows"][1], receipt["rows"][0]
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(scorer.ValidationError, "canonical order"):
                scorer.validate_run_receipt(path, MANIFEST_HASH, ENCODER_HASH, jobs, chunks)


if __name__ == "__main__":
    unittest.main()
