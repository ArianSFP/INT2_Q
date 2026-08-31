#!/usr/bin/env python3
"""Resumable parallel encoder runner for the frozen continuous-waterfill panel."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


PRINT_LOCK = threading.Lock()


def progress(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def valid_job(report_path: Path, container_path: Path, chunk: dict) -> bool:
    if not report_path.is_file() or not container_path.is_file():
        return False
    try:
        report = load_json(report_path)
        parameters = report["parameters"]
        trial = report["trials"][0]
        return (
            int(parameters["block_length"]) == 1 << 18
            and float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"])
            and float(parameters["eta"]) == float(chunk["eta"])
            and int(parameters["alphabet_size"]) == int(chunk["alphabet_size"])
            and int(parameters["container_cap_bytes"]) == 0
            and trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"]
            and int(trial["literal_container_bytes"]) == container_path.stat().st_size
            and trial["literal_container_sha256"] == sha256_path(container_path)
            and all(
                trial[field] is True
                for field in (
                    "arithmetic_roundtrip_bits_match",
                    "online_causal_arithmetic_bits_match",
                    "causal_decoder_frequencies_match",
                    "causal_decoder_frozen_bits_match",
                    "reconstruction_indices_match",
                    "tail_escape_records_roundtrip",
                    "tail_escape_padding_is_zero",
                    "container_header_roundtrip",
                )
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def encode_one(args, chunk: dict) -> dict:
    index = int(chunk["chunk_index"])
    stem = f"wf-{index:03d}"
    report_path = args.output_dir / f"{stem}.json"
    container_path = args.output_dir / f"{stem}.polar.bin"
    log_path = args.output_dir / f"{stem}.log"
    if valid_job(report_path, container_path, chunk):
        report = load_json(report_path)
        return {
            "chunk_index": index,
            "status": "resumed",
            "report": str(report_path),
            "container": str(container_path),
            "container_bytes": container_path.stat().st_size,
            "relative_mse": float(report["trials"][0]["relative_mse"]),
        }
    token = f"{os.getpid()}-{threading.get_ident()}"
    temporary_report = args.output_dir / f".{stem}.{token}.partial.json"
    temporary_container = temporary_report.with_suffix(".polar.bin")
    command = [
        str(args.python),
        str(args.encoder),
        "--polar-repo",
        str(args.polar_repo),
        "--input-bf16",
        str(chunk["normalized_source"]),
        "--test-distortion",
        repr(float(chunk["test_distortion"])),
        "--eta",
        repr(float(chunk["eta"])),
        "--alphabet-size",
        str(int(chunk["alphabet_size"])),
        "--container-cap-bytes",
        "0",
        "--emit-container-hex",
        "--output",
        str(temporary_report),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, cwd=args.repo)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"chunk {index} encoder failed; see {log_path}")
    if not temporary_report.is_file() or not temporary_container.is_file():
        raise RuntimeError(f"chunk {index} encoder omitted outputs")
    temporary_report.replace(report_path)
    temporary_container.replace(container_path)
    if not valid_job(report_path, container_path, chunk):
        raise AssertionError(f"chunk {index} output validation failed")
    report = load_json(report_path)
    return {
        "chunk_index": index,
        "status": "encoded",
        "report": str(report_path),
        "container": str(container_path),
        "container_bytes": container_path.stat().st_size,
        "relative_mse": float(report["trials"][0]["relative_mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    chunks = manifest["chunks"]
    if len(chunks) != 400 or [int(row["chunk_index"]) for row in chunks] != list(range(400)):
        raise AssertionError("expected 400 canonical chunks")
    if sum(int(row["alphabet_size"]) == 128 for row in chunks) != 61:
        raise AssertionError("unexpected A128 census")

    rows: list[dict] = []
    failures: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(encode_one, args, chunk): int(chunk["chunk_index"]) for chunk in chunks}
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index = futures[future]
            try:
                row = future.result()
                rows.append(row)
                progress(f"[{completed_count}/400] chunk {index:03d} {row['status']} {row['container_bytes']} B")
            except BaseException as error:
                failures.append({"chunk_index": index, "error": repr(error)})
                progress(f"[{completed_count}/400] chunk {index:03d} FAILED {error!r}")
    rows.sort(key=lambda row: int(row["chunk_index"]))
    total_bytes = sum(int(row["container_bytes"]) for row in rows)
    panel_values = 400 * (1 << 18)
    receipt = {
        "format": "continuous reverse-waterfilled PLTE base panel run v1",
        "status": "complete" if not failures and len(rows) == 400 else "failed",
        "manifest_sha256": sha256_path(args.manifest),
        "encoder_sha256": sha256_path(args.encoder),
        "chunks": len(rows),
        "a64_chunks": sum(int(chunk["alphabet_size"]) == 64 for chunk in chunks),
        "a128_chunks": sum(int(chunk["alphabet_size"]) == 128 for chunk in chunks),
        "actual_container_bytes": total_bytes,
        "actual_container_bpw_before_outer_side": total_bytes * 8 / panel_values,
        "all_internal_roundtrips_passed": not failures and len(rows) == 400,
        "failures": failures,
        "rows": rows,
    }
    receipt_path = args.output_dir / "run.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
