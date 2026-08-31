#!/usr/bin/env python3
"""Hardened isolated audit copy of the adaptive PLTE candidate runner.

No candidate work is permitted until the base run has one complete receipt
bound to the current manifest and encoder and all 400 canonical report/container
pairs independently validate.  Every base gap is scanned; a high-gap non-A64
chunk fails closed because this runner's A128 alternative is defined for A64
inputs only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path


N = 1 << 18
EXPECTED_CHUNKS = 400
PRINT_LOCK = threading.Lock()
ROUNDTRIP_FIELDS = (
    "arithmetic_roundtrip_bits_match",
    "online_causal_arithmetic_bits_match",
    "causal_decoder_frequencies_match",
    "causal_decoder_frozen_bits_match",
    "reconstruction_indices_match",
    "tail_escape_records_roundtrip",
    "tail_escape_padding_is_zero",
    "container_header_roundtrip",
)


def progress(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def run_logged(command: list[str], log: Path, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    atomic_write_text(log, completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def base_artifacts(base_dir: Path, index: int) -> tuple[Path, Path]:
    stem = base_dir / f"wf-{index:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".polar.bin")


def resolve_path(path_text: str, repo: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo / path


def validate_manifest(manifest: dict) -> list[dict]:
    chunks = manifest["chunks"]
    if len(chunks) != EXPECTED_CHUNKS:
        raise AssertionError(f"expected {EXPECTED_CHUNKS} chunks, found {len(chunks)}")
    if [int(chunk["chunk_index"]) for chunk in chunks] != list(range(EXPECTED_CHUNKS)):
        raise AssertionError("manifest chunk indices are not canonical 0..399")
    if len({int(chunk["chunk_index"]) for chunk in chunks}) != EXPECTED_CHUNKS:
        raise AssertionError("manifest has duplicate chunk indices")
    for chunk in chunks:
        if int(chunk["alphabet_size"]) not in (64, 128):
            raise ValueError(f"unsupported manifest alphabet {chunk['alphabet_size']}")
        if len(chunk["members"]) != 128:
            raise AssertionError("manifest chunk does not contain 128 groups")
        if not (
            math.isfinite(float(chunk["test_distortion"]))
            and float(chunk["test_distortion"]) > 0.0
            and math.isfinite(float(chunk["eta"]))
            and float(chunk["eta"]) > 0.0
        ):
            raise ValueError("non-finite manifest PLTE profile")
    return chunks


def validate_source_file(chunk: dict, repo: Path) -> Path:
    path = resolve_path(str(chunk["normalized_source"]), repo)
    if not path.is_file() or path.stat().st_size != 2 * N:
        raise ValueError(f"normalized source geometry mismatch: {path}")
    if sha256_path(path) != str(chunk["normalized_source_sha256"]):
        raise ValueError(f"normalized source hash mismatch: {path}")
    return path


def validate_report_container(
    report_path: Path,
    container_path: Path,
    chunk: dict,
    expected_alphabet: int,
    expected_encoder_sha256: str,
    repo: Path,
) -> tuple[dict, dict]:
    if not report_path.is_file() or not container_path.is_file():
        raise FileNotFoundError(f"missing report/container: {report_path}, {container_path}")
    report = load_json(report_path)
    if str(report["implementation_sha256"]) != expected_encoder_sha256:
        raise AssertionError(f"encoder hash mismatch in {report_path}")
    parameters = report["parameters"]
    trials = report["trials"]
    if len(trials) != 1:
        raise AssertionError(f"{report_path}: expected exactly one trial")
    trial = trials[0]
    normalized_path = validate_source_file(chunk, repo)
    checks = (
        int(parameters["block_length"]) == N,
        int(parameters["trials"]) == 1,
        int(parameters["alphabet_size"]) == expected_alphabet,
        int(parameters["container_cap_bytes"]) == 0,
        float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"]),
        float(parameters["eta"]) == float(chunk["eta"]),
        int(trial["trial"]) == 0,
        trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"],
        int(trial["source"]["values"]) == N,
        int(trial["tail_escape_count"]) == 0,
        int(trial["base_literal_container_bytes"]) == container_path.stat().st_size,
        int(trial["literal_container_bytes"]) == container_path.stat().st_size,
        trial["literal_container_sha256"] == sha256_path(container_path),
        all(trial[field] is True for field in ROUNDTRIP_FIELDS),
    )
    if not all(checks):
        raise AssertionError(f"report/container validation failed: {report_path}")
    for field in ("relative_mse", "gap_db", "screen_bpw"):
        if not math.isfinite(float(trial[field])):
            raise ValueError(f"non-finite {field} in {report_path}")
    if int(trial["arithmetic_payload_bytes"]) + 8 != container_path.stat().st_size:
        raise AssertionError("zero-tail base byte accounting mismatch")
    if int(trial["total_screen_bits"]) != 8 * container_path.stat().st_size:
        raise AssertionError("report physical-bit accounting mismatch")
    if float(trial["screen_bpw"]) != 8.0 * container_path.stat().st_size / N:
        raise AssertionError("report physical-rate accounting mismatch")
    # The encoder report binds the source hash; this check binds the current
    # physical source file used by any resumed candidate.
    if sha256_path(normalized_path) != trial["source"]["block_bf16_sha256"]:
        raise AssertionError("current normalized source does not match report")
    return report, trial


def validate_complete_base_run(args, manifest: dict) -> list[dict]:
    """Validate the complete base receipt and every canonical physical row."""

    chunks = validate_manifest(manifest)
    manifest_hash = sha256_path(args.manifest)
    encoder_hash = sha256_path(args.encoder)
    receipt_path = args.base_receipt or (args.base_dir / "run.receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"base run receipt is absent: {receipt_path}")
    receipt = load_json(receipt_path)
    if not (
        receipt["status"] == "complete"
        and receipt["manifest_sha256"] == manifest_hash
        and receipt["encoder_sha256"] == encoder_hash
        and int(receipt["chunks"]) == EXPECTED_CHUNKS
        and bool(receipt["all_internal_roundtrips_passed"])
        and receipt["failures"] == []
    ):
        raise RuntimeError("base run receipt is incomplete or not bound to manifest/encoder")
    receipt_rows = receipt["rows"]
    if len(receipt_rows) != EXPECTED_CHUNKS:
        raise RuntimeError("base receipt does not contain exactly 400 rows")
    if [int(row["chunk_index"]) for row in receipt_rows] != list(range(EXPECTED_CHUNKS)):
        raise RuntimeError("base receipt rows are not canonical 0..399")
    if len({int(row["chunk_index"]) for row in receipt_rows}) != EXPECTED_CHUNKS:
        raise RuntimeError("base receipt contains duplicate rows")

    validated: list[dict] = []
    total_bytes = 0
    for index, (chunk, receipt_row) in enumerate(zip(chunks, receipt_rows, strict=True)):
        report_path, container_path = base_artifacts(args.base_dir, index)
        if Path(receipt_row["report"]).resolve() != report_path.resolve():
            raise RuntimeError(f"base receipt report path mismatch at chunk {index}")
        if Path(receipt_row["container"]).resolve() != container_path.resolve():
            raise RuntimeError(f"base receipt container path mismatch at chunk {index}")
        report, trial = validate_report_container(
            report_path,
            container_path,
            chunk,
            int(chunk["alphabet_size"]),
            encoder_hash,
            args.repo,
        )
        if not (
            int(receipt_row["container_bytes"]) == container_path.stat().st_size
            and math.isclose(
                float(receipt_row["relative_mse"]),
                float(trial["relative_mse"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RuntimeError(f"base receipt row mismatch at chunk {index}")
        total_bytes += container_path.stat().st_size
        validated.append(
            {
                "chunk": chunk,
                "report_path": report_path,
                "container_path": container_path,
                "report": report,
                "trial": trial,
            }
        )
    if int(receipt["actual_container_bytes"]) != total_bytes:
        raise RuntimeError("base receipt aggregate byte count mismatch")
    expected_bpw = 8.0 * total_bytes / (EXPECTED_CHUNKS * N)
    if not math.isclose(
        float(receipt["actual_container_bpw_before_outer_side"]),
        expected_bpw,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("base receipt aggregate bpw mismatch")
    return validated


def raw_source_bindings(manifest: dict, chunk: dict, repo: Path) -> list[dict]:
    bindings: list[dict] = []
    for ordinal in sorted({int(member["block_ordinal"]) for member in chunk["members"]}):
        block = manifest["blocks"][ordinal]
        path = resolve_path(str(block["source_path"]), repo)
        if not path.is_file() or path.stat().st_size != 2 * N:
            raise ValueError(f"raw source geometry mismatch: {path}")
        digest = sha256_path(path)
        if digest != str(block["source_sha256"]):
            raise ValueError(f"raw source hash mismatch: {path}")
        bindings.append(
            {
                "block_ordinal": ordinal,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return bindings


def expected_binding(args, manifest: dict, chunk: dict, metadata: Path, container: Path) -> dict:
    normalized = validate_source_file(chunk, args.repo)
    return {
        "manifest_sha256": sha256_path(args.manifest),
        "metadata_sha256": sha256_path(metadata),
        "container_sha256": sha256_path(container),
        "scorer_sha256": sha256_path(args.scorer),
        "decoder_sha256": sha256_path(args.decoder),
        "raw_mask_sha256": sha256_path(args.raw_mask),
        "normalized_source_sha256": sha256_path(normalized),
        "raw_sources": raw_source_bindings(manifest, chunk, args.repo),
    }


def valid_a128(
    report_path: Path,
    container_path: Path,
    chunk: dict,
    encoder_hash: str,
    repo: Path,
) -> bool:
    try:
        validate_report_container(
            report_path, container_path, chunk, 128, encoder_hash, repo
        )
        return True
    except (OSError, KeyError, TypeError, ValueError, AssertionError, json.JSONDecodeError):
        return False


def parse_frame(path: Path) -> tuple[int, int, bytes, bytes]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("short PLTE frame")
    word, scale = struct.unpack("<If", data[:8])
    if not math.isfinite(scale):
        raise ValueError("non-finite PLTE scale")
    logical_bits = word & ((1 << 20) - 1)
    count = word >> 20
    payload_bytes = (logical_bits + 7) // 8
    tail_bytes = (34 * count + 7) // 8
    if len(data) != 8 + payload_bytes + tail_bytes:
        raise ValueError("PLTE exact-frame length mismatch")
    payload = data[8 : 8 + payload_bytes]
    arithmetic_padding = payload_bytes * 8 - logical_bits
    if arithmetic_padding and payload[-1] & ((1 << arithmetic_padding) - 1):
        raise ValueError("nonzero arithmetic padding")
    tail = data[8 + payload_bytes :]
    tail_padding = tail_bytes * 8 - 34 * count
    if tail_padding and tail[-1] & ((1 << tail_padding) - 1):
        raise ValueError("nonzero tail padding")
    return logical_bits, count, payload, tail


def valid_tails(
    report_path: Path,
    base_container: Path,
    requested_ks: tuple[int, ...],
    ranking: str,
    args,
    manifest: dict,
    chunk: dict,
    base_metadata: Path,
) -> bool:
    try:
        report = load_json(report_path)
        expected_ranking = (
            "descending original-coordinate SSE gain, stable ordinal ties"
            if ranking == "raw-gain"
            else "descending normalized squared residual, stable ordinal ties"
        )
        normalized = validate_source_file(chunk, args.repo)
        checks = (
            report["format"] == "exploratory PLTE sparse-tail prefix repack audit v2",
            report["strict_ptq"] is True,
            report["training_or_retraining"] is False,
            int(report["chunk_index"]) == int(chunk["chunk_index"]),
            report["implementation_sha256"] == sha256_path(args.repacker),
            report["base_container_sha256"] == sha256_path(base_container),
            report["base_metadata_sha256"] == sha256_path(base_metadata),
            report["manifest_sha256"] == sha256_path(args.manifest),
            report["decoder_sha256"] == sha256_path(args.decoder),
            report["raw_mask_sha256"] == sha256_path(args.raw_mask),
            report["normalized_source_sha256"] == sha256_path(normalized),
            report["raw_sources"] == raw_source_bindings(manifest, chunk, args.repo),
            report["stable_ranking"] == expected_ranking,
            tuple(int(k) for k in report["requested_escape_counts"]) == requested_ks,
            report["base_decoded_with_clean_decoder"] is True,
            report["all_candidates_independently_reparsed"] is True,
            report["all_scores_apply_reparsed_tail_bytes"] is True,
            report["all_raw_gain_identities_passed"] is True,
        )
        if not all(checks):
            return False
        rows = report["rows"]
        if tuple(int(row["escape_count"]) for row in rows) != requested_ks:
            return False
        _, base_count, base_payload, _ = parse_frame(base_container)
        if base_count != 0:
            return False
        base_bytes = base_container.stat().st_size
        for row, k in zip(rows, requested_ks, strict=True):
            path = Path(row["container_path"])
            expected_path = report_path.parent / f"wf-{int(chunk['chunk_index']):03d}-k{k}.polar.bin"
            if path.resolve() != expected_path.resolve() or not path.is_file():
                return False
            logical, count, payload, _ = parse_frame(path)
            expected_bytes = base_bytes + (34 * k + 7) // 8
            if not (
                count == k
                and payload == base_payload
                and path.stat().st_size == expected_bytes == int(row["container_bytes"])
                and int(row["incremental_tail_bytes"]) == expected_bytes - base_bytes
                and int(row["meaningful_tail_bits"]) == 34 * k
                and sha256_path(path) == row["container_sha256"]
                and bool(row["payload_unchanged"])
                and bool(row["independent_physical_reparse_passed"])
                and bool(row["parsed_tail_applied_for_scoring"])
                and bool(row["raw_gain_identity_passed"])
                and math.isfinite(float(row["raw_sse"]))
                and float(row["raw_sse"]) >= 0.0
                and logical >= 0
            ):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, AssertionError, json.JSONDecodeError):
        return False


def valid_decode_core(path: Path, container_path: Path, index: int) -> dict | None:
    try:
        row = load_json(path)
        if not (
            row["status"] == "passed"
            and int(row["chunk_index"]) == index
            and int(row["container_bytes"]) == container_path.stat().st_size
            and row["container_sha256"] == sha256_path(container_path)
            and float(row["raw_source_energy"]) > 0.0
            and float(row["raw_sse"]) >= 0.0
            and math.isfinite(float(row["raw_source_energy"]))
            and math.isfinite(float(row["raw_sse"]))
            and row["normalized_roundtrip_matches_at_1e_12"] is True
            and int(row["tail_escape_count"]) == 0
            and row["tail_padding_zero"] is True
        ):
            return None
        return row
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def valid_decode(
    path: Path,
    container_path: Path,
    index: int,
    binding: dict,
) -> bool:
    row = valid_decode_core(path, container_path, index)
    return row is not None and row.get("adaptive_input_binding") == binding


def encode_a128_atomic(args, chunk: dict, report_path: Path, log_path: Path) -> Path:
    token = f"{os.getpid()}-{threading.get_ident()}"
    temporary_report = report_path.parent / f".{report_path.stem}.{token}.partial.json"
    temporary_container = temporary_report.with_suffix(".polar.bin")
    command = [
        str(args.python),
        str(args.encoder),
        "--polar-repo",
        str(args.polar_repo),
        "--input-bf16",
        str(resolve_path(str(chunk["normalized_source"]), args.repo)),
        "--test-distortion",
        repr(float(chunk["test_distortion"])),
        "--eta",
        repr(float(chunk["eta"])),
        "--alphabet-size",
        "128",
        "--container-cap-bytes",
        "0",
        "--emit-container-hex",
        "--output",
        str(temporary_report),
    ]
    try:
        run_logged(command, log_path, args.repo)
        if not valid_a128(
            temporary_report,
            temporary_container,
            chunk,
            sha256_path(args.encoder),
            args.repo,
        ):
            raise AssertionError("temporary A128 output did not validate")
        final_container = report_path.with_suffix(".polar.bin")
        os.replace(temporary_container, final_container)
        os.replace(temporary_report, report_path)
        return final_container
    finally:
        for path in (temporary_report, temporary_container):
            if path.exists():
                path.unlink()


def score_a128_atomic(
    args,
    manifest: dict,
    chunk: dict,
    metadata: Path,
    container: Path,
    output: Path,
    log_path: Path,
) -> None:
    token = f"{os.getpid()}-{threading.get_ident()}"
    temporary = output.parent / f".{output.stem}.{token}.partial.json"
    command = [
        str(args.python),
        str(args.scorer),
        "--decoder",
        str(args.decoder),
        "--raw-mask",
        str(args.raw_mask),
        "--manifest",
        str(args.manifest),
        "--chunk-index",
        str(int(chunk["chunk_index"])),
        "--metadata",
        str(metadata),
        "--container",
        str(container),
        "--repo",
        str(args.repo),
        "--output",
        str(temporary),
    ]
    try:
        run_logged(command, log_path, args.repo)
        row = valid_decode_core(temporary, container, int(chunk["chunk_index"]))
        if row is None:
            raise AssertionError("temporary A128 independent decode did not validate")
        row["adaptive_input_binding"] = expected_binding(
            args, manifest, chunk, metadata, container
        )
        atomic_write_json(output, row)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_one(args, manifest: dict, base_row: dict) -> dict:
    chunk = base_row["chunk"]
    index = int(chunk["chunk_index"])
    if int(chunk["alphabet_size"]) != 64:
        raise AssertionError("adaptive A128 generation requires an A64 base trigger")
    base_report = base_row["report_path"]
    base_container = base_row["container_path"]
    a128_dir = args.output_dir / "a128"
    tail_dir = args.output_dir / "tails" / f"wf-{index:03d}"
    log_dir = args.output_dir / "logs"
    for directory in (a128_dir, tail_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    a128_report = a128_dir / f"wf-{index:03d}-a128.json"
    a128_container = a128_dir / f"wf-{index:03d}-a128.polar.bin"
    if not valid_a128(
        a128_report,
        a128_container,
        chunk,
        sha256_path(args.encoder),
        args.repo,
    ):
        produced = encode_a128_atomic(
            args, chunk, a128_report, log_dir / f"wf-{index:03d}-a128.log"
        )
        if produced != a128_container:
            raise AssertionError("unexpected final A128 path")
    if not valid_a128(
        a128_report,
        a128_container,
        chunk,
        sha256_path(args.encoder),
        args.repo,
    ):
        raise AssertionError(f"invalid A128 candidate for chunk {index}")

    tail_report = tail_dir / f"wf-{index:03d}-tail-prefixes.json"
    requested_ks = tuple(sorted(set(args.tail_ks)))
    if not valid_tails(
        tail_report,
        base_container,
        requested_ks,
        args.tail_ranking,
        args,
        manifest,
        chunk,
        base_report,
    ):
        command = [
            str(args.python),
            str(args.repacker),
            "--decoder",
            str(args.decoder),
            "--raw-mask",
            str(args.raw_mask),
            "--manifest",
            str(args.manifest),
            "--chunk-index",
            str(index),
            "--base-metadata",
            str(base_report),
            "--base-container",
            str(base_container),
            "--repo",
            str(args.repo),
            "--output-dir",
            str(tail_dir),
            "--ranking",
            args.tail_ranking,
            "--ks",
            *(str(value) for value in requested_ks),
        ]
        run_logged(command, log_dir / f"wf-{index:03d}-tails.log", args.repo)
    if not valid_tails(
        tail_report,
        base_container,
        requested_ks,
        args.tail_ranking,
        args,
        manifest,
        chunk,
        base_report,
    ):
        raise AssertionError(f"invalid tail candidates for chunk {index}")

    a128_decode = a128_dir / f"wf-{index:03d}-a128.decode.json"
    binding = expected_binding(args, manifest, chunk, a128_report, a128_container)
    if not valid_decode(a128_decode, a128_container, index, binding):
        score_a128_atomic(
            args,
            manifest,
            chunk,
            a128_report,
            a128_container,
            a128_decode,
            log_dir / f"wf-{index:03d}-a128-decode.log",
        )
    if not valid_decode(a128_decode, a128_container, index, binding):
        raise AssertionError(f"invalid A128 raw decode for chunk {index}")

    tails = load_json(tail_report)
    decoded = load_json(a128_decode)
    return {
        "chunk_index": index,
        "trigger_gap_db": float(base_row["trial"]["gap_db"]),
        "base": {
            "report": str(base_report),
            "container": str(base_container),
            "container_bytes": base_container.stat().st_size,
            "container_sha256": sha256_path(base_container),
            "raw_source_energy": float(tails["raw_source_energy"]),
            "raw_sse": float(tails["base_raw_sse"]),
        },
        "a128": {
            "report": str(a128_report),
            "container": str(a128_container),
            "decode": str(a128_decode),
            "container_bytes": a128_container.stat().st_size,
            "container_sha256": sha256_path(a128_container),
            "raw_source_energy": float(decoded["raw_source_energy"]),
            "raw_sse": float(decoded["raw_sse"]),
            "independent_decode_passed": True,
        },
        "tails": tails["rows"],
    }


def triggered_rows(validated: list[dict], threshold: float) -> list[dict]:
    """Compute the trigger from the complete 400-row validated universe."""

    result = [row for row in validated if float(row["trial"]["gap_db"]) > threshold]
    expected = [
        index
        for index, row in enumerate(validated)
        if float(row["trial"]["gap_db"]) > threshold
    ]
    actual = [int(row["chunk"]["chunk_index"]) for row in result]
    if actual != expected:
        raise AssertionError("triggered set is not the exact all-base gap predicate")
    non_a64 = [
        int(row["chunk"]["chunk_index"])
        for row in result
        if int(row["chunk"]["alphabet_size"]) != 64
    ]
    if non_a64:
        raise RuntimeError(
            "high-gap base chunks outside the supported A64 adaptive path: "
            f"{non_a64}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--repacker", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--trigger-gap-db", type=float, default=0.12)
    parser.add_argument(
        "--tail-ks", type=int, nargs="+", default=[1, 3, 7, 15, 30, 60, 120]
    )
    parser.add_argument(
        "--tail-ranking", choices=("raw-gain", "normalized"), default="raw-gain"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not math.isfinite(args.trigger_gap_db):
        raise ValueError("trigger threshold must be finite")
    args.tail_ks = sorted(set(int(k) for k in args.tail_ks))
    if not args.tail_ks or any(k <= 0 or k >= (1 << 12) for k in args.tail_ks):
        raise ValueError(f"invalid tail prefix lengths: {args.tail_ks}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    for path in (
        args.manifest,
        args.encoder,
        args.repacker,
        args.scorer,
        args.decoder,
        args.raw_mask,
        args.python,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.base_dir.is_dir() or not args.repo.is_dir() or not args.polar_repo.is_dir():
        raise NotADirectoryError("base, repository, or polar-reference directory missing")

    manifest = load_json(args.manifest)
    # Critical ordering: no output directory or candidate is created before the
    # complete base receipt and all 400 physical rows pass.
    validated = validate_complete_base_run(args, manifest)
    if len(validated) != EXPECTED_CHUNKS:
        raise AssertionError("base validation did not return exactly 400 rows")
    triggered = triggered_rows(validated, args.trigger_gap_db)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(generate_one, args, manifest, row): int(row["chunk"]["chunk_index"])
            for row in triggered
        }
        for count, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            index = future_map[future]
            try:
                rows.append(future.result())
                progress(f"[{count}/{len(triggered)}] chunk {index:03d} complete")
            except BaseException as error:
                failures.append({"chunk_index": index, "error": repr(error)})
                progress(f"[{count}/{len(triggered)}] chunk {index:03d} FAILED {error!r}")
    rows.sort(key=lambda row: int(row["chunk_index"]))
    trigger_indices = [int(row["chunk"]["chunk_index"]) for row in triggered]
    if [int(row["chunk_index"]) for row in rows] != trigger_indices and not failures:
        raise AssertionError("candidate row set differs from the exact trigger set")
    base_receipt = args.base_receipt or (args.base_dir / "run.receipt.json")
    receipt = {
        "format": "continuous PLTE adaptive candidate receipt audit v2",
        "status": "complete" if not failures and len(rows) == len(triggered) else "failed",
        "strict_ptq": True,
        "training_or_retraining": False,
        "implementation_sha256": sha256_path(Path(__file__)),
        "manifest_sha256": sha256_path(args.manifest),
        "base_receipt_sha256": sha256_path(base_receipt),
        "base_receipt_status": "complete",
        "encoder_sha256": sha256_path(args.encoder),
        "repacker_sha256": sha256_path(args.repacker),
        "scorer_sha256": sha256_path(args.scorer),
        "decoder_sha256": sha256_path(args.decoder),
        "raw_mask_sha256": sha256_path(args.raw_mask),
        "base_reports_scanned": len(validated),
        "scanned_chunk_indices": list(range(EXPECTED_CHUNKS)),
        "trigger_gap_db_strictly_greater_than": args.trigger_gap_db,
        "trigger_predicate_universe": "all 400 canonical validated base gaps",
        "triggered_chunk_indices": trigger_indices,
        "tail_prefixes": args.tail_ks,
        "tail_ranking": args.tail_ranking,
        "rows": rows,
        "failures": failures,
    }
    output = args.output_dir / "candidate.receipt.json"
    atomic_write_json(output, receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
