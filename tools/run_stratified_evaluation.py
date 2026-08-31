#!/usr/bin/env python3
"""Fetch, encode, independently decode, and summarize the Qwen3 coverage panel.

The evidenced encoder and decoder are invoked as isolated subprocesses and are
never imported or modified here. Work products are resumable under ``tmp/``;
only finalized, source-free evidence is written to ``evaluation/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
DEFAULT_MANIFEST = ROOT / "evaluation" / "qwen3_stratified_v1" / "manifest.json"
DEFAULT_WORKSPACE = ROOT / "tmp" / "qwen3_stratified_v1"
ENCODER = PLTE / "agent_root_polar_lattice_gate.py"
DECODER = PLTE / "agent_polar_codec_audit_independent_decoder.py"
PROFILE = PLTE / "agent_root_polar_escape_frozen_profiles.bin"
ROUTER_REPORT = PLTE / "agent_router_adaptive_q4_t0045_all48.json"
ROUTER_CONTAINER = PLTE / "agent_router_adaptive_q4_t0045_all48.bin"
BLOCK_VALUES = 1 << 18
CONTAINER_CAP_BYTES = 81_242
TIER_STEP_BYTES = 64
TIER_MAP_BITS_PER_BLOCK = 4
MAX_TIER = 15
NONROUTER_BLOCKS = 116_422
BASE_LEDGER_BITS = 75_724_918_048
CHECKPOINT_PARAMETERS = 30_532_122_624
BASE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{shard}?download=true"

OVERFLOW_PATTERN = re.compile(
    r"RuntimeError: base polar container (?P<bytes>\d+) exceeds enforced "
    r"81242-byte slot"
)

ENCODER_BOOL_FIELDS = (
    "arithmetic_roundtrip_bits_match",
    "online_causal_arithmetic_bits_match",
    "causal_decoder_frequencies_match",
    "causal_decoder_frozen_bits_match",
    "reconstruction_indices_match",
    "tail_escape_records_roundtrip",
    "tail_escape_padding_is_zero",
    "container_header_roundtrip",
    "passes_container_cap",
    "passes_rate_lt_2p5",
)

_PRINT_LOCK = threading.Lock()


def progress(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    atomic_write_bytes(path, rendered.encode("utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path must be inside repository root: {path}") from error


def validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("format") != "PLTE Qwen3 stratified evaluation manifest v1":
        raise AssertionError("unexpected manifest format")
    if manifest.get("selection", {}).get("new_plte_blocks") != 400:
        raise AssertionError("manifest is not the frozen 400-block panel")
    if len(manifest.get("plte_blocks", [])) != 400:
        raise AssertionError("manifest PLTE row count is not 400")
    if len(manifest.get("router_blocks", [])) != 48:
        raise AssertionError("manifest router census is incomplete")
    if len(manifest.get("rank1_tensors", [])) != 193:
        raise AssertionError("manifest rank-one census is incomplete")
    identities = [(row["tensor"], int(row["block_index"])) for row in manifest["plte_blocks"]]
    if len(set(identities)) != 400:
        raise AssertionError("manifest PLTE rows are not unique")
    progress(f"manifest {sha256_path(path)}: 400 PLTE, 48 routers, 193 rank-one")


def scheduled_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latin-interleave layer roles, then distribute vocabulary strata."""
    layer_rows: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    global_rows: list[dict[str, Any]] = []
    for row in entries:
        if row["layer"] is None:
            global_rows.append(row)
        else:
            layer_rows[int(row["layer"])][str(row["role"])] = row
    roles = sorted({role for rows in layer_rows.values() for role in rows})
    if len(layer_rows) != 48 or len(roles) != 7:
        raise AssertionError("expected a complete 48 by 7 layer-role grid")
    layer_schedule = []
    for round_index in range(7):
        for layer in range(48):
            role = roles[(layer + round_index) % len(roles)]
            layer_schedule.append(layer_rows[layer][role])
    global_rows.sort(key=lambda row: str(row["id"]))
    output: list[dict[str, Any]] = []
    global_index = 0
    for index, row in enumerate(layer_schedule):
        output.append(row)
        target = (index + 1) * len(global_rows) // len(layer_schedule)
        while global_index < target:
            output.append(global_rows[global_index])
            global_index += 1
    output.extend(global_rows[global_index:])
    if len(output) != 400 or {row["id"] for row in output} != {row["id"] for row in entries}:
        raise AssertionError("evaluation schedule changed the frozen selection")
    return output


def source_paths(workspace: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    source = workspace / "sources" / f"{entry['id']}.bf16.bin"
    return source, source.with_suffix(".source.json")


def validate_cached_source(
    source: Path, metadata_path: Path, entry: dict[str, Any], revision: str
) -> dict[str, Any] | None:
    if not source.is_file() or not metadata_path.is_file():
        return None
    metadata = load_json(metadata_path)
    expected_bytes = int(entry.get("source_bytes", entry.get("bytes")))
    if (
        metadata.get("revision") != revision
        or metadata.get("id") != entry["id"]
        or metadata.get("tensor") != entry["tensor"]
        or metadata.get("absolute_byte_range_in_shard")
        != entry["absolute_byte_range_in_shard"]
        or source.stat().st_size != expected_bytes
        or metadata.get("bytes") != expected_bytes
        or metadata.get("sha256") != sha256_path(source)
    ):
        return None
    return metadata


def fetch_one(
    workspace: Path,
    entry: dict[str, Any],
    checkpoint: dict[str, str],
    kind: str,
    retries: int = 5,
) -> dict[str, Any]:
    source, metadata_path = source_paths(workspace, entry)
    cached = validate_cached_source(source, metadata_path, entry, checkpoint["revision"])
    if cached is not None:
        return cached
    start, end = (int(x) for x in entry["absolute_byte_range_in_shard"])
    expected_bytes = int(entry.get("source_bytes", entry.get("bytes")))
    if end - start + 1 != expected_bytes:
        raise AssertionError(f"manifest range length mismatch for {entry['id']}")
    url = BASE_URL.format(
        repo=checkpoint["repo"], revision=checkpoint["revision"], shard=entry["shard"]
    )
    payload: bytes | None = None
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "plte-qwen-stratified-audit/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                if getattr(response, "status", None) != 206:
                    raise RuntimeError(f"HTTP Range not honored: {response.status}")
                payload = response.read()
            if len(payload) != expected_bytes:
                raise RuntimeError(f"short range response: {len(payload)} != {expected_bytes}")
            break
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 16))
    if payload is None:
        raise RuntimeError(f"failed to fetch {entry['id']}: {last_error}")
    atomic_write_bytes(source, payload)
    metadata = {
        "id": entry["id"],
        "kind": kind,
        "repo": checkpoint["repo"],
        "revision": checkpoint["revision"],
        "tensor": entry["tensor"],
        "block_index": entry.get("block_index"),
        "shard": entry["shard"],
        "absolute_byte_range_in_shard": [start, end],
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def parallel_map(
    label: str,
    rows: list[dict[str, Any]],
    workers: int,
    function: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    completed: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, row): str(row["id"]) for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            entry_id = futures[future]
            try:
                completed.append(future.result())
            except BaseException as error:
                failures.append((entry_id, f"{type(error).__name__}: {error}"))
            if index == 1 or index % 10 == 0 or index == len(rows):
                progress(
                    f"{label}: {index}/{len(rows)} complete, "
                    f"{len(failures)} failures"
                )
    return completed, failures


def fetch_sources(
    workspace: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    workers: int,
    include_rank1: bool,
) -> dict[str, Any]:
    checkpoint = manifest["checkpoint"]
    rows: list[tuple[dict[str, Any], str]] = [(row, "plte_block") for row in selected]
    if include_rank1:
        rows.extend((row, "rank1_exact_bf16") for row in manifest["rank1_tensors"])

    wrapped = [{"id": row["id"], "entry": row, "kind": kind} for row, kind in rows]
    records, failures = parallel_map(
        "fetch",
        wrapped,
        workers,
        lambda item: fetch_one(
            workspace, item["entry"], checkpoint, item["kind"]
        ),
    )
    if failures:
        atomic_write_json(workspace / "fetch_failures.json", failures)
        raise RuntimeError(f"source fetch failed for {len(failures)} entries")
    source_manifest = {
        "format": "PLTE Qwen3 fetched source hashes v1",
        "selection_manifest_sha256": sha256_path(manifest_path),
        "checkpoint": checkpoint,
        "records": sorted(records, key=lambda row: str(row["id"])),
        "raw_source_bytes_are_publishable": False,
    }
    atomic_write_json(workspace / "source_manifest.json", source_manifest)
    return source_manifest


def validate_final_source_manifest(
    workspace: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebind every final source receipt to the selection and current bytes."""
    if source_manifest.get("format") != "PLTE Qwen3 fetched source hashes v1":
        raise AssertionError("unexpected source-manifest format")
    if source_manifest.get("selection_manifest_sha256") != sha256_path(manifest_path):
        raise AssertionError("source manifest targets a different selection")
    if source_manifest.get("checkpoint") != manifest["checkpoint"]:
        raise AssertionError("source-manifest checkpoint mismatch")
    if source_manifest.get("raw_source_bytes_are_publishable") is not False:
        raise AssertionError("source publication boundary changed")

    expected: dict[str, tuple[dict[str, Any], str]] = {}
    for entry in manifest["plte_blocks"]:
        expected[str(entry["id"])] = (entry, "plte_block")
    for entry in manifest["rank1_tensors"]:
        entry_id = str(entry["id"])
        if entry_id in expected:
            raise AssertionError("source identity is duplicated across codec classes")
        expected[entry_id] = (entry, "rank1_exact_bf16")

    records = source_manifest.get("records")
    if not isinstance(records, list) or len(records) != len(expected):
        raise AssertionError("source manifest does not contain the complete 593-row census")
    if records != sorted(records, key=lambda row: str(row.get("id"))):
        raise AssertionError("source-manifest records are not ID-sorted")
    ids = [str(row.get("id")) for row in records]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise AssertionError("source-manifest IDs are duplicate, missing, or unexpected")

    output: dict[str, dict[str, Any]] = {}
    checkpoint = manifest["checkpoint"]
    for record in records:
        entry_id = str(record["id"])
        entry, kind = expected[entry_id]
        expected_bytes = int(entry.get("source_bytes", entry.get("bytes")))
        expected_block = entry.get("block_index")
        expected_range = [int(value) for value in entry["absolute_byte_range_in_shard"]]
        if (
            record.get("kind") != kind
            or record.get("repo") != checkpoint["repo"]
            or record.get("revision") != checkpoint["revision"]
            or record.get("tensor") != entry["tensor"]
            or record.get("block_index") != expected_block
            or record.get("shard") != entry["shard"]
            or record.get("absolute_byte_range_in_shard") != expected_range
            or record.get("bytes") != expected_bytes
        ):
            raise AssertionError(f"source receipt identity mismatch: {entry_id}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AssertionError(f"source receipt hash is malformed: {entry_id}")
        source, metadata_path = source_paths(workspace, entry)
        if not source.is_file() or source.stat().st_size != expected_bytes:
            raise AssertionError(f"source byte length mismatch: {entry_id}")
        if sha256_path(source) != digest:
            raise AssertionError(f"source bytes changed after fetch: {entry_id}")
        if not metadata_path.is_file() or load_json(metadata_path) != record:
            raise AssertionError(f"source sidecar differs from final receipt: {entry_id}")
        output[entry_id] = record
    return output


def encoder_paths(workspace: Path, entry: dict[str, Any]) -> tuple[Path, Path, Path]:
    report = workspace / "jobs" / f"{entry['id']}.json"
    container = report.with_suffix(".polar.bin")
    log = workspace / "logs" / f"{entry['id']}.encode.log"
    return report, container, log


def exact_parameters(
    parameters: dict[str, Any], expected_cap_bytes: int = CONTAINER_CAP_BYTES
) -> bool:
    expected = {
        "block_length": BLOCK_VALUES,
        "trials": 1,
        "sigma_source": 3.0,
        "test_channel_distortion": 0.29,
        "eta": 0.5989929996555583,
        "alphabet_size": 64,
        "decision": "map",
        "seed": 20260831,
        "container_cap_bytes": expected_cap_bytes,
    }
    return all(parameters.get(key) == value for key, value in expected.items())


def validate_encoder_result(
    report_path: Path,
    container_path: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    encoder_sha256: str,
    expected_cap_bytes: int = CONTAINER_CAP_BYTES,
) -> dict[str, Any]:
    if not report_path.is_file() or not container_path.is_file():
        raise AssertionError("encoder output pair is incomplete")
    report = load_json(report_path)
    if report.get("implementation_sha256") != encoder_sha256:
        raise AssertionError("encoder implementation hash mismatch")
    if report.get("strict_ptq") is not True or report.get("source_training_or_retraining") is not False:
        raise AssertionError("strict PTQ declaration mismatch")
    if not exact_parameters(report.get("parameters", {}), expected_cap_bytes):
        raise AssertionError("codec parameters differ from frozen profile")
    if len(report.get("trials", [])) != 1:
        raise AssertionError("expected exactly one trial")
    trial = report["trials"][0]
    source = trial.get("source", {})
    if (
        source.get("kind") != "frozen_bf16_weight_block"
        or source.get("block_index") != 0
        or source.get("values") != BLOCK_VALUES
        or source.get("block_bf16_sha256") != source_record["sha256"]
    ):
        raise AssertionError("encoder source provenance mismatch")
    for field in ENCODER_BOOL_FIELDS:
        if trial.get(field) is not True:
            raise AssertionError(f"encoder audit flag is false: {field}")
    container_bytes = container_path.stat().st_size
    if container_bytes > expected_cap_bytes:
        raise AssertionError("literal container exceeds fixed cap")
    if trial.get("literal_container_bytes") != container_bytes:
        raise AssertionError("container length differs from metadata")
    container_sha256 = sha256_path(container_path)
    if trial.get("literal_container_sha256") != container_sha256:
        raise AssertionError("container SHA-256 differs from metadata")
    numeric = (
        "literal_decoded_absolute_mse",
        "relative_mse",
        "screen_bpw",
        "gap_db",
    )
    if any(not math.isfinite(float(trial[field])) for field in numeric):
        raise AssertionError("encoder emitted a non-finite metric")
    block_rms = float(source["block_rms_fp64"])
    if not math.isfinite(block_rms) or block_rms <= 0:
        raise AssertionError("invalid source RMS")
    absolute_mse = float(trial["literal_decoded_absolute_mse"])
    relative_mse = float(trial["relative_mse"])
    if absolute_mse <= 0 or relative_mse <= 0:
        raise AssertionError("encoder emitted a non-positive distortion")
    algebraic_relative_mse = absolute_mse / (block_rms * block_rms)
    if not math.isclose(
        relative_mse, algebraic_relative_mse, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError("encoder absolute/relative MSE algebra mismatch")
    if not math.isclose(
        float(trial.get("literal_decoded_relative_mse")),
        relative_mse,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("encoder literal relative MSE mismatch")
    actual_rate = container_bytes * 8 / BLOCK_VALUES
    actual_gap = 10.0 * math.log10(
        relative_mse / (2.0 ** (-2.0 * actual_rate))
    )
    if trial.get("total_screen_bits") != container_bytes * 8:
        raise AssertionError("encoder total bit count mismatch")
    if not math.isclose(
        float(trial["screen_bpw"]), actual_rate, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError("encoder rate algebra mismatch")
    if not math.isclose(
        float(trial["gap_db"]), actual_gap, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError("encoder gap algebra mismatch")
    return {
        "id": entry["id"],
        "quality_pass_actual_rate": float(trial["gap_db"]) < 0.10,
        "container_bytes": container_bytes,
        "container_sha256": container_sha256,
        "report_sha256": sha256_path(report_path),
    }


def encode_one(
    workspace: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    python: Path,
    polar_repo: Path,
    encoder_sha256: str,
) -> dict[str, Any]:
    report, container, log = encoder_paths(workspace, entry)
    try:
        return validate_encoder_result(
            report, container, entry, source_record, encoder_sha256
        )
    except (AssertionError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        pass

    source, _ = source_paths(workspace, entry)
    partial_report = report.with_name(f".{report.stem}.partial.json")
    partial_container = partial_report.with_suffix(".polar.bin")
    partial_report.unlink(missing_ok=True)
    partial_container.unlink(missing_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(ENCODER),
        "--polar-repo",
        str(polar_repo),
        "--block-length",
        str(BLOCK_VALUES),
        "--trials",
        "1",
        "--sigma-source",
        "3.0",
        "--test-distortion",
        "0.29",
        "--eta",
        "0.5989929996555583",
        "--alphabet-size",
        "64",
        "--decision",
        "map",
        "--seed",
        "20260831",
        "--input-bf16",
        relative_to_root(source),
        "--input-block-start",
        "0",
        "--container-cap-bytes",
        str(CONTAINER_CAP_BYTES),
        "--emit-container-hex",
        "--output",
        str(partial_report),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"encoder exited {result.returncode}: {tail}")
    validated = validate_encoder_result(
        partial_report, partial_container, entry, source_record, encoder_sha256
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_report, report)
    os.replace(partial_container, container)
    validated["wall_seconds"] = time.perf_counter() - started
    return validated


def audit_paths(workspace: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    audit = workspace / "audits" / f"{entry['id']}.json"
    log = workspace / "logs" / f"{entry['id']}.decode.log"
    return audit, log


def validate_decode_audit(
    audit_path: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    encoder_trial: dict[str, Any],
) -> dict[str, Any]:
    audit = load_json(audit_path)
    if audit.get("status") != "decoded in a clean implementation without encoder probabilities":
        raise AssertionError("independent decoder status mismatch")
    if audit.get("container_sha256") != encoder_trial["literal_container_sha256"]:
        raise AssertionError("decoder consumed the wrong container")
    if audit.get("container_bytes") != encoder_trial["literal_container_bytes"]:
        raise AssertionError("decoder container length mismatch")
    if audit.get("source_block_bf16_sha256") != source_record["sha256"]:
        raise AssertionError("decoder source hash mismatch")
    if audit.get("tail_escape_padding_zero") is not True:
        raise AssertionError("decoder found non-zero tail padding")
    if audit.get("decoded_reconstruction_matches_encoder_metric_at_1e_12") is not True:
        raise AssertionError("independent decoder MSE mismatch")
    if audit.get("decoded_indices_match_encoder_metric_at_1e_12") is not True:
        raise AssertionError("independent decoder index mismatch")
    if (
        audit.get("conditional_slot_budget_compatibility", {}).get(
            "fits_conditional_fixed_slot_budget"
        )
        is not True
    ):
        raise AssertionError("independent decoder rejects fixed-slot compatibility")
    delta = abs(
        float(audit["decoded_relative_mse_with_serialized_scale"])
        - float(encoder_trial["relative_mse"])
    )
    if delta > 1e-12:
        raise AssertionError(f"independent decoder metric delta {delta} exceeds 1e-12")
    absolute_delta = abs(
        float(audit["decoded_absolute_mse_with_serialized_scale"])
        - float(encoder_trial["literal_decoded_absolute_mse"])
    )
    if absolute_delta > 1e-12:
        raise AssertionError(
            f"independent decoder absolute-MSE delta {absolute_delta} exceeds 1e-12"
        )
    if audit.get("logical_payload_bits") != encoder_trial.get(
        "arithmetic_logical_bits"
    ):
        raise AssertionError("independent decoder logical length mismatch")
    return {
        "id": entry["id"],
        "audit_sha256": sha256_path(audit_path),
        "container_sha256": audit["container_sha256"],
        "relative_mse": audit["decoded_relative_mse_with_serialized_scale"],
    }


def decode_receipt(
    workspace: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    python: Path,
    audit_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Bind a cached or fresh audit to the exact decoder invocation inputs."""
    report, container, _ = encoder_paths(workspace, entry)
    source, _ = source_paths(workspace, entry)
    if not log_path.is_file():
        raise AssertionError("independent decoder log is missing")
    command = [
        str(python),
        str(DECODER),
        "--container",
        relative_to_root(container),
        "--container-layout",
        "plte-u20-tail-fp32",
        "--metadata",
        relative_to_root(report),
        "--raw-mask",
        relative_to_root(PROFILE),
        "--source-bf16",
        relative_to_root(source),
        "--output",
        str(audit_path.with_name(f".{audit_path.stem}.partial.json")),
    ]
    return {
        "id": entry["id"],
        "decoder_sha256": sha256_path(DECODER),
        "profile_sha256": sha256_path(PROFILE),
        "report_sha256": sha256_path(report),
        "container_sha256": sha256_path(container),
        "source_sha256": sha256_path(source),
        "audit_sha256": sha256_path(audit_path),
        "log_sha256": sha256_path(log_path),
        "canonical_argv": command,
    }


def decode_one(
    workspace: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    python: Path,
) -> dict[str, Any]:
    report, container, _ = encoder_paths(workspace, entry)
    encoder_trial = load_json(report)["trials"][0]
    audit, log = audit_paths(workspace, entry)
    try:
        validated = validate_decode_audit(audit, entry, source_record, encoder_trial)
        validated.update(
            decode_receipt(workspace, entry, source_record, python, audit, log)
        )
        return validated
    except (AssertionError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        pass
    partial = audit.with_name(f".{audit.stem}.partial.json")
    partial.unlink(missing_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(DECODER),
        "--container",
        relative_to_root(container),
        "--container-layout",
        "plte-u20-tail-fp32",
        "--metadata",
        relative_to_root(report),
        "--raw-mask",
        relative_to_root(PROFILE),
        "--source-bf16",
        relative_to_root(source_paths(workspace, entry)[0]),
        "--output",
        str(partial),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=False,
        )
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"independent decoder exited {result.returncode}: {tail}")
    validated = validate_decode_audit(partial, entry, source_record, encoder_trial)
    audit.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, audit)
    validated["wall_seconds"] = time.perf_counter() - started
    validated.update(
        decode_receipt(workspace, entry, source_record, python, audit, log)
    )
    return validated


def audit_rank1(
    workspace: Path,
    manifest: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    total_values = 0
    total_energy = 0.0
    for entry in manifest["rank1_tensors"]:
        source, _ = source_paths(workspace, entry)
        payload = source.read_bytes()
        expected_bytes = int(entry["bytes"])
        expected_values = int(entry["values"])
        if len(payload) != expected_bytes or expected_bytes != 2 * expected_values:
            raise AssertionError(f"rank-one source length mismatch: {entry['id']}")
        decoded = bytes(payload)
        values = (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16)).view(
            np.float32
        )
        if values.size != expected_values or not np.all(np.isfinite(values)):
            raise AssertionError(f"rank-one source values are invalid: {entry['id']}")
        energy = float(np.sum(values.astype(np.float64) ** 2))
        if not math.isfinite(energy) or energy <= 0:
            raise AssertionError(f"rank-one source energy is invalid: {entry['id']}")
        source_sha256 = sha256_bytes(payload)
        if source_sha256 != source_records[entry["id"]]["sha256"]:
            raise AssertionError(f"rank-one source hash mismatch: {entry['id']}")
        decoded_sha256 = sha256_bytes(decoded)
        rows.append(
            {
                "id": entry["id"],
                "layer": entry["layer"],
                "role": entry["role"],
                "tensor": entry["tensor"],
                "values": entry["values"],
                "bytes": len(payload),
                "source_sha256": source_sha256,
                "decoded_sha256": decoded_sha256,
                "source_energy_fp64": energy,
                "literal_decode_exact": decoded == payload,
                "sse": 0.0,
            }
        )
        total_values += expected_values
        total_energy += energy
    if len(rows) != 193 or total_values != 210_944:
        raise AssertionError("rank-one census size mismatch")
    if not all(row["literal_decode_exact"] for row in rows):
        raise AssertionError("rank-one literal decode mismatch")
    return {
        "format": "PLTE Qwen3 rank-one exact-BF16 census v1",
        "codec": "lossless raw BF16 exception",
        "tensor_count": len(rows),
        "values": total_values,
        "source_energy_fp64": total_energy,
        "sse": 0.0,
        "relative_mse": 0.0,
        "all_literal_decodes_exact": True,
        "tensors": rows,
    }


def validate_router_census(manifest: dict[str, Any]) -> dict[str, Any]:
    report = load_json(ROUTER_REPORT)
    if (
        report.get("strict_ptq") is not True
        or report.get("model_training_or_retraining") is not False
    ):
        raise AssertionError("router evidence is not strict PTQ")
    if report.get("router_count") != 48 or len(report.get("routers", [])) != 48:
        raise AssertionError("router report is not a complete 48-layer census")
    if report.get("checkpoint", None) not in (None, manifest["checkpoint"]):
        raise AssertionError("router checkpoint mismatch")
    if report.get("repo") != manifest["checkpoint"]["repo"]:
        raise AssertionError("router repository mismatch")
    if report.get("revision") != manifest["checkpoint"]["revision"]:
        raise AssertionError("router revision mismatch")
    if report.get("router_values") != 48 * BLOCK_VALUES:
        raise AssertionError("router value count mismatch")
    routers = report["routers"]
    if [row.get("layer") for row in routers] != list(range(48)):
        raise AssertionError("router layers are not the ordered 0--47 census")
    manifest_by_layer = {int(row["layer"]): row for row in manifest["router_blocks"]}
    energy = 0.0
    sse = 0.0
    record_bytes = 0
    for row in routers:
        layer = int(row["layer"])
        if row.get("tensor") != manifest_by_layer[layer]["tensor"]:
            raise AssertionError(f"router tensor mismatch at layer {layer}")
        if (
            row.get("input_bytes") != 2 * BLOCK_VALUES
            or not isinstance(row.get("input_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]) is None
        ):
            raise AssertionError(f"router source provenance mismatch at layer {layer}")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or {item.get("q") for item in attempts} != {
            2,
            3,
            4,
        }:
            raise AssertionError(f"router attempt set is incomplete at layer {layer}")
        if not all(item.get("labels_roundtrip") is True for item in attempts):
            raise AssertionError(f"router label round trip failed at layer {layer}")
        if row.get("selected_tag") != 4 or row.get("inverse_decode_exact") is not True:
            raise AssertionError(f"router Q4 inverse decode failed at layer {layer}")
        selected = next(item for item in attempts if item["q"] == 4)
        if not math.isclose(
            float(row["selected_sse"]),
            float(selected["sse"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row["selected_relative_mse"]),
            float(selected["relative_mse"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"router selected metric mismatch at layer {layer}")
        row_energy = float(row["energy"])
        row_sse = float(row["selected_sse"])
        if not all(math.isfinite(value) and value > 0 for value in (row_energy, row_sse)):
            raise AssertionError(f"router metric is invalid at layer {layer}")
        energy += row_energy
        sse += row_sse
        selected_record_bytes = row.get("selected_record_bytes")
        if not isinstance(selected_record_bytes, int) or selected_record_bytes <= 0:
            raise AssertionError(f"router record size is invalid at layer {layer}")
        record_bytes += selected_record_bytes
    if report.get("selected_tag_counts") != {"2": 0, "3": 0, "4": 48, "16": 0}:
        raise AssertionError("router selection is not the literal all-Q4 census")
    aggregate = report["aggregate"]
    for actual, expected, label in (
        (aggregate.get("source_energy"), energy, "energy"),
        (aggregate.get("sse"), sse, "SSE"),
        (aggregate.get("relative_mse"), sse / energy, "relative MSE"),
    ):
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AssertionError(f"router aggregate {label} mismatch")
    if aggregate["container_sha256"] != sha256_path(ROUTER_CONTAINER):
        raise AssertionError("router container SHA-256 mismatch")
    if aggregate["container_bytes"] != ROUTER_CONTAINER.stat().st_size:
        raise AssertionError("router container size mismatch")
    if int(report["format"]["global_bytes"]) + record_bytes != aggregate["container_bytes"]:
        raise AssertionError("router container record framing mismatch")
    independent = report["independent_literal_decode"]
    if (
        independent.get("exact_file_length") is not True
        or independent.get("bytes_consumed") != aggregate["container_bytes"]
        or independent.get("aggregate_energy_match") is not True
        or independent.get("aggregate_sse_match") is not True
    ):
        raise AssertionError("router independent decoder did not consume exact file")
    for actual, expected, label in (
        (independent.get("source_energy"), energy, "energy"),
        (independent.get("sse"), sse, "SSE"),
        (independent.get("relative_mse"), sse / energy, "relative MSE"),
    ):
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AssertionError(f"router independent {label} mismatch")
    return {
        "codec": "literal Q4 router exception",
        "router_count": 48,
        "layers": sorted(int(row["layer"]) for row in report["routers"]),
        "values": report["router_values"],
        "source_energy": aggregate["source_energy"],
        "sse": aggregate["sse"],
        "relative_mse": aggregate["relative_mse"],
        "container_bytes": aggregate["container_bytes"],
        "container_sha256": aggregate["container_sha256"],
        "all_inverse_decodes_exact": True,
        "independent_literal_decode": independent,
    }


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q))


def validate_reservoir_plan_for_finalization(
    workspace: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    ordered_selected: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate the post-hoc plan and preserve its original failure evidence."""
    if plan.get("format") != "PLTE Qwen3 checkpoint rate-reservoir plan v1":
        raise AssertionError("unexpected reservoir plan format")
    if plan.get("strict_ptq") is not True:
        raise AssertionError("reservoir plan is not strict PTQ")
    if plan.get("post_hoc_engineering_amendment") is not True:
        raise AssertionError("reservoir plan must disclose the post-hoc amendment")
    if plan.get("selection_manifest_sha256") != manifest_sha256:
        raise AssertionError("reservoir plan targets a different selection")
    if plan.get("checkpoint") != manifest["checkpoint"]:
        raise AssertionError("reservoir plan checkpoint mismatch")
    if plan.get("encoder_sha256") != manifest["provenance"]["encoder_sha256"]:
        raise AssertionError("reservoir plan encoder mismatch")
    expected_policy = {
        "tier0_bytes": CONTAINER_CAP_BYTES,
        "tier_step_bytes": TIER_STEP_BYTES,
        "tier_formula": "max(0, ceil((base_container_bytes - 81242) / 64))",
        "tier_map_bits_per_nonrouter_block": TIER_MAP_BITS_PER_BLOCK,
        "tier_map_order": "canonical checkpoint nonrouter block order",
        "maximum_representable_tier": MAX_TIER,
        "retry_trigger": "only the exact recognized base-container overflow exception",
        "successful_tier0_artifacts_are_reencoded": False,
        "quality_metrics_affect_tier_selection": False,
    }
    if plan.get("policy") != expected_policy:
        raise AssertionError("reservoir policy differs from the frozen tier construction")

    allocations = plan.get("allocations")
    if not isinstance(allocations, list) or len(allocations) != len(ordered_selected):
        raise AssertionError("reservoir allocation count does not match the panel")
    if allocations != sorted(allocations, key=lambda row: str(row.get("id"))):
        raise AssertionError("reservoir allocations are not ID-sorted")
    allocation_by_id = {str(row.get("id")): row for row in allocations}
    if len(allocation_by_id) != len(allocations):
        raise AssertionError("duplicate reservoir allocation id")
    entry_by_id = {str(row["id"]): row for row in ordered_selected}
    if set(allocation_by_id) != set(entry_by_id):
        raise AssertionError("reservoir allocation does not exactly cover the panel")

    tier_counts: dict[int, int] = defaultdict(int)
    tier0_gaps: list[float] = []
    overflow_rows: list[dict[str, Any]] = []
    failure_logs: dict[str, bytes] = {}
    for entry_id, entry in entry_by_id.items():
        allocation = allocation_by_id[entry_id]
        for key in ("layer", "role", "tensor", "block_index"):
            if allocation.get(key) != entry.get(key):
                raise AssertionError(
                    f"reservoir allocation identity mismatch for {entry_id}: {key}"
                )
        tier = allocation.get("tier")
        if (
            not isinstance(tier, int)
            or isinstance(tier, bool)
            or tier < 0
            or tier > MAX_TIER
        ):
            raise AssertionError(f"invalid reservoir tier for {entry_id}")
        tier_counts[tier] += 1
        expected_cap = CONTAINER_CAP_BYTES + TIER_STEP_BYTES * tier
        if allocation.get("container_cap_bytes") != expected_cap:
            raise AssertionError(f"reservoir cap formula mismatch for {entry_id}")
        if tier == 0:
            if allocation.get("first_pass_status") != "fits_tier0":
                raise AssertionError(f"Tier-0 status mismatch for {entry_id}")
            for key in ("report_sha256", "container_sha256"):
                value = allocation.get(key)
                if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise AssertionError(f"Tier-0 {key} is malformed for {entry_id}")
            container_bytes = allocation.get("container_bytes")
            if not isinstance(container_bytes, int) or not 0 < container_bytes <= expected_cap:
                raise AssertionError(f"Tier-0 size is invalid for {entry_id}")
            gap = float(allocation.get("first_pass_gap_db"))
            relative_mse = float(allocation.get("first_pass_relative_mse"))
            if not math.isfinite(gap) or not math.isfinite(relative_mse) or relative_mse <= 0:
                raise AssertionError(f"Tier-0 metric is invalid for {entry_id}")
            tier0_gaps.append(gap)
            continue

        if (
            allocation.get("first_pass_status")
            != "recognized_base_container_overflow"
            or allocation.get("retry_required") is not True
        ):
            raise AssertionError(f"overflow status mismatch for {entry_id}")
        base_bytes = allocation.get("first_pass_base_container_bytes")
        if not isinstance(base_bytes, int) or base_bytes <= CONTAINER_CAP_BYTES:
            raise AssertionError(f"overflow base length is invalid for {entry_id}")
        if allocation.get("overflow_bytes_above_tier0") != base_bytes - CONTAINER_CAP_BYTES:
            raise AssertionError(f"overflow delta mismatch for {entry_id}")
        expected_tier = math.ceil(
            (base_bytes - CONTAINER_CAP_BYTES) / TIER_STEP_BYTES
        )
        if tier != expected_tier:
            raise AssertionError(f"overflow tier formula mismatch for {entry_id}")
        expected_log_hash = allocation.get("first_pass_log_sha256")
        if (
            not isinstance(expected_log_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_log_hash) is None
        ):
            raise AssertionError(f"overflow log hash is malformed for {entry_id}")
        log_path = workspace / "logs" / f"{entry_id}.encode.log"
        payload = log_path.read_bytes()
        if len(payload) > 1 << 20 or b"\0" in payload:
            raise AssertionError(f"overflow log is not bounded plain text: {entry_id}")
        if sha256_bytes(payload) != expected_log_hash:
            raise AssertionError(f"overflow log hash mismatch for {entry_id}")
        text = payload.decode("utf-8")
        matches = list(OVERFLOW_PATTERN.finditer(text))
        if len(matches) != 1 or int(matches[0].group("bytes")) != base_bytes:
            raise AssertionError(f"overflow log does not prove the planned length: {entry_id}")
        lowered = text.lower()
        forbidden = (
            "begin openssh private key",
            "begin private key",
            "authorization:",
            "bearer ",
            "hf_token",
            "api_key",
            "/root/.ssh/",
        )
        if any(marker in lowered for marker in forbidden):
            raise AssertionError(f"overflow log contains a sensitive marker: {entry_id}")
        failure_logs[entry_id] = payload
        overflow_rows.append(allocation)

    first_pass_expected = {
        "attempted_blocks": len(ordered_selected),
        "tier0_successes": tier_counts.get(0, 0),
        "recognized_overflows": len(overflow_rows),
        "other_failures": 0,
        "tier_counts": {
            str(key): value for key, value in sorted(tier_counts.items())
        },
        "maximum_base_container_bytes": max(
            [CONTAINER_CAP_BYTES]
            + [int(row["first_pass_base_container_bytes"]) for row in overflow_rows]
        ),
        "maximum_overflow_bytes": max(
            [0]
            + [int(row["overflow_bytes_above_tier0"]) for row in overflow_rows]
        ),
        "maximum_valid_tier0_gap_db": max(tier0_gaps),
    }
    if plan.get("first_pass") != first_pass_expected:
        raise AssertionError("reservoir first-pass summary does not match allocations")

    tier_map_bits = TIER_MAP_BITS_PER_BLOCK * NONROUTER_BLOCKS
    sum_tiers = sum(tier * count for tier, count in tier_counts.items())
    strict_limit = (
        CHECKPOINT_PARAMETERS * 5 // 2 - BASE_LEDGER_BITS - tier_map_bits
    ) // (8 * TIER_STEP_BYTES)

    def checkpoint_rate(tier_sum: int) -> float:
        return (
            BASE_LEDGER_BITS
            + tier_map_bits
            + 8 * TIER_STEP_BYTES * tier_sum
        ) / CHECKPOINT_PARAMETERS

    accounting = {
        "base_ledger_bits": BASE_LEDGER_BITS,
        "checkpoint_parameters": CHECKPOINT_PARAMETERS,
        "nonrouter_blocks": NONROUTER_BLOCKS,
        "tier_map_bits": tier_map_bits,
        "bits_per_tier_increment": 8 * TIER_STEP_BYTES,
        "strict_sum_tiers_limit_below_2p5": strict_limit,
        "rate_with_no_overflow_tiers_bpw": checkpoint_rate(0),
        "rate_if_sample_tier_sum_were_checkpoint_total_bpw": checkpoint_rate(
            sum_tiers
        ),
        "rate_if_every_nonrouter_block_were_tier1_bpw": checkpoint_rate(
            NONROUTER_BLOCKS
        ),
        "rate_if_every_nonrouter_block_were_tier10_bpw": checkpoint_rate(
            10 * NONROUTER_BLOCKS
        ),
        "below_2p5_for_every_block_at_tier10": checkpoint_rate(
            10 * NONROUTER_BLOCKS
        )
        < 2.5,
    }
    if plan.get("checkpoint_rate_accounting") != accounting:
        raise AssertionError("reservoir checkpoint accounting does not recompute")
    return accounting, failure_logs


def infer_plte_prefix_length(slot: bytes) -> tuple[int, int, int]:
    """Infer the literal PLTE prefix from its u20-length/tail-count header."""
    if len(slot) < 8:
        raise AssertionError("tiered slot is shorter than the PLTE header")
    header_word = struct.unpack_from("<I", slot, 0)[0]
    logical_bits = header_word & ((1 << 20) - 1)
    escape_count = header_word >> 20
    if logical_bits <= 0:
        raise AssertionError("PLTE arithmetic length is zero")
    arithmetic_bytes = (logical_bits + 7) // 8
    escape_bits = 34 * escape_count
    escape_bytes = (escape_bits + 7) // 8
    literal_bytes = 8 + arithmetic_bytes + escape_bytes
    if literal_bytes > len(slot):
        raise AssertionError("PLTE header length exceeds its charged slot")
    if logical_bits % 8:
        padding_mask = (1 << (8 - logical_bits % 8)) - 1
        if slot[8 + arithmetic_bytes - 1] & padding_mask:
            raise AssertionError("PLTE arithmetic padding bits are non-zero")
    if escape_bits % 8:
        padding_mask = (1 << (8 - escape_bits % 8)) - 1
        if slot[literal_bytes - 1] & padding_mask:
            raise AssertionError("PLTE tail padding bits are non-zero")
    return literal_bytes, logical_bits, escape_count


def audit_packed_artifacts(
    bundle_path: Path,
    tier_map_path: Path,
    slots_path: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read back the emitted bundle/map/slots and extract every literal prefix."""
    bundle = bundle_path.read_bytes()
    tier_map = tier_map_path.read_bytes()
    slots = slots_path.read_bytes()
    if len(tier_map) != (len(results) + 1) // 2:
        raise AssertionError("tier-map byte length mismatch")
    expected_tiers = [int(row["reservoir_tier"]) for row in results]
    expected_map = bytes(
        expected_tiers[index]
        | ((expected_tiers[index + 1] if index + 1 < len(expected_tiers) else 0) << 4)
        for index in range(0, len(expected_tiers), 2)
    )
    if tier_map != expected_map:
        raise AssertionError("tier-map nibbles do not match result order")

    bundle_offset = 0
    slot_offset = 0
    for index, row in enumerate(results):
        tier_byte = tier_map[index // 2]
        decoded_tier = (
            tier_byte & 0x0F if index % 2 == 0 else (tier_byte >> 4) & 0x0F
        )
        tier = int(row["reservoir_tier"])
        if decoded_tier != tier:
            raise AssertionError(f"tier-map extraction mismatch for {row['id']}")
        charged_bytes = CONTAINER_CAP_BYTES + TIER_STEP_BYTES * tier
        if int(row["charged_slot_bytes"]) != charged_bytes:
            raise AssertionError(f"charged slot formula mismatch for {row['id']}")
        if row["container_offset"] != bundle_offset:
            raise AssertionError(f"bundle offset is non-contiguous for {row['id']}")
        if row["tiered_slot_offset"] != slot_offset:
            raise AssertionError(f"slot offset is non-contiguous for {row['id']}")
        slot = slots[slot_offset : slot_offset + charged_bytes]
        if len(slot) != charged_bytes:
            raise AssertionError(f"short tiered slot for {row['id']}")
        literal_bytes, logical_bits, escape_count = infer_plte_prefix_length(slot)
        if literal_bytes != int(row["container_bytes"]):
            raise AssertionError(f"header-derived container length mismatch for {row['id']}")
        trial = row["report"]["trials"][0]
        if (
            logical_bits != int(trial["arithmetic_logical_bits"])
            or escape_count != int(trial["tail_escape_count"])
        ):
            raise AssertionError(f"packed header metadata mismatch for {row['id']}")
        prefix = slot[:literal_bytes]
        segment = bundle[bundle_offset : bundle_offset + literal_bytes]
        if prefix != segment or sha256_bytes(prefix) != row["container_sha256"]:
            raise AssertionError(f"packed prefix/hash mismatch for {row['id']}")
        if any(slot[literal_bytes:]):
            raise AssertionError(f"non-zero reservoir padding for {row['id']}")
        if row["zero_padding_bytes"] != charged_bytes - literal_bytes:
            raise AssertionError(f"recorded reservoir padding mismatch for {row['id']}")
        bundle_offset += literal_bytes
        slot_offset += charged_bytes
    if bundle_offset != len(bundle):
        raise AssertionError("container bundle has unreferenced trailing bytes")
    if slot_offset != len(slots):
        raise AssertionError("tiered slot image has unreferenced trailing bytes")
    return {
        "format": "PLTE Qwen3 tiered-slot readback audit v1",
        "blocks": len(results),
        "order": "results sorted by id; low nibble first",
        "tier_map_bytes": len(tier_map),
        "tier_map_sha256": sha256_bytes(tier_map),
        "container_bundle_bytes": len(bundle),
        "container_bundle_sha256": sha256_bytes(bundle),
        "tiered_slots_bytes": len(slots),
        "tiered_slots_sha256": sha256_bytes(slots),
        "all_header_derived_prefixes_exact": True,
        "all_prefixes_match_independently_decoded_container_hashes": True,
        "all_arithmetic_and_tail_padding_bits_zero": True,
        "all_slot_padding_bytes_zero": True,
        "bundle_and_slot_eof_exact": True,
    }


def validate_decode_receipt(
    workspace: Path,
    entry: dict[str, Any],
    source_record: dict[str, Any],
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    report, container, _ = encoder_paths(workspace, entry)
    audit, log = audit_paths(workspace, entry)
    source, _ = source_paths(workspace, entry)
    expected_hashes = {
        "decoder_sha256": manifest["provenance"]["independent_decoder_sha256"],
        "profile_sha256": manifest["provenance"]["frozen_profile_sha256"],
        "report_sha256": sha256_path(report),
        "container_sha256": sha256_path(container),
        "source_sha256": source_record["sha256"],
        "audit_sha256": sha256_path(audit),
        "log_sha256": sha256_path(log),
    }
    if receipt.get("id") != entry["id"]:
        raise AssertionError(f"decode receipt identity mismatch: {entry['id']}")
    for field, expected in expected_hashes.items():
        if receipt.get(field) != expected:
            raise AssertionError(
                f"decode receipt {field} mismatch for {entry['id']}"
            )
    argv = receipt.get("canonical_argv")
    if not isinstance(argv, list) or len(argv) != 14 or not all(
        isinstance(value, str) for value in argv
    ):
        raise AssertionError(f"decode receipt argv is malformed: {entry['id']}")
    if Path(argv[1]).resolve() != DECODER.resolve():
        raise AssertionError(f"decode receipt used another decoder: {entry['id']}")
    expected_tail = [
        "--container",
        relative_to_root(container),
        "--container-layout",
        "plte-u20-tail-fp32",
        "--metadata",
        relative_to_root(report),
        "--raw-mask",
        relative_to_root(PROFILE),
        "--source-bf16",
        relative_to_root(source),
        "--output",
        str(audit.with_name(f".{audit.stem}.partial.json")),
    ]
    if argv[2:] != expected_tail:
        raise AssertionError(f"decode receipt argv inputs mismatch: {entry['id']}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    energy = sum(float(row["source_energy"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    distortion = sse / energy
    actual_bits = sum(int(row["container_bytes"]) * 8 for row in rows)
    actual_rate = actual_bits / (BLOCK_VALUES * len(rows))
    actual_gap = 10.0 * math.log10(distortion / (2.0 ** (-2.0 * actual_rate)))
    charged_slot_bits = sum(int(row["charged_slot_bytes"]) * 8 + 4 for row in rows)
    charged_slot_rate = charged_slot_bits / (BLOCK_VALUES * len(rows))
    charged_slot_gap = 10.0 * math.log10(
        distortion / (2.0 ** (-2.0 * charged_slot_rate))
    )
    charged_gaps = [float(row["charged_tier_slot_gap_db"]) for row in rows]
    actual_gaps = [float(row["actual_gap_db"]) for row in rows]
    return {
        "blocks": len(rows),
        "source_energy": energy,
        "sse": sse,
        "energy_weighted_relative_mse": distortion,
        "actual_container_bits": actual_bits,
        "mean_actual_bpw": actual_rate,
        "aggregate_gap_at_mean_actual_rate_db": actual_gap,
        "charged_tier_slot_bits_including_4bit_map": charged_slot_bits,
        "mean_charged_tier_slot_bpw_including_4bit_map": charged_slot_rate,
        "aggregate_gap_at_charged_tier_slot_rate_db": charged_slot_gap,
        "pointwise_actual_gap_db": {
            "min": min(actual_gaps),
            "p50": percentile(actual_gaps, 0.50),
            "p95": percentile(actual_gaps, 0.95),
            "p99": percentile(actual_gaps, 0.99),
            "max": max(actual_gaps),
        },
        "pointwise_charged_tier_slot_gap_db": {
            "min": min(charged_gaps),
            "p50": percentile(charged_gaps, 0.50),
            "p95": percentile(charged_gaps, 0.95),
            "p99": percentile(charged_gaps, 0.99),
            "max": max(charged_gaps),
        },
        "charged_tier_slot_failures_ge_0p10db": sum(
            gap >= 0.10 for gap in charged_gaps
        ),
    }


def build_final_artifacts(
    workspace: Path,
    publish_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    reservoir_plan_path: Path | None,
) -> dict[str, Any]:
    if len(selected) != 400:
        raise AssertionError("publication requires the complete 400-block panel")
    for path, expected in (
        (ENCODER, manifest["provenance"]["encoder_sha256"]),
        (DECODER, manifest["provenance"]["independent_decoder_sha256"]),
        (PROFILE, manifest["provenance"]["frozen_profile_sha256"]),
    ):
        if sha256_path(path) != expected:
            raise AssertionError(f"pinned implementation changed before finalization: {path}")
    manifest_sha256 = sha256_path(manifest_path)
    source_manifest = load_json(workspace / "source_manifest.json")
    source_records = validate_final_source_manifest(
        workspace, manifest_path, manifest, source_manifest
    )
    decode_receipts = load_json(workspace / "decode_status.json")
    if not isinstance(decode_receipts, list) or len(decode_receipts) != len(selected):
        raise AssertionError("decode receipt census is incomplete")
    receipt_by_id = {str(row.get("id")): row for row in decode_receipts}
    if len(receipt_by_id) != len(decode_receipts) or set(receipt_by_id) != {
        str(row["id"]) for row in selected
    }:
        raise AssertionError("decode receipts are duplicate, missing, or unexpected")
    publish_dir.mkdir(parents=True, exist_ok=True)

    if reservoir_plan_path is None:
        reservoir_plan = {
            "format": "implicit uniform Tier-0 plan",
            "post_hoc_engineering_amendment": False,
            "allocations": [
                {
                    "id": entry["id"],
                    "tier": 0,
                    "container_cap_bytes": CONTAINER_CAP_BYTES,
                }
                for entry in selected
            ],
        }
        reservoir_plan_sha256 = None
    else:
        reservoir_plan = load_json(reservoir_plan_path)
        if reservoir_plan.get("format") != "PLTE Qwen3 checkpoint rate-reservoir plan v1":
            raise AssertionError("unexpected reservoir plan format")
        if reservoir_plan.get("post_hoc_engineering_amendment") is not True:
            raise AssertionError("reservoir plan must declare the post-hoc amendment")
        if reservoir_plan.get("selection_manifest_sha256") != sha256_path(manifest_path):
            raise AssertionError("reservoir plan targets a different selection")
        reservoir_plan_sha256 = sha256_path(reservoir_plan_path)
    if len(reservoir_plan["allocations"]) != len(selected):
        raise AssertionError("reservoir allocation count does not match the panel")
    allocation_by_id = {
        str(row["id"]): row for row in reservoir_plan["allocations"]
    }
    if len(allocation_by_id) != len(reservoir_plan["allocations"]):
        raise AssertionError("duplicate reservoir allocation id")
    if set(allocation_by_id) != {str(row["id"]) for row in selected}:
        raise AssertionError("reservoir allocation does not exactly cover the panel")

    results = []
    independent = []
    summary_rows = []
    bundle_temporary = publish_dir / ".containers.partial.bin"
    slots_temporary = publish_dir / ".tiered_slots.partial.bin"
    ordered_selected = sorted(selected, key=lambda row: str(row["id"]))
    recomputed_accounting: dict[str, Any] | None = None
    failure_log_payloads: dict[str, bytes] = {}
    if reservoir_plan_path is not None:
        recomputed_accounting, failure_log_payloads = (
            validate_reservoir_plan_for_finalization(
                workspace,
                reservoir_plan,
                manifest,
                manifest_sha256,
                ordered_selected,
            )
        )
    for entry in ordered_selected:
        allocation = allocation_by_id[str(entry["id"])]
        for key in ("tensor", "block_index", "layer", "role"):
            if allocation.get(key) != entry.get(key):
                raise AssertionError(
                    f"reservoir allocation identity mismatch for {entry['id']}: {key}"
                )
    tiers = [int(allocation_by_id[str(entry["id"])]["tier"]) for entry in ordered_selected]
    if any(tier < 0 or tier > 15 for tier in tiers):
        raise AssertionError("tier index does not fit the four-bit map")
    if reservoir_plan_path is not None:
        policy = reservoir_plan["policy"]
        tier0_bytes = int(policy["tier0_bytes"])
        tier_step_bytes = int(policy["tier_step_bytes"])
        for entry in ordered_selected:
            allocation = allocation_by_id[str(entry["id"])]
            tier = int(allocation["tier"])
            expected_cap = tier0_bytes + tier_step_bytes * tier
            if int(allocation["container_cap_bytes"]) != expected_cap:
                raise AssertionError(
                    f"invalid reservoir cap for {entry['id']}: expected {expected_cap}"
                )
            expected_status = (
                "fits_tier0" if tier == 0 else "recognized_base_container_overflow"
            )
            if allocation.get("first_pass_status") != expected_status:
                raise AssertionError(
                    f"invalid first-pass status for {entry['id']}: {expected_status}"
                )
        first_pass = reservoir_plan["first_pass"]
        tier0_successes = tiers.count(0)
        recognized_overflows = len(tiers) - tier0_successes
        if (
            int(first_pass["attempted_blocks"]) != len(selected)
            or int(first_pass["tier0_successes"]) != tier0_successes
            or int(first_pass["recognized_overflows"]) != recognized_overflows
            or int(first_pass["other_failures"]) != 0
        ):
            raise AssertionError("reservoir first-pass totals do not match allocations")
    tier_map = bytes(
        tiers[index] | ((tiers[index + 1] if index + 1 < len(tiers) else 0) << 4)
        for index in range(0, len(tiers), 2)
    )
    tier_map_path = publish_dir / "tier_map.bin"
    atomic_write_bytes(tier_map_path, tier_map)
    failure_log_dir = publish_dir / "original_tier0_failure_logs"
    if failure_log_payloads:
        failure_log_dir.mkdir(parents=True, exist_ok=True)
        for entry_id, payload in failure_log_payloads.items():
            atomic_write_bytes(failure_log_dir / f"{entry_id}.txt", payload)
        expected_log_names = {
            f"{entry_id}.txt" for entry_id in failure_log_payloads
        }
        actual_log_names = {
            path.name for path in failure_log_dir.iterdir() if path.is_file()
        }
        if actual_log_names != expected_log_names:
            raise AssertionError("published Tier-0 failure-log set is stale or incomplete")
    original_failures = []
    for entry in ordered_selected:
        allocation = allocation_by_id[str(entry["id"])]
        if int(allocation["tier"]) == 0:
            continue
        original_failures.append(
            {
                "id": entry["id"],
                "tensor": entry["tensor"],
                "block_index": entry["block_index"],
                "layer": entry["layer"],
                "role": entry["role"],
                "failure": "recognized_base_container_overflow",
                "tier0_cap_bytes": int(reservoir_plan["policy"]["tier0_bytes"]),
                "base_container_bytes": int(
                    allocation["first_pass_base_container_bytes"]
                ),
                "overflow_bytes": int(allocation["overflow_bytes_above_tier0"]),
                "first_pass_log_sha256": allocation["first_pass_log_sha256"],
                "published_log": (
                    f"original_tier0_failure_logs/{entry['id']}.txt"
                ),
                "published_log_sha256": sha256_path(
                    failure_log_dir / f"{entry['id']}.txt"
                ),
            }
        )
    first_pass = reservoir_plan.get("first_pass", {})
    original_tier0_outcome = {
        "format": "PLTE Qwen3 original Tier-0 outcome v1",
        "selection_manifest_sha256": sha256_path(manifest_path),
        "reservoir_plan_sha256": reservoir_plan_sha256,
        "endpoint": "every selected block fits the original 81,242-byte cap",
        "passes": len(original_failures) == 0,
        "attempted_blocks": len(selected),
        "tier0_successes": tiers.count(0),
        "recognized_cap_failures": len(original_failures),
        "other_failures": int(first_pass.get("other_failures", 0)),
        "maximum_base_container_bytes": first_pass.get(
            "maximum_base_container_bytes"
        ),
        "maximum_overflow_bytes": first_pass.get("maximum_overflow_bytes"),
        "maximum_valid_tier0_gap_db": first_pass.get(
            "maximum_valid_tier0_gap_db"
        ),
        "failures": original_failures,
        "claim_boundary": (
            "This is the immutable outcome of the original fixed-cap endpoint. "
            "Later reservoir retries do not reclassify these failures."
        ),
    }
    original_outcome_path = publish_dir / "original_tier0_outcome.json"
    atomic_write_json(original_outcome_path, original_tier0_outcome)
    offset = 0
    slot_offset = 0
    with bundle_temporary.open("wb") as bundle, slots_temporary.open("wb") as slots:
        for entry in ordered_selected:
            allocation = allocation_by_id[str(entry["id"])]
            tier = int(allocation["tier"])
            charged_slot_bytes = int(allocation["container_cap_bytes"])
            report_path, container_path, _ = encoder_paths(workspace, entry)
            audit_path, _ = audit_paths(workspace, entry)
            report = load_json(report_path)
            audit = load_json(audit_path)
            source_record = source_records[entry["id"]]
            validate_encoder_result(
                report_path,
                container_path,
                entry,
                source_record,
                manifest["provenance"]["encoder_sha256"],
                charged_slot_bytes,
            )
            validate_decode_audit(
                audit_path, entry, source_record, report["trials"][0]
            )
            validate_decode_receipt(
                workspace,
                entry,
                source_record,
                receipt_by_id[str(entry["id"])],
                manifest,
            )
            container = container_path.read_bytes()
            bundle.write(container)
            padding_bytes = charged_slot_bytes - len(container)
            if padding_bytes < 0:
                raise AssertionError("literal container exceeds charged reservoir tier")
            slots.write(container)
            slots.write(b"\0" * padding_bytes)
            trial = report["trials"][0]
            if tier == 0 and reservoir_plan_path is not None:
                if allocation["report_sha256"] != sha256_path(report_path):
                    raise AssertionError("Tier-0 artifact changed after reservoir planning")
                if (
                    allocation["container_sha256"] != sha256_bytes(container)
                    or allocation["container_bytes"] != len(container)
                ):
                    raise AssertionError(
                        "Tier-0 container changed after reservoir planning"
                    )
            if tier > 0 and trial["base_literal_container_bytes"] != allocation.get(
                "first_pass_base_container_bytes"
            ):
                raise AssertionError("reservoir retry changed the deterministic base length")
            rms = float(trial["source"]["block_rms_fp64"])
            source_energy = rms * rms * BLOCK_VALUES
            sse = float(trial["literal_decoded_absolute_mse"]) * BLOCK_VALUES
            actual_rate = len(container) * 8 / BLOCK_VALUES
            relative_mse = sse / source_energy
            actual_gap = 10.0 * math.log10(relative_mse / (2.0 ** (-2.0 * actual_rate)))
            charged_slot_bits = (
                8 * charged_slot_bytes + TIER_MAP_BITS_PER_BLOCK
            )
            charged_slot_rate = charged_slot_bits / BLOCK_VALUES
            charged_slot_gap = 10.0 * math.log10(
                relative_mse / (2.0 ** (-2.0 * charged_slot_rate))
            )
            results.append(
                {
                    "id": entry["id"],
                    "tensor": entry["tensor"],
                    "block_index": entry["block_index"],
                    "layer": entry["layer"],
                    "role": entry["role"],
                    "source_sha256": source_record["sha256"],
                    "report_sha256": sha256_path(report_path),
                    "container_offset": offset,
                    "container_bytes": len(container),
                    "container_sha256": sha256_bytes(container),
                    "reservoir_tier": tier,
                    "charged_slot_bytes": charged_slot_bytes,
                    "charged_slot_bits_including_4bit_map": charged_slot_bits,
                    "charged_slot_bpw_including_4bit_map": charged_slot_rate,
                    "tiered_slot_offset": slot_offset,
                    "zero_padding_bytes": padding_bytes,
                    "report": report,
                }
            )
            independent.append(
                {
                    "id": entry["id"],
                    "audit_sha256": sha256_path(audit_path),
                    "audit": audit,
                    "receipt": receipt_by_id[str(entry["id"])],
                }
            )
            summary_rows.append(
                {
                    "id": entry["id"],
                    "layer": entry["layer"],
                    "role": entry["role"],
                    "source_energy": source_energy,
                    "sse": sse,
                    "relative_mse": relative_mse,
                    "container_bytes": len(container),
                    "actual_bpw": actual_rate,
                    "actual_gap_db": actual_gap,
                    "reservoir_tier": tier,
                    "charged_slot_bytes": charged_slot_bytes,
                    "charged_slot_bits_including_4bit_map": charged_slot_bits,
                    "charged_slot_bpw_including_4bit_map": charged_slot_rate,
                    "charged_tier_slot_gap_db": charged_slot_gap,
                }
            )
            offset += len(container)
            slot_offset += charged_slot_bytes
    bundle_path = publish_dir / "containers.polar.bin"
    os.replace(bundle_temporary, bundle_path)
    slots_path = publish_dir / "tiered_slots.bin"
    os.replace(slots_temporary, slots_path)
    packing_readback = audit_packed_artifacts(
        bundle_path, tier_map_path, slots_path, results
    )

    rank1 = audit_rank1(workspace, manifest, source_records)
    routers = validate_router_census(manifest)
    by_role = {}
    for role in sorted({str(row["role"]) for row in summary_rows}):
        by_role[role] = summarize_rows([row for row in summary_rows if row["role"] == role])
    by_layer = {}
    for layer in range(48):
        by_layer[str(layer)] = summarize_rows(
            [row for row in summary_rows if row["layer"] == layer]
        )
    failures = [
        row for row in summary_rows if row["charged_tier_slot_gap_db"] >= 0.10
    ]
    overall = summarize_rows(summary_rows)
    summary = {
        "format": "PLTE Qwen3 stratified evaluation summary v1",
        "checkpoint": manifest["checkpoint"],
        "strict_ptq": True,
        "selection_manifest_sha256": sha256_path(manifest_path),
        "selection_was_held_out_from_previous_evidence": True,
        "post_hoc_engineering_amendment": bool(
            reservoir_plan.get("post_hoc_engineering_amendment")
        ),
        "reservoir_plan_sha256": reservoir_plan_sha256,
        "coverage": {
            "new_plte_blocks": 400,
            "layer_role_cells": 336,
            "layers": 48,
            "plte_layer_roles": 7,
            "embedding_blocks": 32,
            "lm_head_blocks": 32,
            "router_census_blocks": 48,
            "rank1_census_tensors": 193,
            "prior_plus_new_unique_plte_blocks": 447,
        },
        "integrity": {
            "runner_sha256": sha256_path(Path(__file__)),
            "encoder_sha256": manifest["provenance"]["encoder_sha256"],
            "independent_decoder_sha256": manifest["provenance"][
                "independent_decoder_sha256"
            ],
            "frozen_profile_sha256": manifest["provenance"]["frozen_profile_sha256"],
            "all_encoder_internal_decodes_passed": True,
            "independent_clean_decodes": len(independent),
            "all_independent_clean_decodes_passed": True,
            "all_decode_receipts_bound": True,
            "container_bundle_bytes": bundle_path.stat().st_size,
            "container_bundle_sha256": sha256_path(bundle_path),
            "tier_map_bytes": len(tier_map),
            "tier_map_sha256": sha256_bytes(tier_map),
            "tiered_slots_bytes": slots_path.stat().st_size,
            "tiered_slots_sha256": sha256_path(slots_path),
            "all_tier_padding_zero": packing_readback[
                "all_slot_padding_bytes_zero"
            ],
            "packed_artifact_readback": packing_readback,
        },
        "plte_panel": overall,
        "quality_endpoint": {
            "definition": (
                "every new block has all-in charged reservoir-tier slot gap "
                "including its four-bit map charge < 0.10 dB"
            ),
            "passes": len(failures) == 0,
            "failures": failures,
        },
        "original_tier0_endpoint": {
            **original_tier0_outcome,
            "artifact": original_outcome_path.name,
            "artifact_sha256": sha256_path(original_outcome_path),
        },
        "by_role": by_role,
        "by_layer": by_layer,
        "routers": routers,
        "reservoir": {
            "format": reservoir_plan.get("format"),
            "plan_sha256": reservoir_plan_sha256,
            "tier_counts": {
                str(tier): tiers.count(tier) for tier in sorted(set(tiers))
            },
            "maximum_tier": max(tiers),
            "panel_tier_map_order": "results sorted by id; low nibble first",
            "panel_sum_tiers": sum(tiers),
            "checkpoint_rate_accounting": recomputed_accounting,
        },
        "rank1": {
            key: value for key, value in rank1.items() if key != "tensors"
        },
        "claim_boundary": (
            "Measured on the frozen 400-block panel with a post-hoc deterministic "
            "rate-reservoir amendment, plus complete router and rank-one exception "
            "censuses; not an untouched confirmatory holdout, whole-checkpoint "
            "distortion measurement, or worst-case guarantee"
        ),
    }
    source_publish = {
        **source_manifest,
        "selection_manifest_sha256": sha256_path(manifest_path),
    }
    atomic_write_json(
        publish_dir / "results.json",
        {
            "format": "PLTE Qwen3 stratified encoder results v1",
            "selection_manifest_sha256": sha256_path(manifest_path),
            "container_bundle": bundle_path.name,
            "container_bundle_bytes": bundle_path.stat().st_size,
            "container_bundle_sha256": sha256_path(bundle_path),
            "tier_map": "tier_map.bin",
            "tier_map_bytes": len(tier_map),
            "tier_map_sha256": sha256_bytes(tier_map),
            "tiered_slots": slots_path.name,
            "tiered_slots_bytes": slots_path.stat().st_size,
            "tiered_slots_sha256": sha256_path(slots_path),
            "packed_artifact_readback": packing_readback,
            "original_tier0_outcome": original_outcome_path.name,
            "original_tier0_outcome_sha256": sha256_path(original_outcome_path),
            "results": results,
        },
    )
    atomic_write_json(
        publish_dir / "independent_decodes.json",
        {
            "format": "PLTE Qwen3 stratified independent decode audit v1",
            "decoder_sha256": manifest["provenance"]["independent_decoder_sha256"],
            "audits": independent,
        },
    )
    atomic_write_json(publish_dir / "source_hashes.json", source_publish)
    atomic_write_json(publish_dir / "rank1_exact_audit.json", rank1)
    atomic_write_json(publish_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument(
        "--reservoir-plan",
        type=Path,
        help="optional frozen mixed-tier plan used by finalization",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    parser.add_argument("--fetch-workers", type=int, default=16)
    parser.add_argument("--encode-workers", type=int, default=8)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument(
        "--phases",
        default="fetch,encode,decode,finalize",
        help="comma-separated subset of fetch,encode,decode,finalize",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N scheduled PLTE rows; finalize is then forbidden",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    validate_manifest(manifest, manifest_path)
    phases = {part.strip() for part in args.phases.split(",") if part.strip()}
    unknown = phases - {"fetch", "encode", "decode", "finalize"}
    if unknown:
        raise ValueError(f"unknown phases: {sorted(unknown)}")
    if args.limit is not None and (args.limit <= 0 or args.limit > 400):
        raise ValueError("--limit must be between 1 and 400")
    if args.limit is not None and "finalize" in phases:
        raise ValueError("cannot finalize a limited run")
    workspace = args.workspace.resolve()
    relative_to_root(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    selected = scheduled_entries(manifest["plte_blocks"])
    if args.limit is not None:
        selected = selected[: args.limit]
    publish_dir = (
        args.publish_dir.resolve()
        if args.publish_dir
        else manifest_path.parent
    )

    for path, expected in (
        (ENCODER, manifest["provenance"]["encoder_sha256"]),
        (DECODER, manifest["provenance"]["independent_decoder_sha256"]),
        (PROFILE, manifest["provenance"]["frozen_profile_sha256"]),
    ):
        actual = sha256_path(path)
        if actual != expected:
            raise AssertionError(f"pinned file hash mismatch: {path}: {actual} != {expected}")

    source_manifest_path = workspace / "source_manifest.json"
    if "fetch" in phases:
        source_manifest = fetch_sources(
            workspace,
            manifest_path,
            manifest,
            selected,
            args.fetch_workers,
            include_rank1=args.limit is None,
        )
    else:
        source_manifest = load_json(source_manifest_path)
    source_records = {row["id"]: row for row in source_manifest["records"]}

    if "encode" in phases:
        encoded, failures = parallel_map(
            "encode",
            selected,
            args.encode_workers,
            lambda entry: encode_one(
                workspace,
                entry,
                source_records[entry["id"]],
                args.python,
                args.polar_repo,
                manifest["provenance"]["encoder_sha256"],
            ),
        )
        atomic_write_json(workspace / "encode_status.json", encoded)
        if failures:
            atomic_write_json(workspace / "encode_failures.json", failures)
            raise RuntimeError(f"encoding failed for {len(failures)} entries")

    if "decode" in phases:
        decoded, failures = parallel_map(
            "independent decode",
            selected,
            args.decode_workers,
            lambda entry: decode_one(
                workspace,
                entry,
                source_records[entry["id"]],
                args.python,
            ),
        )
        atomic_write_json(
            workspace / "decode_status.json",
            sorted(decoded, key=lambda row: str(row["id"])),
        )
        if failures:
            atomic_write_json(workspace / "decode_failures.json", failures)
            raise RuntimeError(f"independent decoding failed for {len(failures)} entries")

    if "finalize" in phases:
        summary = build_final_artifacts(
            workspace,
            publish_dir,
            manifest_path,
            manifest,
            selected,
            args.reservoir_plan.resolve() if args.reservoir_plan else None,
        )
        progress(json.dumps(summary["plte_panel"], indent=2))
        progress(f"quality endpoint passes: {summary['quality_endpoint']['passes']}")


if __name__ == "__main__":
    main()
