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
BASE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{shard}?download=true"

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
    if not math.isfinite(float(source["block_rms_fp64"])) or float(source["block_rms_fp64"]) <= 0:
        raise AssertionError("invalid source RMS")
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
    return {
        "id": entry["id"],
        "audit_sha256": sha256_path(audit_path),
        "container_sha256": audit["container_sha256"],
        "relative_mse": audit["decoded_relative_mse_with_serialized_scale"],
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
        return validate_decode_audit(audit, entry, source_record, encoder_trial)
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
        decoded = bytes(payload)
        values = (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16)).view(
            np.float32
        )
        energy = float(np.sum(values.astype(np.float64) ** 2))
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
        total_values += int(entry["values"])
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
    if report.get("router_count") != 48 or len(report.get("routers", [])) != 48:
        raise AssertionError("router report is not a complete 48-layer census")
    if report.get("checkpoint", None) not in (None, manifest["checkpoint"]):
        raise AssertionError("router checkpoint mismatch")
    if report.get("repo") != manifest["checkpoint"]["repo"]:
        raise AssertionError("router repository mismatch")
    if report.get("revision") != manifest["checkpoint"]["revision"]:
        raise AssertionError("router revision mismatch")
    if not all(row.get("inverse_decode_exact") is True for row in report["routers"]):
        raise AssertionError("router inverse-decode audit failed")
    aggregate = report["aggregate"]
    if aggregate["container_sha256"] != sha256_path(ROUTER_CONTAINER):
        raise AssertionError("router container SHA-256 mismatch")
    if aggregate["container_bytes"] != ROUTER_CONTAINER.stat().st_size:
        raise AssertionError("router container size mismatch")
    independent = report["independent_literal_decode"]
    if independent.get("exact_file_length") is not True:
        raise AssertionError("router independent decoder did not consume exact file")
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
    source_manifest = load_json(workspace / "source_manifest.json")
    source_records = {row["id"]: row for row in source_manifest["records"]}
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
        if reservoir_plan.get("selection_manifest_sha256") != sha256_path(manifest_path):
            raise AssertionError("reservoir plan targets a different selection")
        reservoir_plan_sha256 = sha256_path(reservoir_plan_path)
    allocation_by_id = {
        str(row["id"]): row for row in reservoir_plan["allocations"]
    }
    if set(allocation_by_id) != {str(row["id"]) for row in selected}:
        raise AssertionError("reservoir allocation does not exactly cover the panel")

    results = []
    independent = []
    summary_rows = []
    bundle_temporary = publish_dir / ".containers.partial.bin"
    slots_temporary = publish_dir / ".tiered_slots.partial.bin"
    ordered_selected = sorted(selected, key=lambda row: str(row["id"]))
    tiers = [int(allocation_by_id[str(entry["id"])]["tier"]) for entry in ordered_selected]
    if any(tier < 0 or tier > 15 for tier in tiers):
        raise AssertionError("tier index does not fit the four-bit map")
    tier_map = bytes(
        tiers[index] | ((tiers[index + 1] if index + 1 < len(tiers) else 0) << 4)
        for index in range(0, len(tiers), 2)
    )
    atomic_write_bytes(publish_dir / "tier_map.bin", tier_map)
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
            container = container_path.read_bytes()
            bundle.write(container)
            padding_bytes = charged_slot_bytes - len(container)
            if padding_bytes < 0:
                raise AssertionError("literal container exceeds charged reservoir tier")
            slots.write(container)
            slots.write(b"\0" * padding_bytes)
            trial = report["trials"][0]
            if tier == 0 and "report_sha256" in allocation:
                if allocation["report_sha256"] != sha256_path(report_path):
                    raise AssertionError("Tier-0 artifact changed after reservoir planning")
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
            charged_slot_rate = 8 * charged_slot_bytes / BLOCK_VALUES
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
                    "charged_tier_slot_gap_db": charged_slot_gap,
                }
            )
            offset += len(container)
            slot_offset += charged_slot_bytes
    bundle_path = publish_dir / "containers.polar.bin"
    os.replace(bundle_temporary, bundle_path)
    slots_path = publish_dir / "tiered_slots.bin"
    os.replace(slots_temporary, slots_path)

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
            "container_bundle_bytes": bundle_path.stat().st_size,
            "container_bundle_sha256": sha256_path(bundle_path),
            "tier_map_bytes": len(tier_map),
            "tier_map_sha256": sha256_bytes(tier_map),
            "tiered_slots_bytes": slots_path.stat().st_size,
            "tiered_slots_sha256": sha256_path(slots_path),
            "all_tier_padding_zero": True,
        },
        "plte_panel": overall,
        "quality_endpoint": {
            "definition": "every new block has charged reservoir-tier slot gap < 0.10 dB",
            "passes": len(failures) == 0,
            "failures": failures,
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
            "checkpoint_rate_accounting": reservoir_plan.get(
                "checkpoint_rate_accounting"
            ),
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
        atomic_write_json(workspace / "decode_status.json", decoded)
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
