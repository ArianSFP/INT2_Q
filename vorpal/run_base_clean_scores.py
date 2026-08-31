#!/usr/bin/env python3
"""Resumable, fail-closed clean scorer for the 400 base waterfill chunks.

This runner never writes beneath ``--base-dir`` and never changes the manifest,
normalized sources, raw Qwen sources, encoder reports, or PLTE containers.  It
invokes ``decode_continuous_chunk.py`` into a private temporary output, validates
the result and every relevant input hash, and then atomically publishes a bound
per-chunk score envelope in ``--output-dir``.

A partial invocation is useful while the base encoder is still running.  A
``complete`` receipt is impossible until all 400 canonical jobs are present and
the encoder's canonical ``run.receipt.json`` independently cross-validates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_FORMAT = "continuous reverse-waterfilled PLTE exploratory manifest v1"
RUN_FORMAT = "continuous reverse-waterfilled PLTE base panel run v1"
CHUNK_SCORE_FORMAT = "continuous reverse-waterfilled PLTE clean chunk decode v1"
ENVELOPE_FORMAT = "continuous reverse-waterfilled PLTE bound base clean score v1"
RECEIPT_FORMAT = "continuous reverse-waterfilled PLTE base clean panel score v1"
BLOCKS = 400
GROUPS_PER_BLOCK = 128
GROUP_VALUES = 1 << 11
BLOCK_VALUES = 1 << 18
GROUPS = BLOCKS * GROUPS_PER_BLOCK
PANEL_VALUES = BLOCKS * BLOCK_VALUES
BASE_LEVELS = 6
HASH_HEX = 64
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
PRINT_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


class ValidationError(RuntimeError):
    """An artifact failed a claim-relevant validation."""


class StopRequested(RuntimeError):
    """A cooperative SIGINT/SIGTERM stop was requested."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class BaseJob:
    chunk_index: int
    report: FileBinding
    container: FileBinding
    normalized_source: FileBinding
    logical_bits: int
    tail_escape_count: int
    encoder_relative_mse: float


@dataclass(frozen=True)
class Provenance:
    manifest_sha256: str
    encoder_sha256: str
    chunk_decoder_sha256: str
    clean_decoder_sha256: str
    raw_mask_sha256: str
    original_source_hash_digest: str
    normalized_source_hash_digest: str


def progress(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{threading.get_ident()}"
    temporary = path.parent / f".{path.name}.{token}.partial"
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary output {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def require_hash(value: object, description: str) -> str:
    text = str(value).lower()
    if len(text) != HASH_HEX:
        raise ValidationError(f"{description} is not a 64-digit SHA256")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValidationError(f"{description} is not hexadecimal") from error
    return text


def exact_int(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{description} must be an integer")
    return value


def finite_float(value: object, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{description} must not be Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{description} is not numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValidationError(f"{description} is not finite and positive")
    return result


def same_float(left: object, right: object, description: str, tolerance: float = 0.0) -> float:
    lhs = finite_float(left, description)
    rhs = finite_float(right, description)
    if not math.isclose(lhs, rhs, rel_tol=tolerance, abs_tol=tolerance):
        raise ValidationError(f"{description} differs: {lhs!r} != {rhs!r}")
    return lhs


def resolve_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def bind_file(path: Path, description: str, expected_bytes: int | None = None) -> FileBinding:
    if not path.is_file():
        raise ValidationError(f"missing {description}: {path}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValidationError(
            f"{description} has {size} bytes, expected {expected_bytes}: {path}"
        )
    return FileBinding(path=path, bytes=size, sha256=sha256_path(path))


def assert_expected_file(
    path: Path, expected_hash: str, description: str, expected_bytes: int | None = None
) -> FileBinding:
    binding = bind_file(path, description, expected_bytes)
    if binding.sha256 != require_hash(expected_hash, f"expected {description} hash"):
        raise ValidationError(
            f"{description} SHA256 mismatch: {binding.sha256} != {expected_hash}"
        )
    return binding


def validate_manifest(manifest: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValidationError("unsupported canonical manifest format")
    if manifest.get("strict_ptq") is not True or manifest.get("training_or_retraining") is not False:
        raise ValidationError("manifest is not strict PTQ without retraining")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationError("manifest parameters are missing")
    expected_parameters = {
        "block_values": BLOCK_VALUES,
        "group_values": GROUP_VALUES,
        "groups_per_polar_block": GROUPS_PER_BLOCK,
        "sigma_source": 3.0,
        "stable_order": "ascending reconstructed qvariance, then canonical group ordinal",
    }
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise ValidationError(f"noncanonical manifest parameter {key!r}")
    census = manifest.get("census")
    if census != {
        "source_blocks": BLOCKS,
        "groups": GROUPS,
        "polar_chunks": BLOCKS,
        "values": PANEL_VALUES,
    }:
        raise ValidationError("noncanonical manifest census")

    blocks = manifest.get("blocks")
    chunks = manifest.get("chunks")
    if not isinstance(blocks, list) or len(blocks) != BLOCKS:
        raise ValidationError("manifest must contain 400 canonical source blocks")
    if not isinstance(chunks, list) or len(chunks) != BLOCKS:
        raise ValidationError("manifest must contain 400 canonical polar chunks")
    ids: set[str] = set()
    for ordinal, block in enumerate(blocks):
        if not isinstance(block, dict) or exact_int(block.get("ordinal"), "block ordinal") != ordinal:
            raise ValidationError(f"block {ordinal} is not in canonical ordinal order")
        identifier = str(block.get("id", ""))
        if not identifier or identifier in ids:
            raise ValidationError(f"block {ordinal} has an empty or repeated id")
        ids.add(identifier)
        require_hash(block.get("source_sha256"), f"block {ordinal} source hash")
        if not str(block.get("source_path", "")):
            raise ValidationError(f"block {ordinal} has no source path")
        labels = block.get("labels_i8")
        if not isinstance(labels, list) or len(labels) != GROUPS_PER_BLOCK:
            raise ValidationError(f"block {ordinal} does not have 128 labels")
        if any(not isinstance(label, int) or isinstance(label, bool) or not -32 <= label <= 31 for label in labels):
            raise ValidationError(f"block {ordinal} contains a non-six-bit label")
        finite_float(block.get("serialized_rms_fp32"), f"block {ordinal} RMS", positive=True)

    seen = bytearray(GROUPS)
    previous_key: tuple[float, int] | None = None
    a128 = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or exact_int(chunk.get("chunk_index"), "chunk index") != index:
            raise ValidationError(f"chunk {index} is not in canonical order")
        require_hash(chunk.get("normalized_source_sha256"), f"chunk {index} normalized hash")
        if not str(chunk.get("normalized_source", "")):
            raise ValidationError(f"chunk {index} has no normalized source path")
        alphabet_size = exact_int(chunk.get("alphabet_size"), f"chunk {index} alphabet")
        if alphabet_size not in (64, 128):
            raise ValidationError(f"chunk {index} has unsupported base alphabet {alphabet_size}")
        a128 += alphabet_size == 128
        finite_float(chunk.get("test_distortion"), f"chunk {index} distortion", positive=True)
        finite_float(chunk.get("eta"), f"chunk {index} eta", positive=True)
        finite_float(chunk.get("nominal_rate"), f"chunk {index} nominal rate", positive=True)
        members = chunk.get("members")
        if not isinstance(members, list) or len(members) != GROUPS_PER_BLOCK:
            raise ValidationError(f"chunk {index} does not have 128 members")
        for member in members:
            canonical = exact_int(member.get("canonical_group_ordinal"), "canonical group ordinal")
            block_ordinal = exact_int(member.get("block_ordinal"), "member block ordinal")
            group_index = exact_int(member.get("group_index"), "member group index")
            if not 0 <= canonical < GROUPS or not 0 <= block_ordinal < BLOCKS or not 0 <= group_index < GROUPS_PER_BLOCK:
                raise ValidationError(f"chunk {index} has out-of-range membership")
            if canonical != block_ordinal * GROUPS_PER_BLOCK + group_index:
                raise ValidationError(f"chunk {index} has inconsistent canonical membership")
            if seen[canonical]:
                raise ValidationError(f"canonical group {canonical} is repeated")
            seen[canonical] = 1
            label = exact_int(member.get("label"), "member label")
            if label != blocks[block_ordinal]["labels_i8"][group_index]:
                raise ValidationError(f"canonical group {canonical} label disagrees with block ledger")
            qscale = finite_float(member.get("qscale"), "member qscale", positive=True)
            qvariance = finite_float(member.get("qvariance"), "member qvariance", positive=True)
            if not math.isclose(qvariance, qscale * qscale, rel_tol=2e-15, abs_tol=0.0):
                raise ValidationError(f"canonical group {canonical} qvariance is not qscale squared")
            expected_qscale = float(blocks[block_ordinal]["serialized_rms_fp32"]) * math.exp2(label / 16.0)
            if not math.isclose(qscale, expected_qscale, rel_tol=2e-15, abs_tol=0.0):
                raise ValidationError(f"canonical group {canonical} qscale is not reconstructible")
            key = (qvariance, canonical)
            if previous_key is not None and key < previous_key:
                raise ValidationError("manifest membership is not in stable qvariance order")
            previous_key = key
    if any(value != 1 for value in seen):
        raise ValidationError("manifest does not cover every canonical group exactly once")
    if a128 != 61:
        raise ValidationError(f"canonical base manifest requires 61 A128 chunks, got {a128}")
    return blocks, chunks


def verify_data_files(
    blocks: list[dict[str, Any]], chunks: list[dict[str, Any]], repo: Path
) -> tuple[str, str]:
    source_digest = hashlib.sha256()
    normalized_digest = hashlib.sha256()
    for ordinal, block in enumerate(blocks):
        path = resolve_path(repo, block["source_path"])
        binding = assert_expected_file(
            path,
            str(block["source_sha256"]),
            f"source block {ordinal}",
            BLOCK_VALUES * 2,
        )
        source_digest.update(ordinal.to_bytes(4, "little"))
        source_digest.update(bytes.fromhex(binding.sha256))
    for index, chunk in enumerate(chunks):
        path = resolve_path(repo, chunk["normalized_source"])
        binding = assert_expected_file(
            path,
            str(chunk["normalized_source_sha256"]),
            f"normalized source {index}",
            BLOCK_VALUES * 2,
        )
        normalized_digest.update(index.to_bytes(4, "little"))
        normalized_digest.update(bytes.fromhex(binding.sha256))
    return source_digest.hexdigest(), normalized_digest.hexdigest()


def parse_container(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValidationError(f"container is shorter than its header: {path}")
    header_word, scale = struct.unpack_from("<If", payload, 0)
    logical_bits = header_word & ((1 << 20) - 1)
    escape_count = header_word >> 20
    arithmetic_bytes = (logical_bits + 7) // 8
    tail_bytes = (34 * escape_count + 7) // 8
    expected = 8 + arithmetic_bytes + tail_bytes
    if len(payload) != expected:
        raise ValidationError(f"container {path} has {len(payload)} bytes, expected {expected}")
    arithmetic = payload[8 : 8 + arithmetic_bytes]
    tail = payload[8 + arithmetic_bytes :]
    arithmetic_padding = arithmetic_bytes * 8 - logical_bits
    if arithmetic_padding and arithmetic[-1] & ((1 << arithmetic_padding) - 1):
        raise ValidationError(f"container {path} has nonzero arithmetic padding")
    tail_padding = tail_bytes * 8 - 34 * escape_count
    if tail_padding and tail[-1] & ((1 << tail_padding) - 1):
        raise ValidationError(f"container {path} has nonzero sparse-tail padding")
    if escape_count > BLOCK_VALUES or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValidationError(f"container {path} has invalid header values")
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "logical_bits": logical_bits,
        "escape_count": escape_count,
        "arithmetic_payload_sha256": hashlib.sha256(arithmetic).hexdigest(),
    }


def inspect_base_job(
    base_dir: Path,
    chunk: dict[str, Any],
    expected_encoder_sha256: str,
    repo: Path = Path("."),
) -> tuple[str, BaseJob | None, str | None]:
    index = exact_int(chunk.get("chunk_index"), "chunk index")
    stem = f"wf-{index:03d}"
    report_path = base_dir / f"{stem}.json"
    container_path = base_dir / f"{stem}.polar.bin"
    if not report_path.exists() or not container_path.exists():
        return "pending", None, None
    try:
        report_binding = bind_file(report_path, f"base report {index}")
        container_binding = bind_file(container_path, f"base container {index}")
        report = load_json(report_path)
        if not isinstance(report, dict):
            raise ValidationError("base report is not an object")
        if report.get("strict_ptq") is not True or report.get("source_training_or_retraining") is not False:
            raise ValidationError("base report is not strict PTQ")
        if require_hash(report.get("implementation_sha256"), "encoder implementation hash") != expected_encoder_sha256:
            raise ValidationError("base report used the wrong encoder implementation")
        parameters = report.get("parameters")
        trials = report.get("trials")
        if not isinstance(parameters, dict) or not isinstance(trials, list) or len(trials) != 1:
            raise ValidationError("base report parameters/trial census is invalid")
        if exact_int(parameters.get("block_length"), "encoder block length") != BLOCK_VALUES:
            raise ValidationError("base report block length is noncanonical")
        if exact_int(parameters.get("trials"), "encoder trial count") != 1:
            raise ValidationError("base report trial count is noncanonical")
        if exact_int(parameters.get("container_cap_bytes"), "container cap") != 0:
            raise ValidationError("base report is not the uncapped base job")
        if exact_int(parameters.get("alphabet_size"), "alphabet size") != int(chunk["alphabet_size"]):
            raise ValidationError("base report alphabet disagrees with manifest")
        same_float(parameters.get("sigma_source"), 3.0, "sigma source")
        same_float(parameters.get("test_channel_distortion"), chunk["test_distortion"], "test distortion")
        same_float(parameters.get("eta"), chunk["eta"], "eta")
        trial = trials[0]
        if not isinstance(trial, dict) or exact_int(trial.get("trial"), "trial ordinal") != 0:
            raise ValidationError("base report trial ordinal is noncanonical")
        source = trial.get("source")
        if not isinstance(source, dict):
            raise ValidationError("base report source binding is missing")
        normalized_path = resolve_path(repo, chunk["normalized_source"])
        if Path(str(source.get("path", ""))).resolve() != normalized_path.resolve():
            raise ValidationError("base report normalized source path disagrees with manifest")
        if require_hash(source.get("block_bf16_sha256"), "base normalized source hash") != str(chunk["normalized_source_sha256"]):
            raise ValidationError("base report normalized source hash disagrees with manifest")
        if exact_int(source.get("values"), "base normalized source values") != BLOCK_VALUES:
            raise ValidationError("base normalized source geometry is invalid")
        normalized_binding = assert_expected_file(
            normalized_path,
            str(chunk["normalized_source_sha256"]),
            f"normalized source {index}",
            BLOCK_VALUES * 2,
        )
        parsed = parse_container(container_path)
        if parsed["bytes"] != container_binding.bytes or parsed["sha256"] != container_binding.sha256:
            raise AssertionError("container changed during validation")
        if exact_int(trial.get("literal_container_bytes"), "literal container bytes") != container_binding.bytes:
            raise ValidationError("base report container byte count disagrees")
        if require_hash(trial.get("literal_container_sha256"), "literal container hash") != container_binding.sha256:
            raise ValidationError("base report container hash disagrees")
        if exact_int(trial.get("arithmetic_logical_bits"), "arithmetic logical bits") != parsed["logical_bits"]:
            raise ValidationError("base report logical bit count disagrees")
        if require_hash(trial.get("arithmetic_payload_sha256"), "arithmetic payload hash") != parsed["arithmetic_payload_sha256"]:
            raise ValidationError("base report arithmetic payload hash disagrees")
        if exact_int(trial.get("tail_escape_count"), "tail escape count") != parsed["escape_count"]:
            raise ValidationError("base report tail escape count disagrees")
        if any(trial.get(field) is not True for field in ROUNDTRIP_FIELDS):
            raise ValidationError("base report does not pass every internal roundtrip")
        if trial.get("passes_container_cap") is not True:
            raise ValidationError("base report does not pass its uncapped container check")
        relative_mse = finite_float(trial.get("relative_mse"), "encoder relative MSE", positive=True)
        return (
            "ready",
            BaseJob(
                chunk_index=index,
                report=report_binding,
                container=container_binding,
                normalized_source=normalized_binding,
                logical_bits=parsed["logical_bits"],
                tail_escape_count=parsed["escape_count"],
                encoder_relative_mse=relative_mse,
            ),
            None,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
        return "invalid", None, f"{type(error).__name__}: {error}"


def validate_run_receipt(
    path: Path,
    manifest_sha256: str,
    encoder_sha256: str,
    jobs: dict[int, BaseJob],
    chunks: list[dict[str, Any]],
) -> FileBinding:
    binding = bind_file(path, "base run receipt")
    receipt = load_json(path)
    if not isinstance(receipt, dict) or receipt.get("format") != RUN_FORMAT:
        raise ValidationError("unsupported base run receipt format")
    if receipt.get("status") != "complete" or receipt.get("all_internal_roundtrips_passed") is not True:
        raise ValidationError("base run receipt is not complete and passed")
    if require_hash(receipt.get("manifest_sha256"), "run manifest hash") != manifest_sha256:
        raise ValidationError("base run receipt manifest hash disagrees")
    if require_hash(receipt.get("encoder_sha256"), "run encoder hash") != encoder_sha256:
        raise ValidationError("base run receipt encoder hash disagrees")
    if receipt.get("failures") != [] or exact_int(receipt.get("chunks"), "run chunk count") != BLOCKS:
        raise ValidationError("base run receipt has failures or an incomplete census")
    expected_a128 = sum(int(chunk["alphabet_size"]) == 128 for chunk in chunks)
    if exact_int(receipt.get("a128_chunks"), "run A128 census") != expected_a128:
        raise ValidationError("base run receipt A128 census disagrees")
    if exact_int(receipt.get("a64_chunks"), "run A64 census") != BLOCKS - expected_a128:
        raise ValidationError("base run receipt A64 census disagrees")
    if set(jobs) != set(range(BLOCKS)):
        raise ValidationError("cannot validate a final run receipt without all base jobs")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != BLOCKS:
        raise ValidationError("base run receipt must have 400 ordered rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or exact_int(row.get("chunk_index"), "run row index") != index:
            raise ValidationError("base run receipt rows are not in canonical order")
        job = jobs[index]
        if Path(str(row.get("report", ""))).resolve() != job.report.path.resolve():
            raise ValidationError(f"run row {index} report path disagrees")
        if Path(str(row.get("container", ""))).resolve() != job.container.path.resolve():
            raise ValidationError(f"run row {index} container path disagrees")
        if exact_int(row.get("container_bytes"), "run row container bytes") != job.container.bytes:
            raise ValidationError(f"run row {index} container bytes disagree")
        same_float(row.get("relative_mse"), job.encoder_relative_mse, f"run row {index} MSE")
        if row.get("status") not in ("encoded", "resumed"):
            raise ValidationError(f"run row {index} has unsupported status")
    total_bytes = sum(job.container.bytes for job in jobs.values())
    if exact_int(receipt.get("actual_container_bytes"), "run total container bytes") != total_bytes:
        raise ValidationError("base run receipt total container bytes disagree")
    same_float(
        receipt.get("actual_container_bpw_before_outer_side"),
        total_bytes * 8.0 / PANEL_VALUES,
        "run container bpw",
        tolerance=1e-15,
    )
    return binding


def score_binding(provenance: Provenance, job: BaseJob) -> dict[str, Any]:
    return {
        "manifest_sha256": provenance.manifest_sha256,
        "encoder_sha256": provenance.encoder_sha256,
        "chunk_decoder_sha256": provenance.chunk_decoder_sha256,
        "clean_decoder_sha256": provenance.clean_decoder_sha256,
        "raw_mask_sha256": provenance.raw_mask_sha256,
        "ordered_original_source_hash_digest": provenance.original_source_hash_digest,
        "ordered_normalized_source_hash_digest": provenance.normalized_source_hash_digest,
        "base_report_bytes": job.report.bytes,
        "base_report_sha256": job.report.sha256,
        "container_bytes": job.container.bytes,
        "container_sha256": job.container.sha256,
        "normalized_source_bytes": job.normalized_source.bytes,
        "normalized_source_sha256": job.normalized_source.sha256,
    }


def validate_score_payload(score: object, job: BaseJob, chunk: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(score, dict) or score.get("format") != CHUNK_SCORE_FORMAT:
        raise ValidationError("unsupported clean chunk score format")
    if score.get("status") != "passed" or score.get("strict_ptq") is not True:
        raise ValidationError("clean chunk score is not passed strict PTQ")
    index = job.chunk_index
    if exact_int(score.get("chunk_index"), "score chunk index") != index:
        raise ValidationError("clean chunk score index disagrees")
    if exact_int(score.get("container_bytes"), "score container bytes") != job.container.bytes:
        raise ValidationError("clean chunk score container bytes disagree")
    if require_hash(score.get("container_sha256"), "score container hash") != job.container.sha256:
        raise ValidationError("clean chunk score container hash disagrees")
    if exact_int(score.get("logical_bits"), "score logical bits") != job.logical_bits:
        raise ValidationError("clean chunk score logical bits disagree")
    same_float(score.get("nominal_rate"), chunk["nominal_rate"], "score nominal rate")
    same_float(score.get("actual_container_bpw"), job.container.bytes * 8.0 / BLOCK_VALUES, "score actual bpw")
    normalized_mse = same_float(
        score.get("normalized_relative_mse"),
        job.encoder_relative_mse,
        "normalized decoder/encoder MSE",
        tolerance=1e-12,
    )
    same_float(score.get("encoder_relative_mse"), job.encoder_relative_mse, "recorded encoder MSE")
    energy = finite_float(score.get("raw_source_energy"), "raw source energy", positive=True)
    sse = finite_float(score.get("raw_sse"), "raw SSE")
    if sse < 0.0:
        raise ValidationError("raw SSE is negative")
    relative_mse = same_float(
        score.get("raw_relative_mse"), sse / energy, "raw relative MSE", tolerance=1e-15
    )
    expected_gap = 10.0 * math.log10(
        relative_mse / (2.0 ** (-2.0 * job.container.bytes * 8.0 / BLOCK_VALUES))
    )
    same_float(
        score.get("raw_gap_at_actual_container_rate_db"),
        expected_gap,
        "raw diagnostic gap",
        tolerance=1e-12,
    )
    if score.get("normalized_roundtrip_matches_at_1e_12") is not True:
        raise ValidationError("normalized clean roundtrip flag is false")
    if score.get("tail_padding_zero") is not True:
        raise ValidationError("clean score tail padding flag is false")
    if exact_int(score.get("tail_escape_count"), "score tail count") != job.tail_escape_count:
        raise ValidationError("clean score tail count disagrees")
    if exact_int(score.get("selected_symbols"), "selected symbol count") <= 0:
        raise ValidationError("clean score has no selected symbols")
    require_hash(score.get("frequency_u16_sha256"), "frequency digest")
    require_hash(score.get("reconstruction_indices_sha256"), "index digest")
    if not str(score.get("cupy_version", "")) or not str(score.get("gpu", "")):
        raise ValidationError("clean score omits CuPy/GPU provenance")
    _ = normalized_mse
    return score


def validate_envelope(
    envelope: object,
    provenance: Provenance,
    job: BaseJob,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("format") != ENVELOPE_FORMAT:
        raise ValidationError("unsupported bound score envelope")
    if envelope.get("status") != "passed" or exact_int(envelope.get("chunk_index"), "envelope index") != job.chunk_index:
        raise ValidationError("bound score envelope status/index is invalid")
    if envelope.get("bindings") != score_binding(provenance, job):
        raise ValidationError("bound score provenance disagrees with current inputs")
    return validate_score_payload(envelope.get("score"), job, chunk)


def revalidate_job_fingerprints(job: BaseJob) -> None:
    for binding, description in (
        (job.report, "base report"),
        (job.container, "base container"),
        (job.normalized_source, "normalized source"),
    ):
        current = bind_file(binding.path, description, binding.bytes)
        if current.sha256 != binding.sha256:
            raise ValidationError(f"{description} changed during clean scoring")


def score_one(
    args: argparse.Namespace,
    provenance: Provenance,
    job: BaseJob,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    if STOP_EVENT.is_set():
        raise StopRequested("cooperative stop requested before chunk launch")
    index = job.chunk_index
    stem = f"wf-{index:03d}"
    output = args.output_dir / f"{stem}.clean.json"
    if output.exists():
        try:
            envelope = load_json(output)
            score = validate_envelope(envelope, provenance, job, chunk)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
            raise ValidationError(
                f"existing score {output} is invalid and was left untouched: {error}"
            ) from error
        return score_row("resumed", output, score, job)

    token = f"{os.getpid()}-{threading.get_ident()}"
    temporary = args.output_dir / f".{stem}.{token}.decoder.partial.json"
    if temporary.exists():
        raise FileExistsError(f"stale decoder temporary output exists: {temporary}")
    if STOP_EVENT.is_set():
        raise StopRequested("cooperative stop requested before clean decoder launch")
    command = [
        str(args.python),
        str(args.chunk_decoder),
        "--decoder",
        str(args.decoder),
        "--raw-mask",
        str(args.raw_mask),
        "--manifest",
        str(args.manifest),
        "--chunk-index",
        str(index),
        "--metadata",
        str(job.report.path),
        "--container",
        str(job.container.path),
        "--repo",
        str(args.repo),
        "--output",
        str(temporary),
    ]
    completed = subprocess.run(
        command,
        cwd=args.repo,
        text=True,
        capture_output=True,
        check=False,
    )
    atomic_bytes(
        args.output_dir / f"{stem}.clean.log",
        (completed.stdout + completed.stderr).encode("utf-8", errors="replace"),
    )
    try:
        if completed.returncode != 0:
            raise RuntimeError(
                f"chunk {index} clean decoder exited {completed.returncode}; see {stem}.clean.log"
            )
        if not temporary.is_file():
            raise RuntimeError(f"chunk {index} clean decoder omitted its output")
        score = validate_score_payload(load_json(temporary), job, chunk)
        revalidate_job_fingerprints(job)
        envelope = {
            "format": ENVELOPE_FORMAT,
            "status": "passed",
            "strict_ptq": True,
            "chunk_index": index,
            "bindings": score_binding(provenance, job),
            "score": score,
        }
        atomic_json(output, envelope)
        published = load_json(output)
        score = validate_envelope(published, provenance, job, chunk)
        return score_row("decoded", output, score, job)
    finally:
        if temporary.exists():
            temporary.unlink()


def score_row(status: str, output: Path, score: dict[str, Any], job: BaseJob) -> dict[str, Any]:
    binding = bind_file(output, "bound clean score")
    return {
        "chunk_index": job.chunk_index,
        "status": status,
        "score_path": str(output),
        "score_bytes": binding.bytes,
        "score_sha256": binding.sha256,
        "container_bytes": job.container.bytes,
        "container_sha256": job.container.sha256,
        "raw_source_energy": float(score["raw_source_energy"]),
        "raw_sse": float(score["raw_sse"]),
        "raw_relative_mse": float(score["raw_relative_mse"]),
        "normalized_relative_mse": float(score["normalized_relative_mse"]),
        "tail_escape_count": int(score["tail_escape_count"]),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["chunk_index"]))
    total_energy = math.fsum(float(row["raw_source_energy"]) for row in ordered)
    total_sse = math.fsum(float(row["raw_sse"]) for row in ordered)
    total_bytes = sum(int(row["container_bytes"]) for row in ordered)
    values = len(ordered) * BLOCK_VALUES
    return {
        "summation": "math.fsum over canonical chunk order",
        "scored_chunks": len(ordered),
        "scored_values": values,
        "raw_source_energy": total_energy,
        "raw_sse": total_sse,
        "raw_relative_mse": total_sse / total_energy if total_energy > 0.0 else None,
        "container_bytes": total_bytes,
        "container_bpw_over_scored_chunks": total_bytes * 8.0 / values if values else None,
        "diagnostic_only_until_complete": len(ordered) != BLOCKS,
    }


def paths_overlap(left: Path, right: Path) -> bool:
    lhs = left.resolve()
    rhs = right.resolve()
    return lhs == rhs or lhs in rhs.parents or rhs in lhs.parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chunk-decoder", type=Path, required=True)
    parser.add_argument("--expected-chunk-decoder-sha256", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--expected-decoder-sha256", required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--expected-raw-mask-sha256", required=True)
    parser.add_argument("--expected-encoder-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="return nonzero unless all 400 scores and the run receipt validate",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if paths_overlap(args.output_dir, args.base_dir):
        raise ValidationError("output-dir must be disjoint from the immutable base-dir")
    for protected in (args.manifest, args.run_receipt, args.chunk_decoder, args.decoder, args.raw_mask):
        if protected.exists() and paths_overlap(args.output_dir, protected):
            raise ValidationError(f"output-dir overlaps protected input {protected}")
    receipt_path = args.receipt or args.output_dir / "score.receipt.json"
    if paths_overlap(receipt_path, args.base_dir):
        raise ValidationError("receipt must not be written beneath the immutable base-dir")
    for protected in (args.manifest, args.run_receipt, args.chunk_decoder, args.decoder, args.raw_mask):
        if protected.exists() and paths_overlap(receipt_path, protected):
            raise ValidationError(f"receipt path overlaps protected input {protected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_binding = assert_expected_file(
        args.manifest, args.expected_manifest_sha256, "canonical manifest"
    )
    encoder_sha256 = require_hash(args.expected_encoder_sha256, "expected encoder hash")
    chunk_decoder_binding = assert_expected_file(
        args.chunk_decoder,
        args.expected_chunk_decoder_sha256,
        "exploratory clean chunk decoder",
    )
    clean_decoder_binding = assert_expected_file(
        args.decoder, args.expected_decoder_sha256, "clean PLTE decoder"
    )
    raw_mask_binding = assert_expected_file(
        args.raw_mask,
        args.expected_raw_mask_sha256,
        "six-level raw mask",
        BASE_LEVELS * ((BLOCK_VALUES + 7) // 8),
    )
    if not args.python.is_file():
        raise ValidationError(f"Python executable does not exist: {args.python}")
    orchestrator_binding = bind_file(Path(__file__), "base clean scorer implementation")
    manifest = load_json(args.manifest)
    blocks, chunks = validate_manifest(manifest)
    source_digest, normalized_digest = verify_data_files(blocks, chunks, args.repo)
    provenance = Provenance(
        manifest_sha256=manifest_binding.sha256,
        encoder_sha256=encoder_sha256,
        chunk_decoder_sha256=chunk_decoder_binding.sha256,
        clean_decoder_sha256=clean_decoder_binding.sha256,
        raw_mask_sha256=raw_mask_binding.sha256,
        original_source_hash_digest=source_digest,
        normalized_source_hash_digest=normalized_digest,
    )

    jobs: dict[int, BaseJob] = {}
    pending: list[int] = []
    failures: list[dict[str, Any]] = []
    for chunk in chunks:
        state, job, error = inspect_base_job(
            args.base_dir, chunk, encoder_sha256, args.repo
        )
        index = int(chunk["chunk_index"])
        if state == "ready" and job is not None:
            jobs[index] = job
        elif state == "pending":
            pending.append(index)
        else:
            failures.append({"chunk_index": index, "stage": "base_job_validation", "error": error})

    rows: list[dict[str, Any]] = []
    interrupted: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_one, args, provenance, job, chunks[index]): index
            for index, job in jobs.items()
        }
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            try:
                row = future.result()
                rows.append(row)
                progress(
                    f"[{completed_count}/{len(futures)} ready] chunk {index:03d} "
                    f"{row['status']} SSE={row['raw_sse']:.9g}"
                )
            except StopRequested:
                interrupted.append(index)
            except BaseException as error:
                failures.append(
                    {
                        "chunk_index": index,
                        "stage": "clean_decode_or_resume_validation",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                progress(f"[{completed_count}/{len(futures)} ready] chunk {index:03d} FAILED {error!r}")
    rows.sort(key=lambda row: int(row["chunk_index"]))

    # Re-hash the complete source corpus after scoring.  This detects input
    # replacement during a long partial or final invocation.
    post_source_digest, post_normalized_digest = verify_data_files(blocks, chunks, args.repo)
    inputs_stable = (
        post_source_digest == source_digest and post_normalized_digest == normalized_digest
    )
    if not inputs_stable:
        failures.append(
            {
                "chunk_index": None,
                "stage": "post_score_input_revalidation",
                "error": "source or normalized-source corpus changed during scoring",
            }
        )

    run_receipt_binding: FileBinding | None = None
    run_receipt_error: str | None = None
    run_receipt_deferred = False
    if args.run_receipt.exists():
        if len(jobs) != BLOCKS:
            # The base encoder may have completed after this invocation took its
            # immutable ready-job snapshot.  Defer final validation to resume.
            run_receipt_deferred = True
        else:
            try:
                run_receipt_binding = validate_run_receipt(
                    args.run_receipt,
                    manifest_binding.sha256,
                    encoder_sha256,
                    jobs,
                    chunks,
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
                run_receipt_error = f"{type(error).__name__}: {error}"
                failures.append(
                    {
                        "chunk_index": None,
                        "stage": "canonical_run_receipt_validation",
                        "error": run_receipt_error,
                    }
                )

    complete = (
        not failures
        and not pending
        and len(jobs) == BLOCKS
        and len(rows) == BLOCKS
        and [int(row["chunk_index"]) for row in rows] == list(range(BLOCKS))
        and run_receipt_binding is not None
        and inputs_stable
    )
    status = "complete" if complete else ("failed" if failures else "partial")
    aggregate = aggregate_rows(rows)
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": status,
        "strict_ptq": True,
        "training_or_retraining": False,
        "claim_boundary": (
            "exact exploratory base SSE only; final selected bundle still requires "
            "the independent outer decoder and exact-source evaluator"
        ),
        "inputs": {
            "manifest_path": str(args.manifest),
            "manifest_bytes": manifest_binding.bytes,
            "manifest_sha256": manifest_binding.sha256,
            "base_clean_scorer_sha256": orchestrator_binding.sha256,
            "expected_encoder_sha256": encoder_sha256,
            "chunk_decoder_sha256": chunk_decoder_binding.sha256,
            "clean_decoder_sha256": clean_decoder_binding.sha256,
            "raw_mask_sha256": raw_mask_binding.sha256,
            "ordered_original_source_hash_digest": source_digest,
            "ordered_normalized_source_hash_digest": normalized_digest,
            "all_400_original_and_normalized_source_hashes_verified_before_and_after": inputs_stable,
        },
        "canonical_base_run_receipt": {
            "path": str(args.run_receipt),
            "present": args.run_receipt.is_file(),
            "valid": run_receipt_binding is not None,
            "deferred_until_next_resume": run_receipt_deferred,
            "bytes": run_receipt_binding.bytes if run_receipt_binding else None,
            "sha256": run_receipt_binding.sha256 if run_receipt_binding else None,
            "error": run_receipt_error,
        },
        "census": {
            "canonical_chunks": BLOCKS,
            "ready_base_jobs": len(jobs),
            "pending_base_jobs": len(pending),
            "valid_clean_scores": len(rows),
            "decoded_this_invocation": sum(row["status"] == "decoded" for row in rows),
            "resumed_this_invocation": sum(row["status"] == "resumed" for row in rows),
        },
        "pending_chunk_indices": pending,
        "cooperatively_interrupted_chunk_indices": sorted(interrupted),
        "failures": failures,
        "aggregate": aggregate,
        "full_panel_raw_source_energy": aggregate["raw_source_energy"] if complete else None,
        "full_panel_raw_sse": aggregate["raw_sse"] if complete else None,
        "full_panel_raw_relative_mse": aggregate["raw_relative_mse"] if complete else None,
        "rows": rows,
    }
    atomic_json(receipt_path, receipt)
    summary = {key: value for key, value in receipt.items() if key not in ("rows", "pending_chunk_indices")}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if failures or (args.require_final and not complete):
        return receipt, 1
    return receipt, 0


def failure_receipt(args: argparse.Namespace, error: BaseException) -> int:
    try:
        if paths_overlap(args.output_dir, args.base_dir):
            raise ValidationError(
                "refusing to write even a failure receipt beneath immutable base-dir"
            )
        receipt_path = args.receipt or args.output_dir / "score.receipt.json"
        for protected in (
            args.manifest,
            args.run_receipt,
            args.chunk_decoder,
            args.decoder,
            args.raw_mask,
        ):
            if protected.exists() and paths_overlap(receipt_path, protected):
                raise ValidationError(
                    f"refusing failure receipt that overlaps protected input {protected}"
                )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "format": RECEIPT_FORMAT,
            "status": "failed",
            "strict_ptq": True,
            "error": f"{type(error).__name__}: {error}",
            "rows": [],
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2), file=sys.stderr)
    except BaseException as receipt_error:
        print(
            f"fatal scorer error {error!r}; additionally failed to write failure receipt: {receipt_error!r}",
            file=sys.stderr,
        )
    return 1


def main() -> None:
    args = parse_args()

    def request_stop(signum: int, _frame: object) -> None:
        STOP_EVENT.set()
        progress(
            f"received signal {signum}; finishing active decoders and cancelling queued chunks"
        )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        _, exit_code = execute(args)
    except BaseException as error:
        exit_code = failure_receipt(args, error)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
