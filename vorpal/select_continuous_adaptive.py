#!/usr/bin/env python3
"""Select and stage the globally best continuous-PLTE adaptive ensemble.

This is a fail-closed, write-once staging tool.  It validates the frozen
400-chunk base run and every candidate named by ``candidate.receipt.json``,
then solves the exact multiple-choice problem over integer container-byte
deltas.  For each byte delta the DP retains the greatest raw-SSE saving and
Pareto-prunes states dominated in both bytes and saving.  The final objective
is evaluated at high Decimal precision as

    (base_raw_sse - saving) * 2 ** (16 * WFOUTR01_bundle_bytes / panel_values)

where the physical bundle includes the 168-byte header, exact side payload,
exact BZ2-compressed raw mask, and selected containers.  In claim-grade
fixed-route mode the side payload is one XZ(canonical all-A64 side) plus an
exact 50-byte A64/A128 route, so its length is invariant across all DP states.
The source energy is constant and therefore irrelevant to selection, but is
used for an audit gap when supplied by the manifest or CLI.

The tool never modifies the base or candidate trees.  It copies exactly one
validated container per chunk into a new staging directory and refuses to
overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import hashlib
import itertools
import json
import lzma
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable


CHUNK_COUNT = 400
BLOCK_VALUES = 1 << 18
LOGICAL_LENGTH_BITS = 20
ESCAPE_RECORD_BITS = 34
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
PROFILE_BYTES = 17
PROFILE_ALPHABET_BYTE = 16
WFOUTR01_HEADER_BYTES = 168
ALPHABET_CODE = {64: 0, 128: 1, 256: 2}
FIXED_ROUTE_BITS = 400
FIXED_ROUTE_BYTES = FIXED_ROUTE_BITS // 8
SOURCE_ENERGY_MAX_BINARY64_ULPS = 4
SOURCE_ENERGY_MAX_RELATIVE_ERROR = Decimal("1e-15")


@dataclass(frozen=True)
class Choice:
    """One validated option for a triggered chunk."""

    chunk_index: int
    option_id: str
    kind: str
    alphabet_size: int
    container_path: Path
    container_bytes: int
    container_sha256: str
    raw_sse: Decimal
    base_raw_sse: Decimal
    savings: Decimal
    escape_count: int


@dataclass(frozen=True)
class State:
    """A DP state after a prefix of triggered chunks."""

    byte_delta: int
    savings: Decimal
    choice_ids: tuple[str, ...]
    alphabet_mask: int = 0


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"not JSON serializable: {type(item).__name__}")

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_decimal_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=Decimal,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number {value!r} in {path}")
        ),
    )


def decimal_arg(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid decimal: {text!r}") from error
    if not value.is_finite():
        raise argparse.ArgumentTypeError("decimal must be finite")
    return value


def require_decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        relation = "positive" if positive else "nonnegative"
        raise AssertionError(f"{label} must be finite and {relation}: {result}")
    return result


def require_serialized_decimal(
    value: Any, label: str, *, positive: bool = False
) -> Decimal:
    """Require a finite JSON non-integer number parsed directly as Decimal.

    Candidate metrics are emitted from binary64 values.  Rejecting strings,
    booleans, integers, and already-converted Python floats prevents permissive
    coercions from bypassing the audited ULP comparison below.
    """

    if type(value) is not Decimal:
        raise AssertionError(f"{label} must be a literal JSON decimal number")
    return require_decimal(value, label, positive=positive)


def source_energy_distance(
    canonical: Any, independently_reduced: Any
) -> tuple[Decimal, Decimal, int]:
    """Return absolute, relative, and positive-binary64 ULP distances."""

    left = require_serialized_decimal(
        canonical, "canonical raw source energy", positive=True
    )
    right = require_serialized_decimal(
        independently_reduced,
        "independently reduced raw source energy",
        positive=True,
    )
    try:
        left_float = float(left)
        right_float = float(right)
    except (OverflowError, ValueError) as error:
        raise AssertionError("raw source energy is not finite binary64") from error
    if not (math.isfinite(left_float) and math.isfinite(right_float)):
        raise AssertionError("raw source energy is not finite binary64")
    left_bits = struct.unpack(">Q", struct.pack(">d", left_float))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right_float))[0]
    # Positive energies have monotone IEEE-754 encodings.
    ulps = abs(left_bits - right_bits)
    absolute = abs(left - right)
    relative = absolute / max(abs(left), abs(right))
    return absolute, relative, ulps


def source_energy_matches_canonical(
    canonical: Any, independently_reduced: Any
) -> tuple[Decimal, Decimal, int]:
    """Fail unless two source-bound reductions agree within the audited gate."""

    absolute, relative, ulps = source_energy_distance(
        canonical, independently_reduced
    )
    if (
        ulps > SOURCE_ENERGY_MAX_BINARY64_ULPS
        or relative > SOURCE_ENERGY_MAX_RELATIVE_ERROR
    ):
        raise AssertionError(
            "raw source energy mismatch exceeds canonical-source gate: "
            f"absolute={absolute}, relative={relative}, ulps={ulps}"
        )
    return absolute, relative, ulps


def require_sha256_hex(value: Any, label: str) -> str:
    """Require the canonical lowercase encoding used by every bound receipt."""

    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{label} is not a canonical SHA-256 digest")
    return value


def base_artifacts(base_dir: Path, index: int) -> tuple[Path, Path]:
    stem = base_dir / f"wf-{index:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".polar.bin")


def parse_container(path: Path, *, expected_escape_count: int | None = None) -> dict[str, int | str]:
    size = path.stat().st_size
    if size < 8:
        raise AssertionError(f"container shorter than header: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        header_word = struct.unpack_from("<I", prefix)[0]
        logical_bits = header_word & ((1 << LOGICAL_LENGTH_BITS) - 1)
        escape_count = header_word >> LOGICAL_LENGTH_BITS
        payload_bytes = (logical_bits + 7) // 8
        tail_bytes = (ESCAPE_RECORD_BITS * escape_count + 7) // 8
        expected_size = 8 + payload_bytes + tail_bytes
        if expected_size != size:
            raise AssertionError(
                f"self-delimiting length mismatch for {path}: {size} != {expected_size}"
            )
        if expected_escape_count is not None and escape_count != expected_escape_count:
            raise AssertionError(
                f"escape count mismatch for {path}: {escape_count} != {expected_escape_count}"
            )
        padding_bits = (-ESCAPE_RECORD_BITS * escape_count) % 8
        if padding_bits and tail_bytes:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1)[0] & ((1 << padding_bits) - 1):
                raise AssertionError(f"nonzero sparse-tail padding in {path}")
    return {
        "container_bytes": size,
        "logical_bits": logical_bits,
        "escape_count": escape_count,
        "payload_bytes": payload_bytes,
        "tail_bytes": tail_bytes,
        "sha256": sha256_path(path),
    }


def validate_encoder_report(
    report_path: Path,
    container_path: Path,
    chunk: dict[str, Any],
    expected_alphabet: int,
) -> dict[str, Any]:
    report = load_json(report_path)
    parameters = report["parameters"]
    trials = report["trials"]
    if not isinstance(trials, list) or len(trials) != 1:
        raise AssertionError(f"expected one trial in {report_path}")
    trial = trials[0]
    parsed = parse_container(container_path)
    if not (
        int(parameters["block_length"]) == BLOCK_VALUES
        and int(parameters["alphabet_size"]) == expected_alphabet
        and int(parameters["container_cap_bytes"]) == 0
        and float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"])
        and float(parameters["eta"]) == float(chunk["eta"])
        and trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"]
        and int(trial["literal_container_bytes"]) == parsed["container_bytes"]
        and trial["literal_container_sha256"] == parsed["sha256"]
        and all(trial.get(field) is True for field in ROUNDTRIP_FIELDS)
    ):
        raise AssertionError(f"encoder/report validation failed for {report_path}")
    return {"report": report, "trial": trial, "container": parsed}


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if manifest.get("strict_ptq") is not True or manifest.get("training_or_retraining") is not False:
        raise AssertionError("manifest is not explicitly strict PTQ without retraining")
    chunks = manifest["chunks"]
    if len(chunks) != CHUNK_COUNT:
        raise AssertionError(f"expected {CHUNK_COUNT} chunks")
    if [int(chunk["chunk_index"]) for chunk in chunks] != list(range(CHUNK_COUNT)):
        raise AssertionError("chunks are not in canonical 0..399 order")
    block_values = int(manifest["parameters"]["block_values"])
    if block_values != BLOCK_VALUES:
        raise AssertionError(f"unexpected chunk length {block_values}")
    panel_values = int(manifest["census"]["values"])
    if panel_values != CHUNK_COUNT * BLOCK_VALUES:
        raise AssertionError("panel value census mismatch")
    for chunk in chunks:
        alphabet = int(chunk["alphabet_size"])
        if alphabet not in (64, 128, 256):
            raise AssertionError(f"unsupported manifest alphabet {alphabet}")
    return chunks, panel_values


def validate_base_run(
    manifest_path: Path,
    manifest: dict[str, Any],
    base_dir: Path,
) -> tuple[list[dict[str, Any]], int, dict[int, dict[str, Any]]]:
    chunks, _ = validate_manifest(manifest)
    receipt_path = base_dir / "run.receipt.json"
    receipt = load_json(receipt_path)
    if not (
        receipt.get("status") == "complete"
        and receipt.get("all_internal_roundtrips_passed") is True
        and int(receipt.get("chunks", -1)) == CHUNK_COUNT
        and receipt.get("manifest_sha256") == sha256_path(manifest_path)
        and not receipt.get("failures")
    ):
        raise AssertionError("base run receipt is incomplete or does not bind the manifest")
    receipt_rows = receipt.get("rows")
    if not isinstance(receipt_rows, list) or sorted(
        int(row["chunk_index"]) for row in receipt_rows
    ) != list(range(CHUNK_COUNT)):
        raise AssertionError("base run receipt does not contain exactly one row per chunk")

    validated: dict[int, dict[str, Any]] = {}
    total_bytes = 0
    for chunk in chunks:
        index = int(chunk["chunk_index"])
        report_path, container_path = base_artifacts(base_dir, index)
        result = validate_encoder_report(
            report_path, container_path, chunk, int(chunk["alphabet_size"])
        )
        validated[index] = {
            **result,
            "report_path": report_path,
            "container_path": container_path,
        }
        total_bytes += int(result["container"]["container_bytes"])
    if int(receipt["actual_container_bytes"]) != total_bytes:
        raise AssertionError("base run total byte count mismatch")
    for row in receipt_rows:
        index = int(row["chunk_index"])
        if int(row["container_bytes"]) != validated[index]["container"]["container_bytes"]:
            raise AssertionError(f"base run row byte mismatch for chunk {index}")
    return chunks, total_bytes, validated


def validate_decode_receipt(
    path: Path,
    index: int,
    container_sha256: str,
    expected_sse: Decimal,
    expected_energy: Decimal,
    *,
    manifest_sha256: str,
    metadata_sha256: str,
    normalized_source_sha256: str,
    scorer_sha256: str,
    decoder_sha256: str,
    raw_mask_sha256: str,
) -> None:
    decoded = load_decimal_json(path)
    binding = decoded.get("adaptive_input_binding")
    expected_binding = {
        "manifest_sha256": manifest_sha256,
        "metadata_sha256": metadata_sha256,
        "container_sha256": container_sha256,
        "scorer_sha256": scorer_sha256,
        "decoder_sha256": decoder_sha256,
        "raw_mask_sha256": raw_mask_sha256,
        "normalized_source_sha256": normalized_source_sha256,
    }
    if not (
        decoded.get("status") == "passed"
        and int(decoded["chunk_index"]) == index
        and decoded["container_sha256"] == container_sha256
        and require_decimal(decoded["raw_sse"], "decode raw_sse") == expected_sse
        and require_serialized_decimal(
            decoded["raw_source_energy"], "decode raw energy", positive=True
        )
        == expected_energy
        and isinstance(binding, dict)
        and all(binding.get(key) == value for key, value in expected_binding.items())
    ):
        raise AssertionError(f"invalid independent decode receipt {path}")


def validate_tail_payload_identity(base_path: Path, tail_path: Path, expected_k: int) -> None:
    base_info = parse_container(base_path, expected_escape_count=0)
    tail_info = parse_container(tail_path, expected_escape_count=expected_k)
    if tail_info["logical_bits"] != base_info["logical_bits"]:
        raise AssertionError("tail candidate changes arithmetic logical length")
    base = base_path.read_bytes()
    tail = tail_path.read_bytes()
    if tail[4 : len(base)] != base[4:]:
        raise AssertionError("tail candidate changes scale or arithmetic payload")
    expected_delta = (ESCAPE_RECORD_BITS * expected_k + 7) // 8
    if len(tail) != len(base) + expected_delta:
        raise AssertionError("tail candidate has an unexpected physical byte delta")


def option_sort_key(choice: Choice) -> tuple[int, int, str]:
    # ``a128`` remains accepted here solely for the low-level DP unit fixtures;
    # validated V3 receipts always materialize ``alphabet-upgrade``.
    kind_order = {"base": 0, "a128": 1, "alphabet-upgrade": 1, "tail": 2}
    return kind_order[choice.kind], choice.escape_count, choice.option_id


def validate_adaptive_receipt(
    receipt_path: Path,
    manifest_path: Path,
    base_receipt_path: Path,
    raw_mask_path: Path,
    chunks: list[dict[str, Any]],
    validated_base: dict[int, dict[str, Any]],
) -> tuple[
    list[int],
    list[list[Choice]],
    dict[int, list[Choice]],
    dict[str, Any],
]:
    receipt = load_decimal_json(receipt_path)
    if not (
        receipt.get("format")
        == "continuous PLTE all-base adaptive candidate receipt v3"
        and receipt.get("status") == "complete"
        and receipt.get("strict_ptq") is True
        and receipt.get("training_or_retraining") is False
        and receipt.get("manifest_sha256") == sha256_path(manifest_path)
        and receipt.get("base_receipt_sha256") == sha256_path(base_receipt_path)
        and receipt.get("base_receipt_status") == "complete"
        and receipt.get("raw_mask_sha256") == sha256_path(raw_mask_path)
        and not receipt.get("failures")
    ):
        raise AssertionError("adaptive candidate receipt is incomplete or unbound")

    for field in (
        "implementation_sha256",
        "pinned_runner_core_sha256",
        "pinned_repacker_core_sha256",
        "manifest_sha256",
        "base_receipt_sha256",
        "encoder_sha256",
        "repacker_sha256",
        "scorer_sha256",
        "decoder_sha256",
        "raw_mask_sha256",
    ):
        require_sha256_hex(receipt.get(field), f"candidate receipt {field}")

    all_indices = list(range(CHUNK_COUNT))
    scanned = [int(value) for value in receipt["scanned_chunk_indices"]]
    if (
        scanned != all_indices
        or int(receipt["base_reports_scanned"]) != CHUNK_COUNT
        or receipt.get("trigger_predicate_universe")
        != "all 400 canonical validated base gaps"
        or receipt.get("row_schema")
        != {
            "base_alphabet_size": "required: 64 or 128",
            "base": "required and explicitly carries alphabet_size",
            "upgrade": "A64: required A128 object; A128: null",
            "tails": "required prefixes against the base alphabet",
        }
    ):
        raise AssertionError("adaptive receipt did not scan all 400 base chunks")
    threshold = require_decimal(
        receipt["trigger_gap_db_strictly_greater_than"], "trigger threshold"
    )
    if threshold != Decimal("0.1"):
        raise AssertionError(f"hardened adaptive trigger must be 0.10 dB, got {threshold}")
    expected_triggered = []
    for index in all_indices:
        gap = Decimal(str(validated_base[index]["trial"]["gap_db"]))
        if gap > threshold:
            expected_triggered.append(index)
    triggered = [int(value) for value in receipt["triggered_chunk_indices"]]
    if triggered != expected_triggered or len(set(triggered)) != len(triggered):
        raise AssertionError("triggered chunk set does not reproduce the frozen strict threshold")
    expected_trigger_counts = {
        "64": sum(int(chunks[index]["alphabet_size"]) == 64 for index in triggered),
        "128": sum(int(chunks[index]["alphabet_size"]) == 128 for index in triggered),
    }
    if receipt.get("triggered_base_alphabet_counts") != expected_trigger_counts:
        raise AssertionError("triggered base-alphabet census mismatch")
    rows = receipt["rows"]
    if [int(row["chunk_index"]) for row in rows] != triggered:
        raise AssertionError("adaptive rows do not exactly match canonical triggered order")
    expected_tail_ks = tuple(int(value) for value in receipt["tail_prefixes"])
    if (
        expected_tail_ks != tuple(sorted(set(expected_tail_ks)))
        or not expected_tail_ks
        or any(value <= 0 or value >= (1 << 12) for value in expected_tail_ks)
    ):
        raise AssertionError("adaptive receipt has invalid tail-prefix schedule")

    by_index: dict[int, list[Choice]] = {}
    groups: list[list[Choice]] = []
    upgrade_energy_comparisons = 0
    nonexact_upgrade_energies = 0
    tail_energy_comparisons = 0
    maximum_energy_absolute = Decimal(0)
    maximum_energy_relative = Decimal(0)
    maximum_energy_ulps = 0
    for row in rows:
        index = int(row["chunk_index"])
        chunk = chunks[index]
        normalized_source = Path(str(chunk["normalized_source"]))
        if not normalized_source.is_absolute():
            normalized_source = manifest_path.parent / normalized_source
        if (
            not normalized_source.is_file()
            or sha256_path(normalized_source)
            != str(chunk["normalized_source_sha256"])
        ):
            raise AssertionError(
                f"canonical normalized source hash mismatch for chunk {index}"
            )
        base_alphabet = int(chunk["alphabet_size"])
        if base_alphabet not in (64, 128):
            raise AssertionError(f"adaptive row {index} has unsupported base alphabet")
        expected_kinds = (
            ["base", "alphabet-upgrade", "tail"]
            if base_alphabet == 64
            else ["base", "tail"]
        )
        if not (
            int(row["base_alphabet_size"]) == base_alphabet
            and row.get("available_option_kinds") == expected_kinds
        ):
            raise AssertionError(f"adaptive option schema mismatch for chunk {index}")
        base_info = row["base"]
        base_path = validated_base[index]["container_path"]
        base_parsed = validated_base[index]["container"]
        base_sse = require_decimal(base_info["raw_sse"], f"chunk {index} base raw_sse")
        raw_energy = require_serialized_decimal(
            base_info["raw_source_energy"], f"chunk {index} raw energy", positive=True
        )
        trigger_gap = require_decimal(
            row["trigger_gap_db"], f"chunk {index} serialized trigger gap"
        )
        if not (
            int(base_info["container_bytes"]) == base_parsed["container_bytes"]
            and base_info["container_sha256"] == base_parsed["sha256"]
            and int(base_info["alphabet_size"]) == base_alphabet
            and Path(str(base_info["container"])).resolve() == base_path.resolve()
            and Path(str(base_info["report"])).resolve()
            == validated_base[index]["report_path"].resolve()
            and trigger_gap == Decimal(str(validated_base[index]["trial"]["gap_db"]))
        ):
            raise AssertionError(f"adaptive base binding mismatch for chunk {index}")

        choices = [
            Choice(
                chunk_index=index,
                option_id="base",
                kind="base",
                alphabet_size=base_alphabet,
                container_path=base_path,
                container_bytes=int(base_parsed["container_bytes"]),
                container_sha256=str(base_parsed["sha256"]),
                raw_sse=base_sse,
                base_raw_sse=base_sse,
                savings=Decimal(0),
                escape_count=int(base_parsed["escape_count"]),
            )
        ]

        upgrade = row.get("upgrade")
        if base_alphabet == 64:
            if not (
                isinstance(upgrade, dict)
                and upgrade.get("kind") == "alphabet-upgrade"
                and int(upgrade.get("from_alphabet_size", -1)) == 64
                and int(upgrade.get("to_alphabet_size", -1)) == 128
                and upgrade.get("independent_decode_passed") is True
            ):
                raise AssertionError(f"A64 chunk {index} omitted required A128 candidate")
            a128_path = Path(str(upgrade["container"]))
            a128_report = Path(str(upgrade["report"]))
            a128_parsed = validate_encoder_report(
                a128_report, a128_path, chunk, 128
            )["container"]
            a128_sse = require_decimal(
                upgrade["raw_sse"], f"chunk {index} A128 raw_sse"
            )
            a128_energy = require_serialized_decimal(
                upgrade["raw_source_energy"],
                f"chunk {index} A128 raw energy",
                positive=True,
            )
            absolute, relative, ulps = source_energy_matches_canonical(
                raw_energy, a128_energy
            )
            upgrade_energy_comparisons += 1
            nonexact_upgrade_energies += int(absolute != 0)
            maximum_energy_absolute = max(maximum_energy_absolute, absolute)
            maximum_energy_relative = max(maximum_energy_relative, relative)
            maximum_energy_ulps = max(maximum_energy_ulps, ulps)
            if not (
                int(upgrade["container_bytes"])
                == a128_parsed["container_bytes"]
                and upgrade["container_sha256"] == a128_parsed["sha256"]
            ):
                raise AssertionError(f"A128 candidate binding mismatch for chunk {index}")
            validate_decode_receipt(
                Path(str(upgrade["decode"])),
                index,
                str(a128_parsed["sha256"]),
                a128_sse,
                a128_energy,
                manifest_sha256=str(receipt["manifest_sha256"]),
                metadata_sha256=sha256_path(a128_report),
                normalized_source_sha256=str(chunk["normalized_source_sha256"]),
                scorer_sha256=str(receipt["scorer_sha256"]),
                decoder_sha256=str(receipt["decoder_sha256"]),
                raw_mask_sha256=str(receipt["raw_mask_sha256"]),
            )
            choices.append(
                Choice(
                    chunk_index=index,
                    option_id="upgrade-a128",
                    kind="alphabet-upgrade",
                    alphabet_size=128,
                    container_path=a128_path,
                    container_bytes=int(a128_parsed["container_bytes"]),
                    container_sha256=str(a128_parsed["sha256"]),
                    raw_sse=a128_sse,
                    base_raw_sse=base_sse,
                    savings=base_sse - a128_sse,
                    escape_count=int(a128_parsed["escape_count"]),
                )
            )
        elif not (
            upgrade is None
            and base_info.get("independent_clean_decode_passed") is True
        ):
            raise AssertionError(f"A128 base chunk {index} must not carry A256/A128 re-encode")

        seen_ks: set[int] = set()
        for tail in row["tails"]:
            k = int(tail["escape_count"])
            if k in seen_ks or k <= 0:
                raise AssertionError(f"duplicate/invalid tail prefix k={k} for chunk {index}")
            seen_ks.add(k)
            path = Path(str(tail["container_path"]))
            parsed = parse_container(path, expected_escape_count=k)
            tail_sse = require_decimal(tail["raw_sse"], f"chunk {index} tail-{k} raw_sse")
            tail_energy = require_serialized_decimal(
                tail["raw_source_energy"], f"chunk {index} tail-{k} energy", positive=True
            )
            tail_energy_comparisons += 1
            if not (
                tail_energy == raw_energy
                and tail.get("payload_unchanged") is True
                and tail.get("independent_physical_reparse_passed") is True
                and tail.get("parsed_tail_applied_for_scoring") is True
                and tail.get("raw_gain_identity_passed") is True
                and int(tail["container_bytes"]) == parsed["container_bytes"]
                and tail["container_sha256"] == parsed["sha256"]
            ):
                raise AssertionError(f"tail candidate binding mismatch for chunk {index}, k={k}")
            validate_tail_payload_identity(base_path, path, k)
            choices.append(
                Choice(
                    chunk_index=index,
                    option_id=f"tail-k{k}",
                    kind="tail",
                    alphabet_size=base_alphabet,
                    container_path=path,
                    container_bytes=int(parsed["container_bytes"]),
                    container_sha256=str(parsed["sha256"]),
                    raw_sse=tail_sse,
                    base_raw_sse=base_sse,
                    savings=base_sse - tail_sse,
                    escape_count=k,
                )
            )
        if tuple(sorted(seen_ks)) != expected_tail_ks:
            raise AssertionError(f"chunk {index} tail-prefix schedule is incomplete")
        choices.sort(key=option_sort_key)
        if len({choice.option_id for choice in choices}) != len(choices):
            raise AssertionError(f"non-unique option identifiers for chunk {index}")
        by_index[index] = choices
        groups.append(choices)
    energy_audit = {
        "canonical_energy": "base/tail single CuPy float64 reduction bound to canonical normalized-source SHA256",
        "independent_energy": "A128 clean-decoder per-group CuPy float64 reduction with exact adaptive_input_binding",
        "literal_json_decimal_type_required": True,
        "maximum_binary64_ulps_allowed": SOURCE_ENERGY_MAX_BINARY64_ULPS,
        "maximum_relative_error_allowed_decimal": decimal_string(
            SOURCE_ENERGY_MAX_RELATIVE_ERROR
        ),
        "upgrade_comparisons": upgrade_energy_comparisons,
        "nonexact_upgrade_comparisons": nonexact_upgrade_energies,
        "tail_exact_comparisons": tail_energy_comparisons,
        "maximum_observed_absolute_error_decimal": decimal_string(
            maximum_energy_absolute
        ),
        "maximum_observed_relative_error_decimal": decimal_string(
            maximum_energy_relative
        ),
        "maximum_observed_binary64_ulps": maximum_energy_ulps,
    }
    return triggered, groups, by_index, energy_audit


def pareto_dp(option_groups: Iterable[list[Choice]]) -> list[State]:
    """Return the raw-side constant-proxy frontier over integer byte deltas.

    This cross-signature pruning is not sufficient for the final WFOUTR01
    objective because XZ(side) depends on the alphabet-code pattern.  It is
    retained as a diagnostic and as an explicitly non-global fallback mode.
    """

    frontier = [State(0, Decimal(0), ())]
    for group_index, group in enumerate(option_groups):
        if not group:
            raise ValueError("empty multiple-choice group")
        base_bytes = next((choice.container_bytes for choice in group if choice.kind == "base"), None)
        if base_bytes is None:
            raise ValueError("each group requires one base choice")
        best_at_delta: dict[int, State] = {}
        for state, choice in itertools.product(frontier, group):
            delta = state.byte_delta + choice.container_bytes - base_bytes
            candidate = State(
                delta,
                state.savings + choice.savings,
                state.choice_ids + (choice.option_id,),
                state.alphabet_mask
                | ((1 << group_index) if choice.alphabet_size == 128 else 0),
            )
            incumbent = best_at_delta.get(delta)
            if (
                incumbent is None
                or candidate.savings > incumbent.savings
                or (
                    candidate.savings == incumbent.savings
                    and candidate.choice_ids < incumbent.choice_ids
                )
            ):
                best_at_delta[delta] = candidate
        pruned: list[State] = []
        best_savings: Decimal | None = None
        for delta in sorted(best_at_delta):
            state = best_at_delta[delta]
            if best_savings is None or state.savings > best_savings:
                pruned.append(state)
                best_savings = state.savings
        frontier = pruned
    return frontier


def signature_pareto_dp(
    option_groups: Iterable[list[Choice]], *, max_states: int
) -> tuple[list[State], int]:
    """Return a physically safe frontier, pruning only within one side pattern.

    States with different A64/A128 signatures are never allowed to dominate
    one another because their XZ-compressed side lengths can differ.  Within a
    signature the raw side is byte-identical, so ordinary byte/SSE dominance
    is exact for the final physical objective.
    """

    frontier = [State(0, Decimal(0), (), 0)]
    peak_states = 1
    for group_index, group in enumerate(option_groups):
        if not group:
            raise ValueError("empty multiple-choice group")
        base_bytes = next(
            (choice.container_bytes for choice in group if choice.kind == "base"), None
        )
        if base_bytes is None:
            raise ValueError("each group requires one base choice")
        best_at_key: dict[tuple[int, int], State] = {}
        for state, choice in itertools.product(frontier, group):
            signature = state.alphabet_mask | (
                (1 << group_index) if choice.alphabet_size == 128 else 0
            )
            delta = state.byte_delta + choice.container_bytes - base_bytes
            candidate = State(
                delta,
                state.savings + choice.savings,
                state.choice_ids + (choice.option_id,),
                signature,
            )
            key = signature, delta
            incumbent = best_at_key.get(key)
            if (
                incumbent is None
                or candidate.savings > incumbent.savings
                or (
                    candidate.savings == incumbent.savings
                    and candidate.choice_ids < incumbent.choice_ids
                )
            ):
                best_at_key[key] = candidate

        by_signature: dict[int, list[State]] = {}
        for (signature, _), state in best_at_key.items():
            by_signature.setdefault(signature, []).append(state)
        pruned: list[State] = []
        for signature in sorted(by_signature):
            best_savings: Decimal | None = None
            for state in sorted(by_signature[signature], key=lambda item: item.byte_delta):
                if best_savings is None or state.savings > best_savings:
                    pruned.append(state)
                    best_savings = state.savings
        frontier = pruned
        peak_states = max(peak_states, len(frontier))
        if len(frontier) > max_states:
            raise RuntimeError(
                f"exact signature DP reached {len(frontier)} states after group "
                f"{group_index + 1}, above --max-exact-states={max_states}; "
                "refusing to silently cross-prune selection-dependent XZ sides"
            )
    return frontier, peak_states


def objective_log(
    raw_sse: Decimal, literal_bytes: int, panel_values: int
) -> Decimal:
    if raw_sse <= 0 or literal_bytes < 0 or panel_values <= 0:
        raise ValueError("invalid objective inputs")
    with localcontext() as context:
        context.prec = 80
        return raw_sse.ln() + Decimal(2).ln() * Decimal(16 * literal_bytes) / Decimal(panel_values)


def choose_best_state(
    frontier: list[State],
    *,
    base_raw_sse: Decimal,
    base_container_bytes: int,
    constant_proxy_bytes: int,
    panel_values: int,
    max_bpw: Decimal,
) -> tuple[State, Decimal]:
    feasible: list[tuple[Decimal, int, tuple[str, ...], State]] = []
    for state in frontier:
        selected_sse = base_raw_sse - state.savings
        total_bytes = base_container_bytes + state.byte_delta + constant_proxy_bytes
        if selected_sse <= 0 or total_bytes <= 0:
            continue
        rate = Decimal(8 * total_bytes) / Decimal(panel_values)
        if rate >= max_bpw:  # Overall project contract is strict, not <=.
            continue
        score = objective_log(selected_sse, total_bytes, panel_values)
        feasible.append((score, total_bytes, state.choice_ids, state))
    if not feasible:
        raise AssertionError(f"no Pareto state satisfies strict {max_bpw} bpw")
    score, _, _, state = min(feasible, key=lambda item: (item[0], item[1], item[2]))
    return state, score


def side_blob_for_signature(
    original_side: bytes,
    chunks: list[dict[str, Any]],
    triggered: list[int],
    signature: int,
) -> bytes:
    profile_bytes = PROFILE_BYTES * len(chunks)
    if len(original_side) < profile_bytes:
        raise AssertionError("raw side is shorter than its fixed-width profile table")
    profile_offset = len(original_side) - profile_bytes
    blob = bytearray(original_side)
    for index, chunk in enumerate(chunks):
        expected = ALPHABET_CODE[int(chunk["alphabet_size"])]
        offset = profile_offset + index * PROFILE_BYTES + PROFILE_ALPHABET_BYTE
        if blob[offset] != expected:
            raise AssertionError(
                f"raw-side alphabet code mismatch at chunk {index}: {blob[offset]} != {expected}"
            )
    for bit_index, chunk_index in enumerate(triggered):
        offset = profile_offset + chunk_index * PROFILE_BYTES + PROFILE_ALPHABET_BYTE
        blob[offset] = 1 if signature & (1 << bit_index) else 0
    return bytes(blob)


def compress_signature_sides(
    frontier: list[State],
    *,
    original_side: bytes,
    chunks: list[dict[str, Any]],
    triggered: list[int],
    workers: int,
    max_signatures: int,
) -> dict[int, dict[str, Any]]:
    signatures = sorted({state.alphabet_mask for state in frontier})
    if len(signatures) > max_signatures:
        raise RuntimeError(
            f"exact physical rerank needs {len(signatures)} XZ side encodes, above "
            f"--max-compression-signatures={max_signatures}"
        )

    def compress_one(signature: int) -> tuple[int, dict[str, Any]]:
        raw = side_blob_for_signature(original_side, chunks, triggered, signature)
        compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9)
        return signature, {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_bytes": len(compressed),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        }

    result: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for signature, row in pool.map(compress_one, signatures):
            result[signature] = row
    return result


def fixed_route_payload(
    raw_side: bytes,
    chunks: list[dict[str, Any]],
    *,
    canonical_xz: bytes | None = None,
) -> dict[str, Any]:
    """Canonicalize all 400 alphabet codes and append a fixed route bitset."""

    if len(chunks) != FIXED_ROUTE_BITS:
        raise AssertionError("fixed-route experiment requires exactly 400 profiles")
    profile_offset = len(raw_side) - PROFILE_BYTES * len(chunks)
    if profile_offset < 0:
        raise AssertionError("raw side is shorter than profile table")
    canonical = bytearray(raw_side)
    route = bytearray(FIXED_ROUTE_BYTES)
    for index in range(len(chunks)):
        offset = profile_offset + index * PROFILE_BYTES + PROFILE_ALPHABET_BYTE
        code = canonical[offset]
        if code not in (0, 1):
            raise ValueError(f"fixed route rejects alphabet code {code} at chunk {index}")
        if code:
            route[index >> 3] |= 1 << (index & 7)
        canonical[offset] = 0
    canonical_bytes = bytes(canonical)
    if canonical_xz is None:
        canonical_xz = lzma.compress(
            canonical_bytes, format=lzma.FORMAT_XZ, preset=9
        )
    payload = canonical_xz + bytes(route)
    return {
        "raw_sha256": hashlib.sha256(raw_side).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "canonical_xz_bytes": len(canonical_xz),
        "canonical_xz_sha256": hashlib.sha256(canonical_xz).hexdigest(),
        "route_bytes": FIXED_ROUTE_BYTES,
        "route_sha256": hashlib.sha256(route).hexdigest(),
        "compressed_bytes": len(payload),
        "compressed_sha256": hashlib.sha256(payload).hexdigest(),
        "_canonical_xz": canonical_xz,
    }


def compress_fixed_route_sides(
    frontier: list[State],
    *,
    original_side: bytes,
    chunks: list[dict[str, Any]],
    triggered: list[int],
) -> dict[int, dict[str, Any]]:
    """Build exact codec-3 payload metadata for retained signatures."""

    signatures = sorted({state.alphabet_mask for state in frontier})
    result = {}
    fixed_length: int | None = None
    canonical_hash: str | None = None
    cached_xz: bytes | None = None
    for signature in signatures:
        raw = side_blob_for_signature(original_side, chunks, triggered, signature)
        row = fixed_route_payload(raw, chunks, canonical_xz=cached_xz)
        if fixed_length is None:
            fixed_length = int(row["compressed_bytes"])
            canonical_hash = str(row["canonical_raw_sha256"])
            cached_xz = bytes(row["_canonical_xz"])
        if (
            int(row["compressed_bytes"]) != fixed_length
            or str(row["canonical_raw_sha256"]) != canonical_hash
        ):
            raise AssertionError("fixed-route payload length/canonical side is not invariant")
        row.pop("_canonical_xz")
        result[signature] = row
    return result


def choose_best_physical_state(
    frontier: list[State],
    *,
    side_compression: dict[int, dict[str, Any]],
    mask_compressed_bytes: int,
    base_raw_sse: Decimal,
    base_container_bytes: int,
    panel_values: int,
    max_bpw: Decimal,
) -> tuple[State, Decimal, dict[str, int]]:
    feasible: list[tuple[Decimal, int, tuple[str, ...], State, dict[str, int]]] = []
    for state in frontier:
        selected_sse = base_raw_sse - state.savings
        xz_bytes = int(side_compression[state.alphabet_mask]["compressed_bytes"])
        physical_prelude = WFOUTR01_HEADER_BYTES + xz_bytes + mask_compressed_bytes
        total_bytes = base_container_bytes + state.byte_delta + physical_prelude
        if selected_sse <= 0 or total_bytes <= 0:
            continue
        rate = Decimal(8 * total_bytes) / Decimal(panel_values)
        if rate >= max_bpw:
            continue
        score = objective_log(selected_sse, total_bytes, panel_values)
        metrics = {
            "header_bytes": WFOUTR01_HEADER_BYTES,
            "side_xz_bytes": xz_bytes,
            "mask_bz2_bytes": mask_compressed_bytes,
            "physical_prelude_bytes": physical_prelude,
            "physical_bundle_bytes": total_bytes,
        }
        feasible.append((score, total_bytes, state.choice_ids, state, metrics))
    if not feasible:
        raise AssertionError(f"no state satisfies strict physical {max_bpw} bpw")
    score, _, _, state, metrics = min(
        feasible, key=lambda item: (item[0], item[1], item[2])
    )
    return state, score, metrics


def run_packer(
    python: Path,
    packer: Path,
    cwd: Path,
    manifest: Path,
    side_name: str,
    receipt_name: str,
) -> tuple[Path, dict[str, Any]]:
    command = [
        str(python),
        str(packer),
        "--manifest",
        str(manifest),
        "--output",
        side_name,
        "--receipt",
        receipt_name,
    ]
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    (cwd / f"{receipt_name}.packer.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"side packer failed ({completed.returncode})")
    side_path = cwd / side_name
    receipt_path = cwd / receipt_name
    receipt = load_json(receipt_path)
    if not (
        receipt.get("status") == "exact round-trip passed"
        and receipt.get("exact_eof") is True
        and int(receipt["side_bytes"]) == side_path.stat().st_size
        and receipt["side_sha256"] == sha256_path(side_path)
    ):
        raise AssertionError("side packer receipt validation failed")
    return side_path, receipt


def run_bundle_packer(
    python: Path,
    packer: Path,
    cwd: Path,
    raw_mask: Path,
) -> tuple[Path, dict[str, Any]]:
    command = [
        str(python),
        str(packer),
        "--side",
        "side.bin",
        "--container-dir",
        "containers",
        "--raw-mask",
        str(raw_mask),
        "--output",
        "selected.wfouter",
        "--receipt",
        "bundle.receipt.json",
    ]
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    (cwd / "bundle.receipt.json.packer.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"WFOUTR01 bundle packer failed ({completed.returncode})")
    bundle = cwd / "selected.wfouter"
    receipt_path = cwd / "bundle.receipt.json"
    receipt = load_json(receipt_path)
    if not (
        receipt.get("status") == "passed"
        and receipt.get("source_free_reparse_passed") is True
        and int(receipt["bundle_bytes"]) == bundle.stat().st_size
        and receipt["bundle_sha256"] == sha256_path(bundle)
        and int(receipt["header_bytes"]) == WFOUTR01_HEADER_BYTES
        and receipt["containers"].get("exact_eof") is True
    ):
        raise AssertionError("WFOUTR01 bundle packer receipt validation failed")
    return bundle, receipt


def decimal_string(value: Decimal) -> str:
    return str(+value)


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def choice_receipt(choice: Choice, base_bytes: int) -> dict[str, Any]:
    return {
        "option_id": choice.option_id,
        "kind": choice.kind,
        "alphabet_size": choice.alphabet_size,
        "container_path": str(choice.container_path),
        "container_bytes": choice.container_bytes,
        "byte_delta_from_base": choice.container_bytes - base_bytes,
        "container_sha256": choice.container_sha256,
        "escape_count": choice.escape_count,
        "base_raw_sse_decimal": decimal_string(choice.base_raw_sse),
        "raw_sse_decimal": decimal_string(choice.raw_sse),
        "raw_sse_savings_decimal": decimal_string(choice.savings),
    }


def stage_selection(args: argparse.Namespace) -> Path:
    manifest_path = args.manifest.resolve()
    base_dir = args.base_dir.resolve()
    candidate_path = args.candidate_receipt.resolve()
    packer = args.packer.resolve()
    bundle_packer = args.bundle_packer.resolve()
    raw_mask = args.raw_mask.resolve()
    # Do not resolve a virtual-environment interpreter symlink: executing its
    # system target directly can bypass pyvenv.cfg and silently change the
    # package/runtime environment used by the side packer.
    python = args.python.absolute()
    output = args.output_dir.resolve()
    for label, path in (
        ("manifest", manifest_path),
        ("candidate receipt", candidate_path),
        ("side packer", packer),
        ("bundle packer", bundle_packer),
        ("raw mask", raw_mask),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if is_within(output, base_dir):
        raise ValueError("output directory must not be inside the immutable base directory")
    candidate_tree = candidate_path.parent
    if is_within(output, candidate_tree):
        raise ValueError("output directory must not be inside the immutable candidate directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    manifest_decimal = load_decimal_json(manifest_path)
    chunks, base_container_bytes, validated_base = validate_base_run(
        manifest_path, manifest, base_dir
    )
    _, panel_values = validate_manifest(manifest)
    triggered, option_groups, options_by_index, source_energy_audit = (
        validate_adaptive_receipt(
        candidate_path,
        manifest_path,
        base_dir / "run.receipt.json",
        raw_mask,
        chunks,
        validated_base,
        )
    )
    candidate_contract = load_json(candidate_path)
    candidate_dependency_hashes = {
        field: candidate_contract[field]
        for field in (
            "implementation_sha256",
            "pinned_runner_core_sha256",
            "pinned_repacker_core_sha256",
            "encoder_sha256",
            "repacker_sha256",
            "scorer_sha256",
            "decoder_sha256",
        )
    }
    base_raw_sse = require_decimal(args.base_total_raw_sse, "aggregate base raw SSE", positive=True)
    triggered_base_sse = sum(
        (options_by_index[index][0].base_raw_sse for index in triggered), Decimal(0)
    )
    if triggered_base_sse > base_raw_sse:
        raise AssertionError(
            "sum of triggered base SSE exceeds externally supplied aggregate base SSE"
        )
    source_energy = args.total_raw_energy
    energy_origin = "--total-raw-energy"
    if source_energy is None:
        try:
            source_energy = require_decimal(
                manifest_decimal["ideal_projection"]["source_energy"],
                "manifest source energy",
                positive=True,
            )
            energy_origin = "manifest.ideal_projection.source_energy"
        except (KeyError, TypeError):
            source_energy = None
            energy_origin = "unavailable"

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent))
    )
    try:
        # Establish the exact original raw side.  It is only an input to XZ;
        # unlike the previous proxy accounting, its 47 KB raw length is never
        # mislabeled or charged as the WFOUTR01 physical prelude.
        original_side, original_side_receipt = run_packer(
            python,
            packer,
            temporary,
            manifest_path,
            "original.side.bin",
            "original.side.receipt.json",
        )
        original_side_blob = original_side.read_bytes()
        raw_side_bytes = len(original_side_blob)
        compressed_original_side = lzma.compress(
            original_side_blob, format=lzma.FORMAT_XZ, preset=9
        )
        mask_blob = raw_mask.read_bytes()
        compressed_mask = bz2.compress(mask_blob, compresslevel=9)
        mask_compressed_bytes = len(compressed_mask)

        # Keep the old constant-raw-side answer as a clearly labeled diagnostic.
        proxy_frontier = pareto_dp(option_groups)
        proxy_state, proxy_log_objective = choose_best_state(
            proxy_frontier,
            base_raw_sse=base_raw_sse,
            base_container_bytes=base_container_bytes,
            constant_proxy_bytes=raw_side_bytes,
            panel_values=panel_values,
            max_bpw=args.max_bpw,
        )
        if args.physical_selection == "exact":
            frontier, peak_states = signature_pareto_dp(
                option_groups, max_states=args.max_exact_states
            )
            selection_guarantee = (
                "global exact over every candidate combination; dominance applied only "
                "within byte-identical A64/A128 side signatures"
            )
        else:
            frontier = proxy_frontier
            peak_states = len(frontier)
            selection_guarantee = (
                "exact WFOUTR01 rerank only over the constant-raw-side proxy frontier; "
                "not a global optimum because cross-signature dominated states were pruned"
            )
        if args.physical_selection == "fixed-route":
            frontier = proxy_frontier
            peak_states = len(frontier)
            selection_guarantee = (
                "global exact: side codec 3 has one fixed 50-byte route and an "
                "alphabet-invariant canonical all-A64 XZ member"
            )
            side_compression = compress_fixed_route_sides(
                frontier,
                original_side=original_side_blob,
                chunks=chunks,
                triggered=triggered,
            )
        else:
            side_compression = compress_signature_sides(
                frontier,
                original_side=original_side_blob,
                chunks=chunks,
                triggered=triggered,
                workers=args.compression_workers,
                max_signatures=args.max_compression_signatures,
            )
        base_side_payload_bytes = (
            int(next(iter(side_compression.values()))["compressed_bytes"])
            if args.physical_selection == "fixed-route"
            else len(compressed_original_side)
        )
        selected_state, selected_log_objective, modeled_physical = (
            choose_best_physical_state(
                frontier,
                side_compression=side_compression,
                mask_compressed_bytes=mask_compressed_bytes,
                base_raw_sse=base_raw_sse,
                base_container_bytes=base_container_bytes,
                panel_values=panel_values,
                max_bpw=args.max_bpw,
            )
        )
        selected_by_index: dict[int, Choice] = {}
        for index, option_id in zip(triggered, selected_state.choice_ids, strict=True):
            matches = [
                choice for choice in options_by_index[index] if choice.option_id == option_id
            ]
            if len(matches) != 1:
                raise AssertionError(f"selection is not unique for chunk {index}: {option_id}")
            selected_by_index[index] = matches[0]

        selected_manifest = json.loads(json.dumps(manifest))
        for index, choice in selected_by_index.items():
            selected_manifest["chunks"][index]["alphabet_size"] = choice.alphabet_size
        selected_manifest_path = temporary / "selected.manifest.json"
        selected_manifest_path.write_text(
            json.dumps(selected_manifest, indent=2) + "\n", encoding="utf-8"
        )
        side_path, side_receipt = run_packer(
            python,
            packer,
            temporary,
            Path("selected.manifest.json"),
            "side.bin",
            "side.receipt.json",
        )
        selected_raw_side_bytes = side_path.stat().st_size
        if selected_raw_side_bytes != raw_side_bytes:
            raise AssertionError(
                "fixed-width WFPLTE01 raw side changed length after alphabet overrides"
            )
        modeled_selected_side = side_blob_for_signature(
            original_side_blob, chunks, triggered, selected_state.alphabet_mask
        )
        if side_path.read_bytes() != modeled_selected_side:
            raise AssertionError(
                "selected packer side is not byte-identical to physical-rerank side"
            )
        selected_side_model = side_compression[selected_state.alphabet_mask]
        if args.physical_selection == "fixed-route":
            actual_model = fixed_route_payload(side_path.read_bytes(), chunks)
            if any(
                actual_model[key] != selected_side_model[key]
                for key in (
                    "raw_sha256",
                    "canonical_raw_sha256",
                    "canonical_xz_bytes",
                    "canonical_xz_sha256",
                    "route_bytes",
                    "route_sha256",
                    "compressed_bytes",
                    "compressed_sha256",
                )
            ):
                raise AssertionError("selected fixed-route payload differs from DP model")
        else:
            actual_side_xz = lzma.compress(
                side_path.read_bytes(), format=lzma.FORMAT_XZ, preset=9
            )
            if not (
                len(actual_side_xz) == selected_side_model["compressed_bytes"]
                and hashlib.sha256(actual_side_xz).hexdigest()
                == selected_side_model["compressed_sha256"]
            ):
                raise AssertionError("selected XZ side differs from physical rerank model")

        staged_dir = temporary / "containers"
        staged_dir.mkdir()
        mapping = []
        chosen_count = 0
        selected_container_bytes = 0
        for index, chunk in enumerate(chunks):
            base = validated_base[index]
            if index in selected_by_index:
                choice = selected_by_index[index]
                selected_raw_sse: str | None = decimal_string(choice.raw_sse)
                saving = decimal_string(choice.savings)
            else:
                choice = Choice(
                    chunk_index=index,
                    option_id="base",
                    kind="base",
                    alphabet_size=int(chunk["alphabet_size"]),
                    container_path=base["container_path"],
                    container_bytes=int(base["container"]["container_bytes"]),
                    container_sha256=str(base["container"]["sha256"]),
                    raw_sse=Decimal(0),
                    base_raw_sse=Decimal(0),
                    savings=Decimal(0),
                    escape_count=int(base["container"]["escape_count"]),
                )
                selected_raw_sse = None
                saving = "0"
            destination = staged_dir / f"wf-{index:03d}.polar.bin"
            shutil.copyfile(choice.container_path, destination)
            staged_sha = sha256_path(destination)
            staged_bytes = destination.stat().st_size
            if staged_sha != choice.container_sha256 or staged_bytes != choice.container_bytes:
                raise AssertionError(f"staged copy mismatch for chunk {index}")
            parsed = parse_container(destination, expected_escape_count=choice.escape_count)
            if parsed["sha256"] != choice.container_sha256:
                raise AssertionError(f"post-stage container parse mismatch for chunk {index}")
            mapping.append(
                {
                    "chunk_index": index,
                    "option_id": choice.option_id,
                    "kind": choice.kind,
                    "base_alphabet_size": int(chunk["alphabet_size"]),
                    "selected_alphabet_size": choice.alphabet_size,
                    "source_container_path": str(choice.container_path),
                    "staged_container": f"containers/{destination.name}",
                    "container_bytes": staged_bytes,
                    "container_sha256": staged_sha,
                    "escape_count": choice.escape_count,
                    "triggered": index in options_by_index,
                    "selected_raw_sse_decimal_if_triggered": selected_raw_sse,
                    "raw_sse_savings_decimal": saving,
                }
            )
            selected_container_bytes += staged_bytes
            chosen_count += 1
        if chosen_count != CHUNK_COUNT or len(mapping) != CHUNK_COUNT:
            raise AssertionError("did not stage exactly one container per chunk")
        if selected_container_bytes != base_container_bytes + selected_state.byte_delta:
            raise AssertionError("selected container total does not match DP byte delta")

        bundle_path, bundle_receipt = run_bundle_packer(
            python, bundle_packer, temporary, raw_mask
        )
        physical_prelude_bytes = (
            int(bundle_receipt["header_bytes"])
            + int(bundle_receipt["side"]["compressed_bytes"])
            + int(bundle_receipt["mask"]["compressed_bytes"])
        )
        physical_bundle_bytes = int(bundle_receipt["bundle_bytes"])
        if not (
            int(bundle_receipt["side"]["raw_bytes"]) == selected_raw_side_bytes
            and bundle_receipt["side"]["raw_sha256"] == sha256_path(side_path)
            and int(bundle_receipt["side"]["compressed_bytes"])
            == int(selected_side_model["compressed_bytes"])
            and bundle_receipt["side"]["compressed_sha256"]
            == selected_side_model["compressed_sha256"]
            and int(bundle_receipt["mask"]["raw_bytes"]) == len(mask_blob)
            and bundle_receipt["mask"]["raw_sha256"]
            == hashlib.sha256(mask_blob).hexdigest()
            and int(bundle_receipt["mask"]["compressed_bytes"])
            == mask_compressed_bytes
            and bundle_receipt["mask"]["compressed_sha256"]
            == hashlib.sha256(compressed_mask).hexdigest()
            and int(bundle_receipt["containers"]["count"]) == CHUNK_COUNT
            and int(bundle_receipt["containers"]["bytes"]) == selected_container_bytes
            and bundle_receipt["containers"]["ordered_sha256"]
            == [row["container_sha256"] for row in mapping]
            and physical_prelude_bytes == modeled_physical["physical_prelude_bytes"]
            and physical_bundle_bytes == modeled_physical["physical_bundle_bytes"]
            and physical_bundle_bytes
            == physical_prelude_bytes + selected_container_bytes
        ):
            raise AssertionError("literal WFOUTR01 receipt differs from selection model")

        selected_raw_sse = base_raw_sse - selected_state.savings
        actual_bpw = Decimal(8 * physical_bundle_bytes) / Decimal(panel_values)
        if actual_bpw >= args.max_bpw:
            raise AssertionError("post-pack physical WFOUTR01 rate violates strict budget")
        recomputed_log = objective_log(
            selected_raw_sse, physical_bundle_bytes, panel_values
        )
        if recomputed_log != selected_log_objective:
            raise AssertionError("packed WFOUTR01 objective differs from selected DP objective")
        with localcontext() as context:
            context.prec = 80
            ln2 = Decimal(2).ln()
            ln10 = Decimal(10).ln()
            objective = recomputed_log.exp()
            base_comparison_bytes = (
                base_container_bytes
                + WFOUTR01_HEADER_BYTES
                + base_side_payload_bytes
                + mask_compressed_bytes
            )
            base_log = objective_log(
                base_raw_sse, base_comparison_bytes, panel_values
            )
            objective_change_db = Decimal(10) * (recomputed_log - base_log) / ln10
            raw_relative_mse = (
                selected_raw_sse / source_energy if source_energy is not None else None
            )
            gap_db = (
                Decimal(10)
                * (
                    raw_relative_mse.ln()
                    + ln2
                    * Decimal(16 * physical_bundle_bytes)
                    / Decimal(panel_values)
                )
                / ln10
                if raw_relative_mse is not None
                else None
            )

        considered = {
            str(index): [
                choice_receipt(
                    choice, int(validated_base[index]["container"]["container_bytes"])
                )
                for choice in options_by_index[index]
            ]
            for index in triggered
        }
        frontier_rows = [
            {
                "alphabet_signature_hex": hex(state.alphabet_mask),
                "byte_delta": state.byte_delta,
                "savings_decimal": decimal_string(state.savings),
                "choice_ids": list(state.choice_ids),
            }
            for state in frontier
        ]
        selection_receipt = {
            "format": (
                "continuous PLTE exact adaptive selection receipt fixed-route v2"
                if args.physical_selection == "fixed-route"
                else "continuous PLTE exact adaptive selection receipt v1"
            ),
            "status": "passed",
            "strict_ptq": True,
            "write_once_output": True,
            "objective": "(base_raw_sse - savings) * 2**(16 * WFOUTR01_bundle_bytes / panel_values)",
            "physical_selection_mode": args.physical_selection,
            "selection_guarantee": selection_guarantee,
            "selection_arithmetic": {
                "byte_axis": "exact integer physical bytes",
                "sse_axis": "exact Decimal values parsed from serialized candidate JSON",
                "transcendental_comparison": "Python Decimal ln at 80-digit precision",
                "tie_break": "objective, then total bytes, then lexicographic option IDs",
                "pareto_rule_exact_mode": "discard a state only within an identical A64/A128 signature iff another has no more container bytes and no less SSE saving",
                "side_physical_model": (
                    "codec 3: one XZ(canonical all-A64 side) plus fixed 50-byte route400; selected bytes verified against isolated v2 packer"
                    if args.physical_selection == "fixed-route"
                    else "Python lzma FORMAT_XZ preset 9 for every retained signature; selected bytes verified against pack_bundle.py"
                ),
                "mask_physical_model": "Python bz2 level 9 once; selected bytes verified against pack_bundle.py",
            },
            "inputs": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_path(manifest_path),
                "base_dir": str(base_dir),
                "base_run_receipt_sha256": sha256_path(base_dir / "run.receipt.json"),
                "candidate_receipt_path": str(candidate_path),
                "candidate_receipt_sha256": sha256_path(candidate_path),
                "candidate_v3_declared_dependency_hashes": candidate_dependency_hashes,
                "packer_path": str(packer),
                "packer_sha256": sha256_path(packer),
                "bundle_packer_path": str(bundle_packer),
                "bundle_packer_sha256": sha256_path(bundle_packer),
                "raw_mask_path": str(raw_mask),
                "raw_mask_bytes": len(mask_blob),
                "raw_mask_sha256": hashlib.sha256(mask_blob).hexdigest(),
                "python": str(python),
                "base_total_raw_sse_decimal": decimal_string(base_raw_sse),
                "total_raw_energy_decimal": (
                    decimal_string(source_energy) if source_energy is not None else None
                ),
                "total_raw_energy_origin": energy_origin,
            },
            "validation": {
                "canonical_chunks": CHUNK_COUNT,
                "one_staged_option_per_chunk": chosen_count == CHUNK_COUNT,
                "all_base_reports_and_containers_hash_validated": True,
                "all_adaptive_reports_decodes_and_container_hashes_validated": True,
                "source_energy_validation": source_energy_audit,
                "tail_payload_identity_validated": True,
                "original_raw_side_bytes": raw_side_bytes,
                "selected_raw_side_bytes": selected_raw_side_bytes,
                "raw_side_length_invariant_after_alphabet_regeneration": True,
                "selected_raw_side_matches_reranked_signature_byte_for_byte": True,
                "selected_physical_side_payload_matches_rerank_hash_and_length": True,
                "selected_bz2_mask_matches_rerank_hash_and_length": True,
                "selected_manifest_side_roundtrip": side_receipt["status"],
                "wfoutr01_source_free_reparse": bundle_receipt[
                    "source_free_reparse_passed"
                ],
            },
            "dp": {
                "triggered_chunks": triggered,
                "triggered_count": len(triggered),
                "option_count": sum(len(group) for group in option_groups),
                "proxy_frontier_states": len(proxy_frontier),
                "physical_frontier_states": len(frontier),
                "peak_exact_signature_states": peak_states,
                "distinct_physical_side_signatures": len(side_compression),
                "fixed_route_bits": (
                    FIXED_ROUTE_BITS
                    if args.physical_selection == "fixed-route"
                    else None
                ),
                "frontier_sha256": canonical_sha256(frontier_rows),
                "selected_alphabet_signature_hex": hex(selected_state.alphabet_mask),
                "selected_choice_ids": list(selected_state.choice_ids),
                "selected_byte_delta": selected_state.byte_delta,
                "selected_raw_sse_savings_decimal": decimal_string(selected_state.savings),
                "raw_side_constant_proxy_selected_choice_ids": list(
                    proxy_state.choice_ids
                ),
                "raw_side_constant_proxy_log_objective_decimal": decimal_string(
                    proxy_log_objective
                ),
                "raw_side_constant_proxy_is_claim_accounting": False,
            },
            "artifacts": {
                "selected_manifest": "selected.manifest.json",
                "selected_manifest_sha256": sha256_path(selected_manifest_path),
                "literal_side": "side.bin",
                "literal_side_raw_bytes": selected_raw_side_bytes,
                "literal_side_sha256": sha256_path(side_path),
                "side_receipt": "side.receipt.json",
                "side_receipt_sha256": sha256_path(temporary / "side.receipt.json"),
                "container_directory": "containers",
                "physical_bundle": "selected.wfouter",
                "physical_bundle_bytes": physical_bundle_bytes,
                "physical_bundle_sha256": sha256_path(bundle_path),
                "bundle_receipt": "bundle.receipt.json",
                "bundle_receipt_sha256": sha256_path(
                    temporary / "bundle.receipt.json"
                ),
            },
            "accounting": {
                "panel_values": panel_values,
                "base_container_bytes": base_container_bytes,
                "base_original_side_physical_payload_bytes": base_side_payload_bytes,
                "base_modeled_physical_bundle_bytes": base_comparison_bytes,
                "selected_container_bytes": selected_container_bytes,
                "raw_side_bytes_not_directly_charged": selected_raw_side_bytes,
                "wfoutr01_header_bytes": int(bundle_receipt["header_bytes"]),
                "side_physical_payload_bytes": int(
                    bundle_receipt["side"]["compressed_bytes"]
                ),
                "side_canonical_xz_bytes": (
                    int(bundle_receipt["side"]["canonical_xz_bytes"])
                    if args.physical_selection == "fixed-route"
                    else int(bundle_receipt["side"]["compressed_bytes"])
                ),
                "side_fixed_route_bytes": (
                    int(bundle_receipt["side"]["route_bytes"])
                    if args.physical_selection == "fixed-route"
                    else 0
                ),
                "mask_bz2_compressed_bytes": int(
                    bundle_receipt["mask"]["compressed_bytes"]
                ),
                "physical_prelude_bytes": physical_prelude_bytes,
                "physical_bundle_bytes": physical_bundle_bytes,
                "strict_max_bpw_decimal": decimal_string(args.max_bpw),
                "physical_all_in_bpw_decimal": decimal_string(actual_bpw),
                "strict_rate_pass": actual_bpw < args.max_bpw,
                "selected_total_raw_sse_decimal": decimal_string(selected_raw_sse),
                "raw_relative_mse_decimal": (
                    decimal_string(raw_relative_mse) if raw_relative_mse is not None else None
                ),
                "gaussian_reference_gap_db_decimal": (
                    decimal_string(gap_db) if gap_db is not None else None
                ),
                "objective_decimal": decimal_string(objective),
                "objective_change_from_base_db_decimal": decimal_string(
                    objective_change_db
                ),
            },
            "options_considered": considered,
            "selection_map": mapping,
        }
        receipt_path = temporary / "selection.receipt.json"
        receipt_path.write_text(
            json.dumps(selection_receipt, indent=2) + "\n", encoding="utf-8"
        )
        receipt_sha = sha256_path(receipt_path)
        (temporary / "selection.receipt.sha256").write_text(
            f"{receipt_sha}  selection.receipt.json\n", encoding="ascii"
        )

        # Original-manifest pack outputs only established the actual fixed
        # prelude length.  They are not part of the selected literal stream.
        for path in (
            original_side,
            temporary / "original.side.receipt.json",
            temporary / "original.side.receipt.json.packer.log",
        ):
            path.unlink()
        os.replace(temporary, output)
        return output / "selection.receipt.json"
    except BaseException:
        print(f"selection failed; preserved diagnostic staging tree: {temporary}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--base-total-raw-sse", type=decimal_arg, required=True)
    parser.add_argument("--total-raw-energy", type=decimal_arg)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--bundle-packer", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-bpw", type=decimal_arg, default=Decimal("2.5"))
    parser.add_argument(
        "--physical-selection",
        choices=("exact", "proxy", "fixed-route"),
        default="exact",
        help="fixed-route is globally exact with isolated side codec 3",
    )
    parser.add_argument("--max-exact-states", type=int, default=1_000_000)
    parser.add_argument("--max-compression-signatures", type=int, default=100_000)
    parser.add_argument("--compression-workers", type=int, default=8)
    args = parser.parse_args()
    if args.base_total_raw_sse <= 0:
        parser.error("--base-total-raw-sse must be positive")
    if args.total_raw_energy is not None and args.total_raw_energy <= 0:
        parser.error("--total-raw-energy must be positive")
    if args.max_bpw <= 0:
        parser.error("--max-bpw must be positive")
    if args.max_exact_states <= 0 or args.max_compression_signatures <= 0:
        parser.error("exact state/signature limits must be positive")
    if args.compression_workers <= 0:
        parser.error("--compression-workers must be positive")
    receipt = stage_selection(args)
    summary = load_json(receipt)
    print(
        json.dumps(
            {
                "selection_receipt": str(receipt),
                "selection_receipt_sha256": sha256_path(receipt),
                **summary["accounting"],
                "selected_choice_ids": summary["dp"]["selected_choice_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
