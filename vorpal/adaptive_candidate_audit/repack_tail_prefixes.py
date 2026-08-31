#!/usr/bin/env python3
"""Hardened isolated audit copy of the sparse-tail prefix repacker.

This file is intentionally outside the tracked implementation.  It decodes an
already validated base PLTE container through the published clean decoder,
ranks exact original-coordinate SSE gains, emits deterministic tail prefixes,
and reparses every physical candidate before scoring it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
from pathlib import Path

import cupy as cp
import numpy as np


N = 1 << 18
GROUP_VALUES = 1 << 11
GROUPS_PER_CHUNK = N // GROUP_VALUES
LOGICAL_LENGTH_BITS = 20
ESCAPE_RECORD_BITS = 34
MAX_ESCAPE_RECORDS = (1 << (32 - LOGICAL_LENGTH_BITS)) - 1
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tail_prefix_clean_decoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def bf16_u16(path: Path, expected_values: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype="<u2")
    if expected_values is not None and values.size != expected_values:
        raise ValueError(
            f"{path}: expected {expected_values} BF16 values, found {values.size}"
        )
    return values


def bf16_float(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)


def pack_escape_records(positions: np.ndarray, values: np.ndarray) -> bytes:
    positions = np.asarray(positions, dtype=np.int64)
    values = np.asarray(values, dtype=np.uint16)
    if positions.size != values.size:
        raise ValueError("escape position/value length mismatch")
    if positions.size and (
        np.any(positions < 0)
        or np.any(positions >= N)
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("escape positions must be strictly increasing and in range")
    combined = 0
    for position, value in zip(positions, values, strict=True):
        combined = (
            combined << ESCAPE_RECORD_BITS
        ) | (int(position) << 16) | int(value)
    meaningful_bits = ESCAPE_RECORD_BITS * int(positions.size)
    padding_bits = (-meaningful_bits) % 8
    combined <<= padding_bits
    return combined.to_bytes((meaningful_bits + padding_bits) // 8, "big")


def parse_container_bytes(
    container: bytes,
) -> tuple[int, float, bytes, np.ndarray, np.ndarray, int, int]:
    """Independent exact-EOF parser for one u20/tail PLTE frame."""

    if len(container) < 8:
        raise ValueError("PLTE container shorter than fixed header")
    header_word, scale = struct.unpack("<If", container[:8])
    if not math.isfinite(scale):
        raise ValueError("non-finite decoder scale")
    logical_bits = header_word & ((1 << LOGICAL_LENGTH_BITS) - 1)
    escape_count = header_word >> LOGICAL_LENGTH_BITS
    payload_bytes = (logical_bits + 7) // 8
    tail_bytes = (ESCAPE_RECORD_BITS * escape_count + 7) // 8
    expected = 8 + payload_bytes + tail_bytes
    if len(container) != expected:
        raise ValueError(f"container expected {expected} bytes, found {len(container)}")
    payload = container[8 : 8 + payload_bytes]
    arithmetic_padding = payload_bytes * 8 - logical_bits
    if arithmetic_padding and payload[-1] & ((1 << arithmetic_padding) - 1):
        raise ValueError("nonzero arithmetic padding")
    tail = container[8 + payload_bytes :]
    meaningful_tail_bits = ESCAPE_RECORD_BITS * escape_count
    tail_padding = tail_bytes * 8 - meaningful_tail_bits
    combined = int.from_bytes(tail, "big")
    if tail_padding and combined & ((1 << tail_padding) - 1):
        raise ValueError("nonzero sparse-tail padding")
    combined >>= tail_padding
    positions = np.empty(escape_count, dtype=np.int32)
    values = np.empty(escape_count, dtype=np.uint16)
    record_mask = (1 << ESCAPE_RECORD_BITS) - 1
    for index in range(escape_count - 1, -1, -1):
        record = combined & record_mask
        combined >>= ESCAPE_RECORD_BITS
        positions[index] = record >> 16
        values[index] = record & 0xFFFF
    if combined:
        raise AssertionError("tail parser left meaningful bits")
    if escape_count and (
        np.any(positions < 0)
        or np.any(positions >= N)
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("invalid sparse-tail positions")
    return (
        logical_bits,
        float(scale),
        payload,
        positions,
        values,
        arithmetic_padding,
        tail_padding,
    )


def decode_base(
    decoder,
    raw_mask: Path,
    metadata: dict,
    payload: bytes,
    logical_bits: int,
) -> np.ndarray:
    parameters = metadata["parameters"]
    trial = metadata["trials"][0]
    n = int(parameters["block_length"])
    alphabet_size = int(parameters["alphabet_size"])
    levels = int(math.log2(alphabet_size))
    if (1 << levels) != alphabet_size or alphabet_size not in (64, 128):
        raise ValueError(f"unsupported base alphabet {alphabet_size}")
    reverse = decoder.bit_reverse_indices(n)
    layers = decoder.sc_layers(n)
    base_levels = min(levels, 6)
    flags = decoder.raw_frozen_flags(raw_mask, n, base_levels)
    flags.extend(np.zeros(n, dtype=np.uint8) for _ in range(levels - base_levels))
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])
    if not (math.isfinite(sigma_source) and math.isfinite(distortion) and math.isfinite(eta)):
        raise ValueError("non-finite PLTE profile")
    if sigma_source <= 0.0 or not 0.0 < distortion < sigma_source**2 or eta <= 0.0:
        raise ValueError("invalid PLTE profile")
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    alphabet = eta * np.arange(
        -alphabet_size // 2 + 1, alphabet_size // 2 + 1, dtype=np.float64
    )
    weights = np.exp(-0.5 * (alphabet / sigma_recon) ** 2)
    arithmetic = decoder.ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    seed = int(parameters["seed"])
    trial_index = int(trial["trial"])
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(
            seed + 104729 * trial_index + 1000003 * level
        )
        frozen = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = decoder.leaf_prior_ratios(weights, previous, level)
        decoded_x, _ = decoder.decode_sc_level(
            prior_lr, flags[level_index], frozen, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
    return alphabet[previous]


def resolve_path(path_text: str, repo: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo / path


def validate_metadata(
    metadata: dict,
    chunk: dict,
    normalized_path: Path,
    base_container: Path,
) -> tuple[dict, dict]:
    parameters = metadata["parameters"]
    trials = metadata["trials"]
    if len(trials) != 1:
        raise AssertionError("base metadata must contain exactly one trial")
    trial = trials[0]
    required = (
        int(parameters["block_length"]) == N,
        int(parameters["alphabet_size"]) == int(chunk["alphabet_size"]),
        int(parameters["alphabet_size"]) in (64, 128),
        int(parameters["container_cap_bytes"]) == 0,
        float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"]),
        float(parameters["eta"]) == float(chunk["eta"]),
        int(trial["trial"]) == 0,
        int(trial["tail_escape_count"]) == 0,
        int(trial["literal_container_bytes"]) == base_container.stat().st_size,
        trial["literal_container_sha256"] == sha256_path(base_container),
        trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"],
        sha256_path(normalized_path) == chunk["normalized_source_sha256"],
        all(trial[field] is True for field in ROUNDTRIP_FIELDS),
    )
    if not all(required):
        raise AssertionError("base report/container/profile validation failed")
    if not math.isfinite(float(trial["gap_db"])):
        raise ValueError("base gap is non-finite")
    return parameters, trial


def original_arrays(
    manifest: dict,
    chunk: dict,
    repo: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    group_values = int(manifest["parameters"]["group_values"])
    groups_per_block = int(manifest["parameters"]["groups_per_polar_block"])
    if group_values != GROUP_VALUES or groups_per_block != GROUPS_PER_CHUNK:
        raise AssertionError("unexpected manifest group geometry")
    members = chunk["members"]
    if len(members) != GROUPS_PER_CHUNK:
        raise AssertionError("chunk must contain exactly 128 members")
    source_cache: dict[int, np.ndarray] = {}
    source_bindings: dict[int, dict[str, object]] = {}
    raw_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    seen_ordinals: set[int] = set()
    for member in members:
        block_ordinal = int(member["block_ordinal"])
        group_index = int(member["group_index"])
        canonical = int(member["canonical_group_ordinal"])
        if canonical != block_ordinal * groups_per_block + group_index:
            raise AssertionError("member canonical ordinal mismatch")
        if canonical in seen_ordinals:
            raise AssertionError("duplicate canonical group member")
        seen_ordinals.add(canonical)
        if not 0 <= block_ordinal < len(manifest["blocks"]):
            raise IndexError("member block ordinal out of range")
        if not 0 <= group_index < groups_per_block:
            raise IndexError("member group index out of range")
        if block_ordinal not in source_cache:
            block = manifest["blocks"][block_ordinal]
            source_path = resolve_path(str(block["source_path"]), repo)
            actual_hash = sha256_path(source_path)
            if actual_hash != str(block["source_sha256"]):
                raise ValueError(f"raw source hash mismatch for block {block_ordinal}")
            words = bf16_u16(source_path, N)
            source_cache[block_ordinal] = bf16_float(words).astype(np.float64)
            source_bindings[block_ordinal] = {
                "block_ordinal": block_ordinal,
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": actual_hash,
            }
        begin = group_index * group_values
        raw_parts.append(source_cache[block_ordinal][begin : begin + group_values])
        qscale = float(member["qscale"])
        if not math.isfinite(qscale) or qscale <= 0.0:
            raise ValueError("member qscale must be finite and positive")
        scale_parts.append(np.full(group_values, qscale, dtype=np.float64))
    return (
        np.concatenate(raw_parts),
        np.concatenate(scale_parts),
        [source_bindings[index] for index in sorted(source_bindings)],
    )


def parse_expected(rows: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"expected K=PATH, got {row!r}")
        key, value = row.split("=", 1)
        k = int(key)
        if k in result:
            raise ValueError(f"duplicate expected prefix {k}")
        result[k] = Path(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--base-container", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", required=True)
    parser.add_argument(
        "--ranking", choices=("normalized", "raw-gain"), default="raw-gain"
    )
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args()

    requested_ks = tuple(sorted(set(int(value) for value in args.ks)))
    if not requested_ks or any(k <= 0 or k > MAX_ESCAPE_RECORDS for k in requested_ks):
        raise ValueError(f"invalid tail prefix lengths {requested_ks}")
    if not args.decoder.is_file() or not args.raw_mask.is_file():
        raise FileNotFoundError("clean decoder or raw mask missing")
    decoder = load_module(args.decoder)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    if len(chunks) != 400 or [int(row["chunk_index"]) for row in chunks] != list(range(400)):
        raise AssertionError("expected 400 canonical manifest chunks")
    if not 0 <= args.chunk_index < len(chunks):
        raise IndexError("chunk index out of range")
    chunk = chunks[args.chunk_index]
    normalized_path = resolve_path(str(chunk["normalized_source"]), args.repo)
    normalized_u16 = bf16_u16(normalized_path, N)
    validate_metadata(metadata, chunk, normalized_path, args.base_container)

    base_literal = args.base_container.read_bytes()
    (
        logical_bits,
        scale,
        payload,
        base_positions,
        _base_values,
        arithmetic_padding,
        tail_padding,
    ) = parse_container_bytes(base_literal)
    trial = metadata["trials"][0]
    if base_positions.size or tail_padding:
        raise AssertionError("input must be a zero-tail base container")
    if arithmetic_padding and payload[-1] & ((1 << arithmetic_padding) - 1):
        raise AssertionError("base arithmetic padding is nonzero")
    if logical_bits != int(trial["arithmetic_logical_bits"]):
        raise AssertionError("base logical-bit count mismatch")
    if sha256_bytes(payload) != str(trial["arithmetic_payload_sha256"]):
        raise AssertionError("base arithmetic payload hash mismatch")

    base_reconstruction = (
        decode_base(decoder, args.raw_mask, metadata, payload, logical_bits) * scale
    )
    normalized = bf16_float(normalized_u16).astype(np.float64)
    if normalized.size != base_reconstruction.size:
        raise AssertionError("source/reconstruction length mismatch")
    normalized_gpu = cp.asarray(normalized)
    base_gpu = cp.asarray(base_reconstruction)
    raw, qscale, raw_source_bindings = original_arrays(manifest, chunk, args.repo)
    raw_gpu = cp.asarray(raw)
    qscale_gpu = cp.asarray(qscale)
    raw_energy = float(cp.sum(cp.square(raw_gpu), dtype=cp.float64).get())
    norm_energy = float(cp.sum(cp.square(normalized_gpu), dtype=cp.float64).get())
    base_norm_sse = float(
        cp.sum(cp.square(normalized_gpu - base_gpu), dtype=cp.float64).get()
    )
    base_raw_error = cp.square(raw_gpu - qscale_gpu * base_gpu)
    escaped_raw_gpu = qscale_gpu * normalized_gpu
    escaped_raw_error = cp.square(raw_gpu - escaped_raw_gpu)
    base_raw_sse = float(cp.sum(base_raw_error, dtype=cp.float64).get())
    normalized_errors = cp.asnumpy(cp.square(normalized_gpu - base_gpu))
    raw_gains = cp.asnumpy(base_raw_error - escaped_raw_error)
    if not (
        math.isfinite(raw_energy)
        and raw_energy > 0.0
        and math.isfinite(norm_energy)
        and norm_energy > 0.0
        and np.all(np.isfinite(normalized_errors))
        and np.all(np.isfinite(raw_gains))
    ):
        raise ValueError("non-finite source energy, residual, or raw-coordinate gain")
    ranking_values = normalized_errors if args.ranking == "normalized" else raw_gains
    ranking = np.argsort(-ranking_values, kind="stable")
    if args.ranking == "raw-gain":
        positive = int(np.count_nonzero(raw_gains[ranking] > 0.0))
        if requested_ks[-1] > positive:
            raise ValueError(
                f"only {positive} positive raw-gain escapes for requested prefixes"
            )
    expected = parse_expected(args.expect)
    unknown_expected = sorted(set(expected) - set(requested_ks))
    if unknown_expected:
        raise ValueError(f"expected byte checks not requested: {unknown_expected}")

    rows = []
    previous_raw_reduction = -math.inf
    for k in requested_ks:
        chosen = np.sort(ranking[:k].astype(np.int32))
        chosen_values = normalized_u16[chosen].copy()
        tail = pack_escape_records(chosen, chosen_values)
        header = logical_bits | (k << LOGICAL_LENGTH_BITS)
        container = struct.pack("<If", header, scale) + payload + tail
        expected_bytes = len(base_literal) + (ESCAPE_RECORD_BITS * k + 7) // 8
        if len(container) != expected_bytes:
            raise AssertionError("unexpected packed length")
        if k in expected and container != expected[k].read_bytes():
            raise AssertionError(f"k={k} does not reproduce expected container")

        (
            parsed_bits,
            parsed_scale,
            parsed_payload,
            parsed_positions,
            parsed_values,
            parsed_arithmetic_padding,
            parsed_tail_padding,
        ) = parse_container_bytes(container)
        if not (
            parsed_bits == logical_bits
            and struct.pack("<f", parsed_scale) == struct.pack("<f", scale)
            and parsed_payload == payload
            and np.array_equal(parsed_positions, chosen)
            and np.array_equal(parsed_values, chosen_values)
            and parsed_arithmetic_padding == arithmetic_padding
        ):
            raise AssertionError("independent candidate reparse mismatch")

        # Score the reconstruction obtained from the reparsed physical bytes,
        # never from the pre-pack chosen arrays.
        reconstruction = base_gpu.copy()
        escaped = (
            (parsed_values.astype(np.uint32) << np.uint32(16)).view(np.float32)
        )
        reconstruction[cp.asarray(parsed_positions)] = cp.asarray(
            escaped, dtype=cp.float64
        )
        norm_sse = float(
            cp.sum(cp.square(normalized_gpu - reconstruction), dtype=cp.float64).get()
        )
        raw_sse = float(
            cp.sum(
                cp.square(raw_gpu - qscale_gpu * reconstruction), dtype=cp.float64
            ).get()
        )
        raw_reduction = base_raw_sse - raw_sse
        analytic_reduction = float(np.sum(raw_gains[ranking[:k]], dtype=np.float64))
        tolerance = max(1e-12, 2e-12 * max(1.0, abs(base_raw_sse)))
        if not math.isclose(raw_reduction, analytic_reduction, rel_tol=2e-12, abs_tol=tolerance):
            raise AssertionError(
                f"raw SSE-gain identity failed for k={k}: "
                f"{raw_reduction} != {analytic_reduction}"
            )
        if args.ranking == "raw-gain" and raw_reduction + tolerance < previous_raw_reduction:
            raise AssertionError("raw-gain prefixes are not monotone")
        previous_raw_reduction = raw_reduction

        output = args.output_dir / f"wf-{args.chunk_index:03d}-k{k}.polar.bin"
        atomic_write_bytes(output, container)
        emitted = output.read_bytes()
        if emitted != container:
            raise AssertionError("post-write candidate bytes changed")
        parse_container_bytes(emitted)
        actual_bpw = 8.0 * len(emitted) / normalized.size
        row = {
            "chunk_index": args.chunk_index,
            "escape_count": k,
            "container_bytes": len(emitted),
            "incremental_tail_bytes": len(emitted) - len(base_literal),
            "meaningful_tail_bits": ESCAPE_RECORD_BITS * k,
            "tail_padding_bits": parsed_tail_padding,
            "container_sha256": sha256_bytes(emitted),
            "container_path": str(output),
            "payload_unchanged": parsed_payload == payload,
            "independent_physical_reparse_passed": True,
            "parsed_tail_applied_for_scoring": True,
            "expected_container_byte_equal": k not in expected or emitted == expected[k].read_bytes(),
            "normalized_sse": norm_sse,
            "normalized_relative_mse": norm_sse / norm_energy,
            "normalized_sse_reduction": base_norm_sse - norm_sse,
            "raw_source_energy": raw_energy,
            "raw_sse": raw_sse,
            "raw_relative_mse": raw_sse / raw_energy,
            "raw_sse_reduction": raw_reduction,
            "analytic_selected_raw_gain_sum": analytic_reduction,
            "raw_gain_identity_passed": True,
            "actual_container_bpw": actual_bpw,
            "raw_gap_at_actual_container_rate_db": 10.0
            * math.log10((raw_sse / raw_energy) / (2.0 ** (-2.0 * actual_bpw))),
        }
        rows.append(row)

    result = {
        "format": "exploratory PLTE sparse-tail prefix repack audit v2",
        "strict_ptq": True,
        "training_or_retraining": False,
        "chunk_index": args.chunk_index,
        "implementation_sha256": sha256_path(Path(__file__)),
        "base_container_bytes": len(base_literal),
        "base_container_sha256": sha256_path(args.base_container),
        "base_metadata_sha256": sha256_path(args.base_metadata),
        "manifest_sha256": sha256_path(args.manifest),
        "decoder_sha256": sha256_path(args.decoder),
        "raw_mask_sha256": sha256_path(args.raw_mask),
        "normalized_source_path": str(normalized_path),
        "normalized_source_bytes": normalized_path.stat().st_size,
        "normalized_source_sha256": sha256_path(normalized_path),
        "raw_sources": raw_source_bindings,
        "base_normalized_sse": base_norm_sse,
        "base_raw_sse": base_raw_sse,
        "raw_source_energy": raw_energy,
        "stable_ranking": (
            "descending normalized squared residual, stable ordinal ties"
            if args.ranking == "normalized"
            else "descending original-coordinate SSE gain, stable ordinal ties"
        ),
        "requested_escape_counts": list(requested_ks),
        "base_decoded_with_clean_decoder": True,
        "all_candidates_independently_reparsed": True,
        "all_scores_apply_reparsed_tail_bytes": True,
        "all_raw_gain_identities_passed": True,
        "all_expected_containers_byte_equal": all(
            row["expected_container_byte_equal"] for row in rows
        ),
        "rows": rows,
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    report = args.output_dir / f"wf-{args.chunk_index:03d}-tail-prefixes.json"
    atomic_write_json(report, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
