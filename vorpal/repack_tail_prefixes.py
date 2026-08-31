#!/usr/bin/env python3
"""Decode a base PLTE chunk once and append exact sparse-tail prefixes.

This is an exploratory optimization utility.  It preserves the arithmetic
payload byte-for-byte and reproduces the encoder's stable normalized-residual
ranking.  It also scores every prefix in original Qwen coordinates with CuPy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path

import cupy as cp
import numpy as np


LOGICAL_LENGTH_BITS = 20
ESCAPE_RECORD_BITS = 34


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tail_prefix_decoder", path)
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


def bf16_u16(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2")


def bf16_float(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)


def pack_escape_records(positions: np.ndarray, values: np.ndarray) -> bytes:
    if positions.size != values.size:
        raise ValueError("escape position/value length mismatch")
    combined = 0
    for position, value in zip(positions, values, strict=True):
        p = int(position)
        if p < 0 or p >= (1 << 18):
            raise ValueError(f"escape position outside 18-bit block: {p}")
        combined = (combined << ESCAPE_RECORD_BITS) | (p << 16) | int(value)
    meaningful_bits = ESCAPE_RECORD_BITS * int(positions.size)
    padding_bits = (-meaningful_bits) % 8
    combined <<= padding_bits
    return combined.to_bytes((meaningful_bits + padding_bits) // 8, "big")


def decode_base(dec, raw_mask: Path, metadata: dict, payload: bytes, logical_bits: int) -> np.ndarray:
    parameters = metadata["parameters"]
    trial = metadata["trials"][0]
    n = int(parameters["block_length"])
    levels = int(math.log2(int(parameters["alphabet_size"])))
    reverse = dec.bit_reverse_indices(n)
    layers = dec.sc_layers(n)
    base_levels = min(levels, 6)
    flags = dec.raw_frozen_flags(raw_mask, n, base_levels)
    flags.extend(np.zeros(n, dtype=np.uint8) for _ in range(levels - base_levels))
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    alphabet_size = int(parameters["alphabet_size"])
    alphabet = eta * np.arange(
        -alphabet_size // 2 + 1, alphabet_size // 2 + 1, dtype=np.float64
    )
    weights = np.exp(-0.5 * (alphabet / sigma_recon) ** 2)
    arithmetic = dec.ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    seed = int(parameters["seed"])
    trial_index = int(trial["trial"])
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(seed + 104729 * trial_index + 1000003 * level)
        frozen = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = dec.leaf_prior_ratios(weights, previous, level)
        decoded_x, _ = dec.decode_sc_level(
            prior_lr, flags[level_index], frozen, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
    return alphabet[previous]


def original_arrays(manifest: dict, chunk: dict, repo: Path) -> tuple[np.ndarray, np.ndarray]:
    group_values = int(manifest["parameters"]["group_values"])
    source_cache: dict[int, np.ndarray] = {}
    raw_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for member in chunk["members"]:
        block_ordinal = int(member["block_ordinal"])
        if block_ordinal not in source_cache:
            source_path = Path(manifest["blocks"][block_ordinal]["source_path"])
            if not source_path.is_absolute():
                source_path = repo / source_path
            source_cache[block_ordinal] = bf16_float(bf16_u16(source_path)).astype(np.float64)
        group_index = int(member["group_index"])
        begin = group_index * group_values
        raw_parts.append(source_cache[block_ordinal][begin : begin + group_values])
        scale_parts.append(
            np.full(group_values, float(member["qscale"]), dtype=np.float64)
        )
    return np.concatenate(raw_parts), np.concatenate(scale_parts)


def parse_expected(rows: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for row in rows:
        key, value = row.split("=", 1)
        result[int(key)] = Path(value)
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
    parser.add_argument("--ranking", choices=("normalized", "raw-gain"), default="normalized")
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args()

    dec = load_module(args.decoder)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][args.chunk_index]
    if int(chunk["chunk_index"]) != args.chunk_index:
        raise AssertionError("chunk order mismatch")
    logical_bits, scale, payload, base_bytes, positions, _, padding_zero = dec.read_plte_container(
        args.base_container
    )
    if positions.size or not padding_zero:
        raise AssertionError("input must be a clean base container")
    expected_base = int(metadata["trials"][0]["literal_container_bytes"])
    if base_bytes != expected_base:
        raise AssertionError((base_bytes, expected_base))

    base_reconstruction = decode_base(dec, args.raw_mask, metadata, payload, logical_bits) * scale
    normalized_u16 = bf16_u16(Path(chunk["normalized_source"]))
    normalized = bf16_float(normalized_u16).astype(np.float64)
    if normalized.size != base_reconstruction.size:
        raise AssertionError("source/reconstruction length mismatch")
    normalized_gpu = cp.asarray(normalized)
    base_gpu = cp.asarray(base_reconstruction)
    raw, qscale = original_arrays(manifest, chunk, args.repo)
    raw_gpu = cp.asarray(raw)
    qscale_gpu = cp.asarray(qscale)
    raw_energy = float(cp.sum(cp.square(raw_gpu), dtype=cp.float64).get())
    norm_energy = float(cp.sum(cp.square(normalized_gpu), dtype=cp.float64).get())
    base_norm_sse = float(cp.sum(cp.square(normalized_gpu - base_gpu), dtype=cp.float64).get())
    base_raw_sse = float(
        cp.sum(cp.square(raw_gpu - qscale_gpu * base_gpu), dtype=cp.float64).get()
    )
    normalized_errors = cp.asnumpy(cp.square(normalized_gpu - base_gpu))
    escaped_raw_gpu = qscale_gpu * normalized_gpu
    raw_gains = cp.asnumpy(
        cp.square(raw_gpu - qscale_gpu * base_gpu)
        - cp.square(raw_gpu - escaped_raw_gpu)
    )
    ranking_values = normalized_errors if args.ranking == "normalized" else raw_gains
    ranking = np.argsort(-ranking_values, kind="stable")
    if args.ranking == "raw-gain":
        positive = int(np.count_nonzero(raw_gains[ranking] > 0.0))
        if max(args.ks) > positive:
            raise ValueError(f"only {positive} positive raw-gain escapes for requested prefixes")
    expected = parse_expected(args.expect)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in sorted(set(args.ks)):
        if k <= 0 or k >= (1 << 12):
            raise ValueError(f"invalid prefix length {k}")
        chosen = np.sort(ranking[:k].astype(np.int32))
        chosen_values = normalized_u16[chosen].copy()
        tail = pack_escape_records(chosen, chosen_values)
        header = logical_bits | (k << LOGICAL_LENGTH_BITS)
        container = struct.pack("<If", header, scale) + payload + tail
        parsed = dec.read_plte_container_bytes(container) if hasattr(dec, "read_plte_container_bytes") else None
        if len(container) != base_bytes + (ESCAPE_RECORD_BITS * k + 7) // 8:
            raise AssertionError("unexpected packed length")
        if k in expected and container != expected[k].read_bytes():
            raise AssertionError(f"k={k} does not reproduce {expected[k]}")
        reconstruction = base_gpu.copy()
        reconstruction[cp.asarray(chosen)] = normalized_gpu[cp.asarray(chosen)]
        norm_sse = float(
            cp.sum(cp.square(normalized_gpu - reconstruction), dtype=cp.float64).get()
        )
        raw_sse = float(
            cp.sum(
                cp.square(raw_gpu - qscale_gpu * reconstruction), dtype=cp.float64
            ).get()
        )
        actual_bpw = 8.0 * len(container) / normalized.size
        output = args.output_dir / f"wf-{args.chunk_index:03d}-k{k}.polar.bin"
        output.write_bytes(container)
        row = {
            "chunk_index": args.chunk_index,
            "escape_count": k,
            "container_bytes": len(container),
            "container_sha256": sha256_bytes(container),
            "container_path": str(output),
            "payload_unchanged": container[8 : 8 + len(payload)] == payload,
            "expected_container_byte_equal": k not in expected or container == expected[k].read_bytes(),
            "normalized_sse": norm_sse,
            "normalized_relative_mse": norm_sse / norm_energy,
            "normalized_sse_reduction": base_norm_sse - norm_sse,
            "raw_source_energy": raw_energy,
            "raw_sse": raw_sse,
            "raw_relative_mse": raw_sse / raw_energy,
            "raw_sse_reduction": base_raw_sse - raw_sse,
            "actual_container_bpw": actual_bpw,
            "raw_gap_at_actual_container_rate_db": 10.0
            * math.log10((raw_sse / raw_energy) / (2.0 ** (-2.0 * actual_bpw))),
        }
        rows.append(row)

    result = {
        "format": "exploratory PLTE sparse-tail prefix repack v1",
        "strict_ptq": True,
        "chunk_index": args.chunk_index,
        "base_container_bytes": base_bytes,
        "base_container_sha256": sha256_path(args.base_container),
        "base_metadata_sha256": sha256_path(args.base_metadata),
        "manifest_sha256": sha256_path(args.manifest),
        "decoder_sha256": sha256_path(args.decoder),
        "raw_mask_sha256": sha256_path(args.raw_mask),
        "normalized_source_sha256": sha256_path(Path(chunk["normalized_source"])),
        "base_normalized_sse": base_norm_sse,
        "base_raw_sse": base_raw_sse,
        "raw_source_energy": raw_energy,
        "stable_ranking": (
            "descending normalized squared residual, stable ordinal ties"
            if args.ranking == "normalized"
            else "descending original-coordinate SSE gain, stable ordinal ties"
        ),
        "all_expected_containers_byte_equal": all(
            row["expected_container_byte_equal"] for row in rows
        ),
        "rows": rows,
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    report = args.output_dir / f"wf-{args.chunk_index:03d}-tail-prefixes.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
