#!/usr/bin/env python3
"""Clean exploratory decoder/scorer for one continuous-waterfill PLTE chunk."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cupy as cp
import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("clean_plte_decoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_values(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dec = load_module(args.decoder)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][args.chunk_index]
    if int(chunk["chunk_index"]) != args.chunk_index:
        raise AssertionError("chunk manifest order mismatch")
    parameters = metadata["parameters"]
    trial = metadata["trials"][0]
    n = int(parameters["block_length"])
    if n != 1 << 18 or len(chunk["members"]) != 128:
        raise AssertionError("unexpected chunk geometry")
    levels = int(math.log2(int(parameters["alphabet_size"])))
    logical_bits, scale, payload, container_bytes, escape_pos, escape_u16, padding_zero = dec.read_plte_container(args.container)
    if logical_bits != int(trial["arithmetic_logical_bits"]):
        raise AssertionError("logical bit count mismatch")
    if sha256_path(args.container) != trial["literal_container_sha256"]:
        raise AssertionError("container hash mismatch")
    if hashlib.sha256(payload).hexdigest() != trial["arithmetic_payload_sha256"]:
        raise AssertionError("payload hash mismatch")
    if not padding_zero:
        raise AssertionError("nonzero tail padding")

    reverse = dec.bit_reverse_indices(n)
    layers = dec.sc_layers(n)
    capacities = [float(value) for value in parameters["capacity_schedule"]]
    base_levels = min(levels, 6)
    flags = dec.raw_frozen_flags(args.raw_mask, n, base_levels)
    # Higher lattice bitplanes are procedurally fully open in the encoder.
    flags.extend(np.zeros(n, dtype=np.uint8) for _ in range(levels - base_levels))
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    alphabet_size = int(parameters["alphabet_size"])
    alphabet = eta * np.arange(-alphabet_size // 2 + 1, alphabet_size // 2 + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (alphabet / sigma_recon) ** 2)
    arithmetic = dec.ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected_count = 0
    seed = int(parameters["seed"])
    trial_index = int(trial["trial"])
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(seed + 104729 * trial_index + 1000003 * level)
        frozen = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = dec.leaf_prior_ratios(weights, previous, level)
        decoded_x, frequencies = dec.decode_sc_level(
            prior_lr, flags[level_index], frozen, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_count += frequencies.size
    reconstruction = alphabet[previous] * float(scale)
    if escape_pos.size:
        escaped = (escape_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        reconstruction[escape_pos] = escaped.astype(np.float64)

    normalized_path = Path(chunk["normalized_source"])
    normalized_source = bf16_values(normalized_path).astype(np.float64)
    norm_source_gpu = cp.asarray(normalized_source)
    norm_recon_gpu = cp.asarray(reconstruction)
    norm_sse = float(cp.sum(cp.square(norm_source_gpu - norm_recon_gpu), dtype=cp.float64).get())
    norm_energy = float(cp.sum(cp.square(norm_source_gpu), dtype=cp.float64).get())
    normalized_relative_mse = norm_sse / norm_energy
    expected = float(trial["relative_mse"])
    if abs(normalized_relative_mse - expected) > 1e-12:
        raise AssertionError(f"normalized decode mismatch: {normalized_relative_mse} vs {expected}")

    source_cache: dict[int, np.ndarray] = {}
    raw_energy = 0.0
    raw_sse = 0.0
    group_values = int(manifest["parameters"]["group_values"])
    for position, member in enumerate(chunk["members"]):
        block_ordinal = int(member["block_ordinal"])
        if block_ordinal not in source_cache:
            source_path = Path(manifest["blocks"][block_ordinal]["source_path"])
            if not source_path.is_absolute():
                source_path = args.repo / source_path
            source_cache[block_ordinal] = bf16_values(source_path).astype(np.float64)
        group_index = int(member["group_index"])
        begin = group_index * group_values
        end = begin + group_values
        raw = cp.asarray(source_cache[block_ordinal][begin:end])
        decoded = cp.asarray(
            reconstruction[position * group_values : (position + 1) * group_values]
            * float(member["qscale"])
        )
        raw_energy += float(cp.sum(cp.square(raw), dtype=cp.float64).get())
        raw_sse += float(cp.sum(cp.square(raw - decoded), dtype=cp.float64).get())

    raw_relative_mse = raw_sse / raw_energy
    actual_bpw = container_bytes * 8.0 / n
    result = {
        "format": "continuous reverse-waterfilled PLTE clean chunk decode v1",
        "status": "passed",
        "strict_ptq": True,
        "chunk_index": args.chunk_index,
        "nominal_rate": float(chunk["nominal_rate"]),
        "actual_container_bpw": actual_bpw,
        "container_bytes": container_bytes,
        "container_sha256": sha256_path(args.container),
        "logical_bits": logical_bits,
        "selected_symbols": selected_count,
        "frequency_u16_sha256": frequency_hash.hexdigest(),
        "reconstruction_indices_sha256": hashlib.sha256(previous.astype("<i2", copy=False).tobytes()).hexdigest(),
        "normalized_relative_mse": normalized_relative_mse,
        "encoder_relative_mse": expected,
        "raw_source_energy": raw_energy,
        "raw_sse": raw_sse,
        "raw_relative_mse": raw_relative_mse,
        "raw_gap_at_actual_container_rate_db": 10.0 * math.log10(raw_relative_mse / (2.0 ** (-2.0 * actual_bpw))),
        "normalized_roundtrip_matches_at_1e_12": True,
        "tail_escape_count": int(escape_pos.size),
        "tail_padding_zero": bool(padding_zero),
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
