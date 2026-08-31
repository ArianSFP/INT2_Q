#!/usr/bin/env python3
"""Clean-process decoder audit for the experimental polar-lattice container.

This file intentionally does not import ``agent_root_polar_lattice_gate`` and
does not consume encoder-generated probability arrays.  It reconstructs the
causal entropy model from the public reliability tables, fixed codec
parameters, the deterministic frozen-coset seed, and previously decoded bits.

It is an audit tool, not a proposed final container specification: the current
container contains only ``u32 logical_bits``, ``f32 scale``, and the arithmetic
payload.  The JSON sidecar and reliability tables are therefore required side
information and are reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def bit_reverse_indices(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    if 1 << depth != n:
        raise ValueError("block length must be a power of two")
    source = np.arange(n, dtype=np.uint32)
    result = np.zeros(n, dtype=np.uint32)
    for _ in range(depth):
        result = (result << 1) | (source & 1)
        source >>= 1
    return result.astype(np.int64)


def sc_layers(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    result = np.ones(n + 1, dtype=np.int32)
    result[0] = depth
    for one_based in range(2, n + 1):
        layer = 1
        cursor = one_based
        while cursor % 2 == 1:
            layer += 1
            cursor = (cursor + 1) // 2
        result[one_based - 1] = layer
    return result


def polar_transform(bits: np.ndarray) -> np.ndarray:
    result = np.asarray(bits, dtype=np.uint8).copy()
    stride = 1
    while stride < result.size:
        rows = result.reshape(-1, 2 * stride)
        rows[:, :stride] ^= rows[:, stride:]
        stride *= 2
    return result


def reliability_flags(
    table_dir: Path, n: int, capacities: list[float]
) -> list[np.ndarray]:
    reverse = bit_reverse_indices(n)
    logn = int(math.log2(n))
    result: list[np.ndarray] = []
    for level, capacity in enumerate(capacities, start=1):
        keep = min(n, max(0, int(math.ceil(n * capacity))))
        flag = np.zeros(n, dtype=np.uint8)
        if keep == 0:
            flag[:] = 1
        elif keep != n and level <= 3:
            filename = (
                f"Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_"
                f"Lvl_{level}_n_{logn}.mat"
            )
            pe = np.asarray(loadmat(table_dir / filename)["PeLast"]).ravel()
            if pe.size != n:
                raise ValueError(f"unexpected reliability-table size in {filename}")
            ordered = np.argsort(pe[reverse], kind="stable")
            freeze_index = np.sort(ordered[keep:])
            flag[reverse[freeze_index]] = 1
        elif keep != n:
            flag[: n - keep] = 1
        result.append(flag)
    return result


def frozen_map_flags(path: Path, n: int, levels: int) -> list[np.ndarray]:
    """Load the serialized decoder map without importing any encoder code."""
    saved = np.load(path, allow_pickle=False)
    if int(saved["block_length"]) != n or int(saved["levels"]) != levels:
        raise ValueError("decoder-map geometry mismatch")
    if str(saved["bitorder"]) != "little":
        raise ValueError("decoder-map bit order mismatch")
    packed = np.asarray(saved["packed_freeze_flags"], dtype=np.uint8)
    if packed.shape != (levels, (n + 7) // 8):
        raise ValueError(f"unexpected packed decoder-map shape: {packed.shape}")
    return [
        np.unpackbits(packed[level], bitorder="little")[:n].astype(np.uint8)
        for level in range(levels)
    ]


def raw_frozen_flags(path: Path, n: int, levels: int) -> list[np.ndarray]:
    """Load the PLTE six-mask artifact (level-major, MSB-first)."""
    payload = path.read_bytes()
    stride = (n + 7) // 8
    expected = levels * stride
    if len(payload) != expected:
        raise ValueError(f"raw mask bytes {len(payload)} != {expected}")
    return [
        np.unpackbits(
            np.frombuffer(payload[level * stride : (level + 1) * stride], dtype=np.uint8),
            bitorder="big",
        )[:n].astype(np.uint8)
        for level in range(levels)
    ]


def leaf_prior_ratios(
    weights: np.ndarray, previous: np.ndarray, level: int
) -> np.ndarray:
    lower_mod = 1 << (level - 1)
    bit_value = 1 << (level - 1)
    ratios = np.empty(lower_mod, dtype=np.float64)
    indices = np.arange(weights.size)
    for context in range(lower_mod):
        matching = indices % lower_mod == context
        mass0 = weights[matching & ((indices & bit_value) == 0)].sum()
        mass1 = weights[matching & ((indices & bit_value) != 0)].sum()
        ratios[context] = mass0 / max(float(mass1), 1e-300)
    return np.clip(ratios[previous], 1e-30, 1e30)


class ArithmeticBinaryDecoder:
    def __init__(self, payload: bytes, logical_bits: int):
        if logical_bits < 0 or logical_bits > len(payload) * 8:
            raise ValueError("logical payload length does not fit payload bytes")
        self.full = 1 << 32
        self.half = 1 << 31
        self.quarter = 1 << 30
        self.three_quarters = 3 << 30
        self.bits = np.unpackbits(
            np.frombuffer(payload, dtype=np.uint8), bitorder="big"
        )
        self.logical_bits = logical_bits
        self.cursor = 0
        self.low = 0
        self.high = self.full - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()

    def _read(self) -> int:
        if self.cursor >= self.logical_bits:
            return 0
        value = int(self.bits[self.cursor])
        self.cursor += 1
        return value

    def decode(self, freq1: int) -> int:
        f1 = min(65535, max(1, int(freq1)))
        f0 = 65536 - f1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * 65536 - 1) // width
        split = self.low + width * f0 // 65536 - 1
        if scaled < f0:
            value = 0
            self.high = split
        else:
            value = 1
            self.low = split + 1
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & (self.full - 1)
            self.high = ((self.high << 1) & (self.full - 1)) | 1
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()
        return value


def decode_sc_level(
    leaf_lr: np.ndarray,
    freeze_flag: np.ndarray,
    frozen_external: np.ndarray,
    reverse: np.ndarray,
    layers: np.ndarray,
    arithmetic: ArithmeticBinaryDecoder,
) -> tuple[np.ndarray, np.ndarray]:
    n = leaf_lr.size
    depth = int(math.log2(n))
    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    frequencies: list[int] = []

    def bounded(value: np.ndarray | float) -> np.ndarray | float:
        return np.clip(value, 1e-30, 1e30)

    for i0 in range(n):
        one_based = i0 + 1
        if one_based == 1:
            end = int(layers[i0])
            col = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            lr_reg[:, col] = bounded((left * right + 1.0) / (left + right))
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[: count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right)
                )
        elif one_based == n // 2 + 1:
            end = int(layers[i0])
            col = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, col] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[: count // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right)
                )
        elif one_based % 2 == 0:
            end = int(layers[i0])
            left = float(lr_reg[0, end])
            right = float(lr_reg[1, end])
            used = int(mu_reg[0, 0])
            lr_reg[0, end - 1] = bounded(left ** (1 - 2 * used) * right)
        else:
            end = int(layers[i0])
            count = 1 << end
            left = lr_reg[0:count:2, end]
            right = lr_reg[1:count:2, end]
            used = mu_reg[: count // 2, end - 1].astype(np.int8)
            lr_reg[: count // 2, end - 1] = bounded(
                np.power(left, 1 - 2 * used) * right
            )
            for layer in range(end - 1, 0, -1):
                count2 = 1 << layer
                left = lr_reg[0:count2:2, layer]
                right = lr_reg[1:count2:2, layer]
                lr_reg[: count2 // 2, layer - 1] = bounded(
                    (left * right + 1.0) / (left + right)
                )

        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        if freeze_flag[i0]:
            value = int(frozen_external[reverse[i0]])
        else:
            p1 = 1.0 / (1.0 + root_lr)
            freq1 = min(65535, max(1, int(math.floor(p1 * 65536.0 + 0.5))))
            frequencies.append(freq1)
            value = arithmetic.decode(freq1)
        internal[i0] = value

        if one_based % 2 == 1:
            mu_reg[0, 0] = value
        else:
            end = int(layers[one_based])
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = value
            for layer in range(1, end):
                length = 1 << (layer - 1)
                left = mu_reg[:length, layer - 1]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[: 2 * length] = merged
            mu_reg[: 1 << max(end - 1, 0), end - 1] = temp

    return polar_transform(internal[reverse]), np.asarray(frequencies, dtype=np.uint16)


def read_first_container(path: Path) -> tuple[int, float, bytes, int]:
    container = path.read_bytes()
    if len(container) < 8:
        raise ValueError("container is shorter than its fixed header")
    logical_bits, scale = struct.unpack("<If", container[:8])
    payload_bytes = (logical_bits + 7) // 8
    expected = 8 + payload_bytes
    if len(container) != expected:
        raise ValueError(
            f"single-block audit expected {expected} bytes, found {len(container)}"
        )
    return logical_bits, float(scale), container[8:], len(container)


def read_plte_container(
    path: Path,
) -> tuple[int, float, bytes, int, np.ndarray, np.ndarray, bool]:
    """Parse the fixed-budget PLTE core plus 34-bit sparse BF16 records."""
    container = path.read_bytes()
    if len(container) < 8:
        raise ValueError("PLTE container is shorter than its header")
    header_word, scale = struct.unpack("<If", container[:8])
    logical_bits = header_word & ((1 << 20) - 1)
    count = header_word >> 20
    payload_bytes = (logical_bits + 7) // 8
    tail_bytes = (34 * count + 7) // 8
    expected = 8 + payload_bytes + tail_bytes
    if len(container) != expected:
        raise ValueError(f"PLTE container expected {expected} bytes, found {len(container)}")
    payload = container[8 : 8 + payload_bytes]
    tail = container[8 + payload_bytes :]
    meaningful = 34 * count
    padding = 8 * len(tail) - meaningful
    combined = int.from_bytes(tail, "big")
    padding_zero = (combined & ((1 << padding) - 1)) == 0 if padding else True
    combined >>= padding
    positions = np.empty(count, dtype=np.int32)
    values = np.empty(count, dtype=np.uint16)
    mask = (1 << 34) - 1
    for index in range(count - 1, -1, -1):
        record = combined & mask
        combined >>= 34
        positions[index] = record >> 16
        values[index] = record & 0xFFFF
    if count and np.any(positions[1:] <= positions[:-1]):
        raise ValueError("PLTE tail positions are not strictly increasing")
    return logical_bits, float(scale), payload, len(container), positions, values, padding_zero


def read_fixed_fp16_slot(
    path: Path, payload_bits: int
) -> tuple[int, float, bytes, int]:
    container = path.read_bytes()
    expected = 2 + (payload_bits + 7) // 8
    if len(container) != expected:
        raise ValueError(
            f"fixed-slot audit expected {expected} physical bytes, found "
            f"{len(container)}"
        )
    scale = float(np.frombuffer(container[:2], dtype="<f2")[0])
    payload = container[2:]
    trailing = len(payload) * 8 - payload_bits
    if trailing:
        bits = np.unpackbits(np.frombuffer(payload[-1:], dtype=np.uint8), bitorder="big")
        if np.any(bits[8 - trailing :]):
            raise ValueError("nonzero physical tail padding in fixed slot")
    return payload_bits, scale, payload, len(container)


def write_fixed_fp16_slot(
    output: Path,
    scale: float,
    payload: bytes,
    logical_bits: int,
    fixed_payload_bits: int,
) -> None:
    if logical_bits > fixed_payload_bits:
        raise ValueError(
            f"arithmetic payload overflow: {logical_bits} > {fixed_payload_bits}"
        )
    source_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8), bitorder="big"
    )[:logical_bits]
    slot_bits = np.zeros(fixed_payload_bits, dtype=np.uint8)
    slot_bits[:logical_bits] = source_bits
    packed = np.packbits(slot_bits, bitorder="big").tobytes()
    scale_bytes = np.asarray([scale], dtype="<f2").tobytes()
    output.write_bytes(scale_bytes + packed)


def bf16_values(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument(
        "--container-layout",
        choices=("plte-u20-tail-fp32", "current-u32-fp32", "fixed-fp16-slot"),
        default="plte-u20-tail-fp32",
    )
    parser.add_argument("--fixed-slot-payload-bits", type=int, default=563500)
    parser.add_argument("--write-fixed-fp16-slot", type=Path)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--metadata", type=Path, required=True)
    map_group = parser.add_mutually_exclusive_group(required=True)
    map_group.add_argument("--table-dir", type=Path)
    map_group.add_argument("--map", type=Path)
    map_group.add_argument("--raw-mask", type=Path)
    parser.add_argument("--source-bf16", type=Path)
    parser.add_argument(
        "--regenerate-synthetic-cupy",
        action="store_true",
        help="regenerate the synthetic audit source with the recorded CuPy RNG",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    trial_row = metadata["trials"][0]
    n = int(parameters["block_length"])
    trial = int(trial_row["trial"])
    seed = int(parameters["seed"])
    levels = int(math.log2(int(parameters["alphabet_size"])))
    sigma_source = float(parameters["sigma_source"])
    distortion = float(parameters["test_channel_distortion"])
    eta = float(parameters["eta"])
    capacities = [float(value) for value in parameters["capacity_schedule"]]
    if len(capacities) != levels:
        raise ValueError("capacity schedule does not match the alphabet")

    escape_positions = np.empty(0, dtype=np.int32)
    escape_values = np.empty(0, dtype=np.uint16)
    escape_padding_zero = True
    if args.container_layout == "plte-u20-tail-fp32":
        if args.convert_only or args.write_fixed_fp16_slot is not None:
            raise ValueError("legacy fixed-slot conversion is not defined for PLTE tails")
        (
            logical_bits,
            scale,
            payload,
            container_bytes,
            escape_positions,
            escape_values,
            escape_padding_zero,
        ) = read_plte_container(args.container)
        if logical_bits != int(trial_row["arithmetic_logical_bits"]):
            raise AssertionError("PLTE container and metadata logical lengths disagree")
        if escape_positions.size != int(trial_row["tail_escape_count"]):
            raise AssertionError("PLTE container and metadata tail counts disagree")
        if hashlib.sha256(payload).hexdigest() != trial_row["arithmetic_payload_sha256"]:
            raise AssertionError("PLTE arithmetic payload hash does not match metadata")
        if not escape_padding_zero:
            raise AssertionError("PLTE escape padding is nonzero")
        if (
            hashlib.sha256(args.container.read_bytes()).hexdigest()
            != trial_row["literal_container_sha256"]
        ):
            raise AssertionError("PLTE container hash does not match metadata")
    elif args.container_layout == "current-u32-fp32":
        logical_bits, scale, payload, container_bytes = read_first_container(
            args.container
        )
        if logical_bits != int(trial_row["arithmetic_logical_bits"]):
            raise AssertionError("container and metadata logical lengths disagree")
        if hashlib.sha256(payload).hexdigest() != trial_row["arithmetic_payload_sha256"]:
            raise AssertionError("payload hash does not match metadata")
        if (
            hashlib.sha256(args.container.read_bytes()).hexdigest()
            != trial_row["literal_container_sha256"]
        ):
            raise AssertionError("container hash does not match metadata")
        if args.write_fixed_fp16_slot is not None:
            write_fixed_fp16_slot(
                args.write_fixed_fp16_slot,
                scale,
                payload,
                logical_bits,
                args.fixed_slot_payload_bits,
            )
            if args.convert_only:
                converted = {
                    "status": "converted current container to fixed FP16 slot",
                    "source_logical_bits": logical_bits,
                    "fixed_payload_bits": args.fixed_slot_payload_bits,
                    "zero_fill_bits": args.fixed_slot_payload_bits - logical_bits,
                    "fixed_slot_logical_bits_including_scale": (
                        16 + args.fixed_slot_payload_bits
                    ),
                    "physical_file_bytes_with_one_block_tail_padding": (
                        args.write_fixed_fp16_slot.stat().st_size
                    ),
                    "output_sha256": hashlib.sha256(
                        args.write_fixed_fp16_slot.read_bytes()
                    ).hexdigest(),
                }
                print(json.dumps(converted, indent=2))
                return
    else:
        if args.convert_only:
            raise ValueError("--convert-only requires the current container layout")
        if args.write_fixed_fp16_slot is not None:
            raise ValueError("cannot convert an already fixed-slot container")
        logical_bits, scale, payload, container_bytes = read_fixed_fp16_slot(
            args.container, args.fixed_slot_payload_bits
        )

    reverse = bit_reverse_indices(n)
    layers = sc_layers(n)
    if args.map is not None:
        flags = frozen_map_flags(args.map, n, levels)
    elif args.raw_mask is not None:
        flags = raw_frozen_flags(args.raw_mask, n, levels)
    else:
        flags = reliability_flags(args.table_dir, n, capacities)
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    alphabet_size = int(parameters["alphabet_size"])
    alphabet = eta * np.arange(
        -alphabet_size // 2 + 1, alphabet_size // 2 + 1, dtype=np.float64
    )
    weights = np.exp(-0.5 * (alphabet / sigma_recon) ** 2)
    arithmetic = ArithmeticBinaryDecoder(payload, logical_bits)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected_count = 0
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(
            seed + 104729 * trial + 1000003 * level
        )
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior_lr = leaf_prior_ratios(weights, previous, level)
        decoded_x, frequencies = decode_sc_level(
            prior_lr,
            flags[level_index],
            frozen_external,
            reverse,
            layers,
            arithmetic,
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_count += frequencies.size

    reconstruction = (alphabet[previous] * scale).astype(np.float64)
    if escape_positions.size:
        escape_float = (
            (escape_values.astype(np.uint32) << np.uint32(16)).view(np.float32)
        )
        reconstruction[escape_positions] = escape_float.astype(np.float64)
    result: dict[str, object] = {
        "status": "decoded in a clean implementation without encoder probabilities",
        "container_sha256": hashlib.sha256(args.container.read_bytes()).hexdigest(),
        "container_bytes": container_bytes,
        "logical_payload_bits": logical_bits,
        "payload_bytes": len(payload),
        "serialized_scale": scale,
        "selected_symbols_decoded": selected_count,
        "reconstruction_indices_sha256": hashlib.sha256(
            previous.astype("<i2", copy=False).tobytes()
        ).hexdigest(),
        "reconstruction_fp64_sha256": hashlib.sha256(
            reconstruction.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "causal_frequency_u16_sha256": frequency_hash.hexdigest(),
        "tail_escape_count": int(escape_positions.size),
        "tail_escape_padding_zero": escape_padding_zero,
        "tail_escape_positions_sha256": hashlib.sha256(
            escape_positions.astype("<i4", copy=False).tobytes()
        ).hexdigest(),
        "decoder_map_source": (
            str(args.map)
            if args.map is not None
            else str(args.raw_mask)
            if args.raw_mask is not None
            else str(args.table_dir)
        ),
        "decoder_required_side_information": (
            [
                "JSON codec parameters and capacity schedule",
                "serialized six-level frozen decoder map",
                "NumPy default_rng/PCG64 frozen-bit generation convention",
                "tensor/block ordering outside this one-block container",
            ]
            if args.map is not None
            else [
                "JSON codec parameters and capacity schedule",
                (
                    "serialized six-level raw frozen mask"
                    if args.raw_mask is not None
                    else "three public D=0.20 reliability tables"
                ),
                "NumPy default_rng/PCG64 frozen-bit generation convention",
                "tensor/block ordering outside this one-block container",
            ]
        ),
        "container_self_describing": False,
        "conditional_slot_budget_compatibility": {
            "container_layout": args.container_layout,
            "container_scale_bits": (
                16 if args.container_layout == "fixed-fp16-slot" else 32
            ),
            "budgeted_scale_bits": (
                32 if args.container_layout == "plte-u20-tail-fp32" else 16
            ),
            "container_length_header_bits_per_block": (
                0 if args.container_layout == "fixed-fp16-slot" else 32
            ),
            "budgeted_length_header_bits_per_block": (
                32 if args.container_layout == "plte-u20-tail-fp32" else 0
            ),
            "container_payload_logical_bits": logical_bits,
            "fixed_slot_cap_bytes": int(parameters["container_cap_bytes"]),
            "actual_container_bytes": container_bytes,
            "zero_padding_bytes_needed_by_hypothetical_packer": max(
                0, int(parameters["container_cap_bytes"]) - container_bytes
            ),
            "fits_conditional_fixed_slot_budget": (
                (
                    args.container_layout == "plte-u20-tail-fp32"
                    and container_bytes <= int(parameters["container_cap_bytes"])
                )
                or (
                    args.container_layout == "fixed-fp16-slot"
                    and logical_bits == args.fixed_slot_payload_bits
                )
            ),
            "realized_checkpoint_packer_exercised": False,
        },
    }
    source: np.ndarray | None = None
    if args.source_bf16 is not None:
        source_raw_all = np.fromfile(args.source_bf16, dtype="<u2")
        source_all = bf16_values(args.source_bf16)
        block_index = int(trial_row["source"]["block_index"])
        source_raw = source_raw_all[block_index * n : (block_index + 1) * n]
        source = source_all[block_index * n : (block_index + 1) * n].astype(
            np.float64
        )
        if source.size != n:
            raise ValueError("source file does not contain the audited block")
        source_hash = hashlib.sha256(source_raw.tobytes()).hexdigest()
        expected_source_hash = trial_row["source"].get("block_bf16_sha256")
        if expected_source_hash is not None and source_hash != expected_source_hash:
            raise AssertionError("source BF16 block hash does not match metadata")
        result["source_audit"] = "frozen BF16 source file"
        result["source_block_bf16_sha256"] = source_hash
    elif args.regenerate_synthetic_cupy:
        if trial_row["source"]["kind"] != "synthetic_gaussian":
            raise ValueError("metadata does not describe a synthetic source")
        import cupy as cp

        source = cp.asnumpy(
            cp.random.RandomState(seed + 104729 * trial).normal(
                0.0, sigma_source, size=n, dtype=cp.float64
            )
        )
        result["source_audit"] = "regenerated from recorded CuPy RandomState seed"

    if source is not None:
        squared = np.square(source - reconstruction)
        serialized_scale_relative_mse = float(
            squared.sum() / np.square(source).sum()
        )
        if trial_row["source"]["kind"] == "frozen_bf16_weight_block":
            encoder_block_rms = float(trial_row["source"]["block_rms_fp64"])
        else:
            # Synthetic trials are generated directly at sigma_source and are
            # not RMS-normalized by the encoder.
            encoder_block_rms = sigma_source
        normalized_source = source * (sigma_source / encoder_block_rms)
        normalized_reconstruction = alphabet[previous].copy()
        if escape_positions.size:
            normalized_reconstruction[escape_positions] = normalized_source[escape_positions]
        normalized_squared = np.square(
            normalized_source - normalized_reconstruction
        )
        normalized_relative_mse = float(
            normalized_squared.sum() / np.square(normalized_source).sum()
        )
        expected_mse = float(trial_row["relative_mse"])
        result["decoded_relative_mse_with_serialized_scale"] = (
            serialized_scale_relative_mse
        )
        result["decoded_absolute_mse_with_serialized_scale"] = float(
            squared.mean(dtype=np.float64)
        )
        result["decoded_normalized_relative_mse_before_scale_serialization"] = (
            normalized_relative_mse
        )
        result["encoder_metadata_relative_mse"] = expected_mse
        result["normalized_mse_abs_difference"] = abs(
            normalized_relative_mse - expected_mse
        )
        result["serialized_scale_mse_delta_from_encoder_metric"] = (
            serialized_scale_relative_mse - expected_mse
        )
        ledger_fp16_scale = float(np.float16(encoder_block_rms / sigma_source))
        fp16_reconstruction = alphabet[previous] * ledger_fp16_scale
        if escape_positions.size:
            fp16_reconstruction[escape_positions] = source[escape_positions]
        fp16_squared = np.square(source - fp16_reconstruction)
        fp16_relative_mse = float(fp16_squared.sum() / np.square(source).sum())
        result["ledger_fp16_scale"] = ledger_fp16_scale
        result["decoded_relative_mse_with_ledger_fp16_scale"] = fp16_relative_mse
        result["decoded_absolute_mse_with_ledger_fp16_scale"] = float(
            fp16_squared.mean(dtype=np.float64)
        )
        result["fp16_scale_mse_delta_from_encoder_metric"] = (
            fp16_relative_mse - expected_mse
        )
        comparison_mse = (
            serialized_scale_relative_mse
            if args.container_layout == "plte-u20-tail-fp32"
            else normalized_relative_mse
        )
        result["decoded_reconstruction_matches_encoder_metric_at_1e_12"] = (
            abs(comparison_mse - expected_mse) <= 1e-12
        )
        result["decoded_indices_match_encoder_metric_at_1e_12"] = result[
            "decoded_reconstruction_matches_encoder_metric_at_1e_12"
        ]
        if not result["decoded_reconstruction_matches_encoder_metric_at_1e_12"]:
            raise AssertionError(
                "clean decoder reconstruction does not reproduce the encoder metric"
            )

    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
