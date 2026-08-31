#!/usr/bin/env python3
"""Build a VORPAL-preserving joint procedural SPARC residual wrapper.

The strict-PTQ encoder first applies 100 high-efficiency coordinate pulses per
role.  It then performs sequential matching pursuit against four deterministic
signed Walsh-Hadamard bases per 131072-value group.  Each stage emits one
20-bit (basis, coefficient, sign) symbol per group and one shared FP16
amplitude.  The emitted wrapper contains the original VORPAL bundle byte for
byte plus the complete residual extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import cupy as cp
import numpy as np


ROLES = (
    "mlp.experts.{expert}.up_proj.weight",
    "mlp.experts.{expert}.down_proj.weight",
    "mlp.experts.{expert}.gate_proj.weight",
)
PANEL_BLOCKS = 400
BLOCK_VALUES = 262144
PANEL_VALUES = PANEL_BLOCKS * BLOCK_VALUES
ROLE_BLOCKS = 48
ROLE_VALUES = ROLE_BLOCKS * BLOCK_VALUES
TRANSFORM_VALUES = 131072
LOG_TRANSFORM_VALUES = 17
GROUPS_PER_ROLE = ROLE_VALUES // TRANSFORM_VALUES
PROCEDURAL_BANKS = 4
BANK_BITS = 2
STAGE_SYMBOL_BITS = LOG_TRANSFORM_VALUES + 1 + BANK_BITS
STAGE_CODE_BYTES = GROUPS_PER_ROLE * STAGE_SYMBOL_BITS // 8
STAGE_RECORD_BYTES = 2 + STAGE_CODE_BYTES
COORDINATE_PULSES = 100
MAX_STAGES = 450
STRICT_MAX_BYTES = 32767999
SEED_ROLE_STRIDE = 1_000_000

EXT_MAGIC = b"VJSPRC41"
EXT_HEADER = struct.Struct("<8sBBBBIII")
ROLE_DESCRIPTOR = struct.Struct("<BBHHHeIII")
WRAPPER_MAGIC = b"VJWRAP42"
# The physical wrapper binds the base bundle, residual extension, selected
# manifest, exact-source evaluation, reconstruction, and encoder implementation.
WRAPPER_HEADER = struct.Struct("<8sIIIII32s32s32s32s32s32sI")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_payload(payload: bytes) -> np.ndarray:
    if len(payload) != BLOCK_VALUES * 2:
        raise ValueError("source block has wrong BF16 length")
    words = np.frombuffer(payload, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


class BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.accumulator = 0
        self.pending = 0
        self.meaningful = 0

    def write(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or (width and value >= 1 << width):
            raise ValueError((value, width))
        self.accumulator = (self.accumulator << width) | value
        self.pending += width
        self.meaningful += width
        while self.pending >= 8:
            shift = self.pending - 8
            self.data.append((self.accumulator >> shift) & 255)
            self.accumulator &= (1 << shift) - 1 if shift else 0
            self.pending = shift

    def finish(self) -> tuple[bytes, int]:
        if self.pending:
            self.data.append((self.accumulator << (8 - self.pending)) & 255)
            self.accumulator = 0
            self.pending = 0
        return bytes(self.data), self.meaningful


class BitReader:
    def __init__(self, payload: bytes, meaningful: int) -> None:
        self.payload = payload
        self.meaningful = meaningful
        self.offset = 0

    def read(self, width: int) -> int:
        if self.offset + width > self.meaningful:
            raise ValueError("bitstream truncated")
        value = 0
        for _ in range(width):
            byte = self.payload[self.offset >> 3]
            value = (value << 1) | ((byte >> (7 - (self.offset & 7))) & 1)
            self.offset += 1
        return value


def rice_bit_count(positions: np.ndarray, rice_b: int) -> int:
    gaps = np.diff(np.concatenate((np.asarray([-1]), positions))) - 1
    return int(np.sum(gaps >> rice_b, dtype=np.int64)) + len(positions) * (
        rice_b + 2
    )


def encode_rice(
    positions: np.ndarray, signs: np.ndarray, rice_b: int
) -> tuple[bytes, int]:
    writer = BitWriter()
    previous = -1
    mask = (1 << rice_b) - 1
    for position, sign in zip(positions, signs, strict=True):
        gap = int(position) - previous - 1
        quotient = gap >> rice_b
        for _ in range(quotient):
            writer.write(0, 1)
        writer.write(1, 1)
        writer.write(gap & mask, rice_b)
        writer.write(1 if int(sign) > 0 else 0, 1)
        previous = int(position)
    return writer.finish()


def decode_rice(
    payload: bytes, bit_count: int, count: int, rice_b: int
) -> tuple[np.ndarray, np.ndarray]:
    reader = BitReader(payload, bit_count)
    positions = np.empty(count, dtype=np.int64)
    signs = np.empty(count, dtype=np.int8)
    previous = -1
    for index in range(count):
        quotient = 0
        while reader.read(1) == 0:
            quotient += 1
        gap = (quotient << rice_b) | reader.read(rice_b)
        position = previous + gap + 1
        if position <= previous or position >= ROLE_VALUES:
            raise ValueError("Rice position outside role")
        positions[index] = position
        signs[index] = 1 if reader.read(1) else -1
        previous = position
    if reader.offset != bit_count:
        raise ValueError("unused Rice bits")
    return positions, signs


def fwht_gpu(values: cp.ndarray) -> cp.ndarray:
    transformed = values.copy()
    width = 1
    while width < TRANSFORM_VALUES:
        view = transformed.reshape(transformed.shape[0], -1, 2, width)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        transformed = view.reshape(transformed.shape)
        width *= 2
    return transformed / math.sqrt(TRANSFORM_VALUES)


def mix_signs(linear: cp.ndarray, seed: int | cp.ndarray) -> cp.ndarray:
    if isinstance(seed, cp.ndarray):
        offset = (seed[:, None] + cp.uint32(1)) * cp.uint32(0x9E3779B9)
    else:
        offset = cp.uint32(((seed + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    value = (linear + offset).astype(cp.uint32)
    value ^= value >> cp.uint32(16)
    value *= cp.uint32(0x7FEB352D)
    value ^= value >> cp.uint32(15)
    value *= cp.uint32(0x846CA68B)
    value ^= value >> cp.uint32(16)
    return 1.0 - 2.0 * (value & cp.uint32(1)).astype(cp.float64)


def hadamard_row_signs(indices: cp.ndarray, coordinates: cp.ndarray) -> cp.ndarray:
    value = indices[:, None].astype(cp.uint32) & coordinates[None, :]
    value ^= value >> cp.uint32(16)
    value ^= value >> cp.uint32(8)
    value ^= value >> cp.uint32(4)
    parity = (cp.uint32(0x6996) >> (value & cp.uint32(15))) & cp.uint32(1)
    return 1.0 - 2.0 * parity.astype(cp.float64)


def encode_stage_codes(
    banks: np.ndarray, indices: np.ndarray, positive: np.ndarray
) -> bytes:
    writer = BitWriter()
    for bank, index, sign in zip(banks, indices, positive, strict=True):
        code = (int(bank) << (LOG_TRANSFORM_VALUES + 1)) | (
            int(index) << 1
        ) | int(bool(sign))
        writer.write(code, STAGE_SYMBOL_BITS)
    payload, meaningful = writer.finish()
    if meaningful != GROUPS_PER_ROLE * STAGE_SYMBOL_BITS or len(payload) != STAGE_CODE_BYTES:
        raise AssertionError("stage code geometry mismatch")
    return payload


def decode_stage_codes(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(payload) != STAGE_CODE_BYTES:
        raise ValueError("stage code payload length mismatch")
    reader = BitReader(payload, GROUPS_PER_ROLE * STAGE_SYMBOL_BITS)
    banks = np.empty(GROUPS_PER_ROLE, dtype=np.uint8)
    indices = np.empty(GROUPS_PER_ROLE, dtype=np.uint32)
    positive = np.empty(GROUPS_PER_ROLE, dtype=bool)
    for group in range(GROUPS_PER_ROLE):
        code = reader.read(STAGE_SYMBOL_BITS)
        positive[group] = bool(code & 1)
        indices[group] = (code >> 1) & ((1 << LOG_TRANSFORM_VALUES) - 1)
        banks[group] = code >> (LOG_TRANSFORM_VALUES + 1)
    if reader.offset != reader.meaningful or np.any(banks >= PROCEDURAL_BANKS):
        raise ValueError("invalid stage codes")
    return banks, indices, positive


def role_mask(ordinals: list[int]) -> bytes:
    mask = np.zeros(PANEL_BLOCKS, dtype=np.uint8)
    mask[np.asarray(ordinals)] = 1
    payload = np.packbits(mask, bitorder="little").tobytes()
    if len(payload) != 50:
        raise AssertionError("role mask length")
    return payload


def parse_role_mask(payload: bytes) -> list[int]:
    if len(payload) != 50:
        raise ValueError("role mask length")
    mask = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    ordinals = np.flatnonzero(mask[:PANEL_BLOCKS]).tolist()
    if len(ordinals) != ROLE_BLOCKS or np.any(mask[PANEL_BLOCKS:]):
        raise ValueError("role mask census/padding")
    return ordinals


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = Path(str(path) + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_wrapper(
    base: bytes,
    extension: bytes,
    manifest_sha256: str,
    evaluation_sha256: str,
    reconstruction_sha256: str,
    encoder_sha256: str,
) -> bytes:
    base_hash = bytes.fromhex(sha256_bytes(base))
    extension_hash = bytes.fromhex(sha256_bytes(extension))
    zero = WRAPPER_HEADER.pack(
        WRAPPER_MAGIC,
        2,
        WRAPPER_HEADER.size,
        PANEL_VALUES,
        len(base),
        len(extension),
        base_hash,
        extension_hash,
        bytes.fromhex(manifest_sha256),
        bytes.fromhex(evaluation_sha256),
        bytes.fromhex(reconstruction_sha256),
        bytes.fromhex(encoder_sha256),
        0,
    )
    crc = zlib.crc32(zero[:-4]) & 0xFFFFFFFF
    header = zero[:-4] + struct.pack("<I", crc)
    return header + base + extension


def parse_wrapper(payload: bytes) -> tuple[bytes, bytes, dict[str, str]]:
    if len(payload) < WRAPPER_HEADER.size:
        raise ValueError("wrapper truncated")
    fields = WRAPPER_HEADER.unpack(payload[:WRAPPER_HEADER.size])
    (
        magic,
        version,
        header_bytes,
        panel_values,
        base_bytes,
        ext_bytes,
        base_hash,
        ext_hash,
        manifest_hash,
        evaluation_hash,
        reconstruction_hash,
        encoder_hash,
        crc,
    ) = fields
    if (
        magic != WRAPPER_MAGIC
        or version != 2
        or header_bytes != WRAPPER_HEADER.size
        or panel_values != PANEL_VALUES
        or zlib.crc32(payload[:WRAPPER_HEADER.size - 4]) & 0xFFFFFFFF != crc
    ):
        raise ValueError("wrapper header invalid")
    if len(payload) != WRAPPER_HEADER.size + base_bytes + ext_bytes:
        raise ValueError("wrapper lengths do not sum")
    base = payload[WRAPPER_HEADER.size:WRAPPER_HEADER.size + base_bytes]
    extension = payload[WRAPPER_HEADER.size + base_bytes:]
    if sha256_bytes(base) != base_hash.hex() or sha256_bytes(extension) != ext_hash.hex():
        raise ValueError("wrapper payload hash mismatch")
    return base, extension, {
        "manifest_sha256": manifest_hash.hex(),
        "evaluation_sha256": evaluation_hash.hex(),
        "reconstruction_sha256": reconstruction_hash.hex(),
        "encoder_sha256": encoder_hash.hex(),
    }


def gap_db(sse: float, energy: float, rate: float) -> float:
    return 10.0 * math.log10((sse / energy) / (2.0 ** (-2.0 * rate)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-stages", type=int, default=MAX_STAGES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_stages != MAX_STAGES:
        raise ValueError(f"normative experiment requires {MAX_STAGES} stages")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if len(manifest["blocks"]) != PANEL_BLOCKS:
        raise ValueError("manifest is not the 400-block panel")
    base_payload = args.base_bundle.read_bytes()
    if (
        len(base_payload) != int(evaluation["encoded_bytes"])
        or sha256_bytes(base_payload) != evaluation["encoded_sha256"]
    ):
        raise ValueError("base bundle does not match evaluation")
    if sha256_path(args.reconstruction) != evaluation["reconstruction_sha256"]:
        raise ValueError("reconstruction does not match evaluation")
    reconstruction = np.memmap(
        args.reconstruction,
        dtype="<f8",
        mode="r",
        shape=(PANEL_BLOCKS, BLOCK_VALUES),
    )

    role_results: list[dict[str, object]] = []
    for role_id, role in enumerate(ROLES):
        started = time.time()
        ordinals = [
            ordinal
            for ordinal, block in enumerate(manifest["blocks"])
            if block["role"] == role
        ]
        if len(ordinals) != ROLE_BLOCKS:
            raise ValueError(f"wrong block census for {role}")
        residual = cp.empty((GROUPS_PER_ROLE, TRANSFORM_VALUES), dtype=cp.float64)
        energy = 0.0
        source_rows = []
        for local_block, ordinal in enumerate(ordinals):
            block = manifest["blocks"][ordinal]
            source_path = Path(block["source_path"])
            if not source_path.is_absolute():
                source_path = args.source_root / source_path
            source_bytes = source_path.read_bytes()
            source_hash = sha256_bytes(source_bytes)
            if source_hash != block["source_sha256"]:
                raise ValueError(f"source hash mismatch at ordinal {ordinal}")
            source_gpu = cp.asarray(bf16_payload(source_bytes), dtype=cp.float64)
            reconstruction_gpu = cp.asarray(
                np.asarray(reconstruction[ordinal]), dtype=cp.float64
            )
            residual[local_block * 2:(local_block + 1) * 2] = (
                source_gpu - reconstruction_gpu
            ).reshape(2, TRANSFORM_VALUES)
            energy += float(cp.sum(source_gpu * source_gpu, dtype=cp.float64).get())
            source_rows.append(
                {
                    "ordinal": ordinal,
                    "layer": int(block.get("layer", local_block)),
                    "id": block["id"],
                    "sha256": source_hash,
                }
            )
        base_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
        published = next(row for row in evaluation["by_role"] if row["role"] == role)
        if not math.isclose(energy, published["source_energy"], rel_tol=3e-15, abs_tol=3e-12):
            raise ValueError("role energy mismatch")
        if not math.isclose(base_sse, published["sse"], rel_tol=3e-15, abs_tol=3e-12):
            raise ValueError("role SSE mismatch")

        flat = residual.ravel()
        absolute = cp.abs(flat)
        coordinate_indices_gpu = cp.argpartition(absolute, -COORDINATE_PULSES)[
            -COORDINATE_PULSES:
        ]
        coordinate_order_gpu = cp.argsort(-absolute[coordinate_indices_gpu])
        coordinate_indices_gpu = coordinate_indices_gpu[coordinate_order_gpu]
        coordinate_amplitude = np.float16(
            float(cp.mean(absolute[coordinate_indices_gpu], dtype=cp.float64).get())
        )
        coordinate_signs_gpu = cp.sign(flat[coordinate_indices_gpu])
        flat[coordinate_indices_gpu] -= coordinate_signs_gpu * float(
            coordinate_amplitude
        )
        post_coordinate_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
        coordinate_gain = base_sse - post_coordinate_sse
        coordinate_indices = cp.asnumpy(coordinate_indices_gpu).astype(np.int64)
        coordinate_signs = cp.asnumpy(coordinate_signs_gpu).astype(np.int8)
        coordinate_sort = np.argsort(coordinate_indices, kind="stable")
        coordinate_positions = coordinate_indices[coordinate_sort]
        coordinate_signs = coordinate_signs[coordinate_sort]
        rice_b = min(
            range(21),
            key=lambda value: rice_bit_count(coordinate_positions, value),
        )
        coordinate_payload, coordinate_bits = encode_rice(
            coordinate_positions, coordinate_signs, rice_b
        )
        decoded_positions, decoded_signs = decode_rice(
            coordinate_payload, coordinate_bits, COORDINATE_PULSES, rice_b
        )
        if not np.array_equal(decoded_positions, coordinate_positions) or not np.array_equal(
            decoded_signs, coordinate_signs
        ):
            raise AssertionError("coordinate pulse round trip failed")

        linear = cp.arange(ROLE_VALUES, dtype=cp.uint32).reshape(
            GROUPS_PER_ROLE, TRANSFORM_VALUES
        )
        coordinates = cp.arange(TRANSFORM_VALUES, dtype=cp.uint32)
        stage_amplitudes: list[np.float16] = []
        stage_gains: list[float] = []
        stage_codes: list[bytes] = []
        for stage in range(MAX_STAGES):
            best_abs = cp.zeros(GROUPS_PER_ROLE, dtype=cp.float64)
            best_coefficient = cp.zeros(GROUPS_PER_ROLE, dtype=cp.float64)
            best_index = cp.zeros(GROUPS_PER_ROLE, dtype=cp.uint32)
            best_bank = cp.zeros(GROUPS_PER_ROLE, dtype=cp.uint32)
            for bank in range(PROCEDURAL_BANKS):
                seed = role_id * SEED_ROLE_STRIDE + stage * PROCEDURAL_BANKS + bank
                diagonal = mix_signs(linear, seed)
                transformed = fwht_gpu(residual * diagonal)
                index = cp.argmax(cp.abs(transformed), axis=1).astype(cp.uint32)
                coefficient = transformed[cp.arange(GROUPS_PER_ROLE), index]
                magnitude = cp.abs(coefficient)
                replace = magnitude > best_abs
                best_abs = cp.where(replace, magnitude, best_abs)
                best_coefficient = cp.where(replace, coefficient, best_coefficient)
                best_index = cp.where(replace, index, best_index)
                best_bank = cp.where(replace, bank, best_bank)
            amplitude = np.float16(
                float(cp.mean(best_abs, dtype=cp.float64).get())
            )
            decoded_amplitude = float(amplitude)
            gain = float(
                cp.sum(
                    2.0 * decoded_amplitude * best_abs
                    - decoded_amplitude * decoded_amplitude,
                    dtype=cp.float64,
                ).get()
            )
            selected_seed = (
                cp.uint32(role_id * SEED_ROLE_STRIDE + stage * PROCEDURAL_BANKS)
                + best_bank
            )
            diagonal = mix_signs(linear, selected_seed)
            atom = hadamard_row_signs(best_index, coordinates) * diagonal
            residual -= (
                cp.sign(best_coefficient)[:, None]
                * (decoded_amplitude / math.sqrt(TRANSFORM_VALUES))
                * atom
            )
            banks_cpu = cp.asnumpy(best_bank).astype(np.uint8)
            indices_cpu = cp.asnumpy(best_index).astype(np.uint32)
            positive_cpu = cp.asnumpy(best_coefficient > 0.0)
            stage_amplitudes.append(amplitude)
            stage_gains.append(gain)
            stage_codes.append(
                encode_stage_codes(banks_cpu, indices_cpu, positive_cpu)
            )
            if (stage + 1) % 100 == 0:
                print(
                    f"{role}: {stage + 1}/{MAX_STAGES} stages, "
                    f"prefix gain {coordinate_gain + math.fsum(stage_gains):.12f}, "
                    f"elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
        analytic_sse = base_sse - coordinate_gain - math.fsum(stage_gains)
        direct_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
        if not math.isclose(analytic_sse, direct_sse, rel_tol=5e-14, abs_tol=2e-11):
            raise ValueError("stage analytic/direct SSE mismatch")
        role_results.append(
            {
                "role_id": role_id,
                "role": role,
                "ordinals": ordinals,
                "role_mask": role_mask(ordinals),
                "source_rows": source_rows,
                "energy": energy,
                "base_sse": base_sse,
                "coordinate_amplitude": coordinate_amplitude,
                "coordinate_gain": coordinate_gain,
                "coordinate_positions": coordinate_positions,
                "coordinate_signs": coordinate_signs,
                "rice_b": rice_b,
                "coordinate_payload": coordinate_payload,
                "coordinate_bits": coordinate_bits,
                "stage_amplitudes": stage_amplitudes,
                "stage_gains": np.asarray(stage_gains, dtype=np.float64),
                "stage_codes": stage_codes,
            }
        )
        del residual, flat, absolute, linear, coordinates
        cp.get_default_memory_pool().free_all_blocks()

    del reconstruction
    fixed_extension_bytes = EXT_HEADER.size + 4
    for result in role_results:
        fixed_extension_bytes += (
            ROLE_DESCRIPTOR.size
            + 50
            + len(result["coordinate_payload"])
        )
    fixed_wrapper_and_extension = WRAPPER_HEADER.size + fixed_extension_bytes
    available_stage_bytes = STRICT_MAX_BYTES - len(base_payload) - fixed_wrapper_and_extension
    maximum_total_stages = available_stage_bytes // STAGE_RECORD_BYTES
    if maximum_total_stages > len(ROLES) * MAX_STAGES:
        maximum_total_stages = len(ROLES) * MAX_STAGES
    total_candidate_bytes = (
        len(base_payload)
        + fixed_wrapper_and_extension
        + maximum_total_stages * STAGE_RECORD_BYTES
    )
    rate = total_candidate_bytes * 8.0 / PANEL_VALUES
    reference = 2.0 ** (-2.0 * rate)
    cumulative = [
        np.concatenate(
            (
                np.asarray([result["coordinate_gain"]], dtype=np.float64),
                result["coordinate_gain"]
                + np.cumsum(result["stage_gains"], dtype=np.float64),
            )
        )
        for result in role_results
    ]
    best = None
    total = maximum_total_stages
    for up_count in range(max(0, total - 2 * MAX_STAGES), min(MAX_STAGES, total) + 1):
        down_min = max(0, total - up_count - MAX_STAGES)
        down_max = min(MAX_STAGES, total - up_count)
        if down_min > down_max:
            continue
        down_counts = np.arange(down_min, down_max + 1, dtype=np.int64)
        gate_counts = total - up_count - down_counts
        up_sse = float(role_results[0]["base_sse"]) - cumulative[0][up_count]
        up_gap = 10.0 * math.log10(
            (up_sse / float(role_results[0]["energy"])) / reference
        )
        down_sse = float(role_results[1]["base_sse"]) - cumulative[1][down_counts]
        gate_sse = float(role_results[2]["base_sse"]) - cumulative[2][gate_counts]
        down_gaps = 10.0 * np.log10(
            (down_sse / float(role_results[1]["energy"])) / reference
        )
        gate_gaps = 10.0 * np.log10(
            (gate_sse / float(role_results[2]["energy"])) / reference
        )
        worst = np.maximum(np.maximum(down_gaps, gate_gaps), up_gap)
        index = int(np.argmin(worst))
        candidate = {
            "worst_gap": float(worst[index]),
            "gaps": [
                up_gap,
                float(down_gaps[index]),
                float(gate_gaps[index]),
            ],
            "counts": [
                up_count,
                int(down_counts[index]),
                int(gate_counts[index]),
            ],
        }
        if best is None or (
            candidate["worst_gap"], candidate["counts"]
        ) < (best["worst_gap"], best["counts"]):
            best = candidate
    if best is None:
        raise RuntimeError("stage allocation failed")

    extension = bytearray(
        EXT_HEADER.pack(
            EXT_MAGIC,
            1,
            len(ROLES),
            LOG_TRANSFORM_VALUES,
            PROCEDURAL_BANKS,
            PANEL_BLOCKS,
            BLOCK_VALUES,
            GROUPS_PER_ROLE,
        )
    )
    selected_rows = []
    for result, stage_count, predicted_gap in zip(
        role_results, best["counts"], best["gaps"], strict=True
    ):
        descriptor = ROLE_DESCRIPTOR.pack(
            int(result["role_id"]),
            int(result["rice_b"]),
            ROLE_BLOCKS,
            COORDINATE_PULSES,
            int(stage_count),
            float(result["coordinate_amplitude"]),
            int(result["coordinate_bits"]),
            len(result["coordinate_payload"]),
            STAGE_RECORD_BYTES,
        )
        extension.extend(descriptor)
        extension.extend(result["role_mask"])
        extension.extend(result["coordinate_payload"])
        for stage in range(stage_count):
            extension.extend(struct.pack("<e", float(result["stage_amplitudes"][stage])))
            extension.extend(result["stage_codes"][stage])
        savings = float(cumulative[int(result["role_id"])][stage_count])
        corrected_sse = float(result["base_sse"]) - savings
        selected_rows.append(
            {
                "role": result["role"],
                "role_id": result["role_id"],
                "block_ordinals": result["ordinals"],
                "source_energy": result["energy"],
                "base_sse": result["base_sse"],
                "coordinate_pulses": COORDINATE_PULSES,
                "coordinate_amplitude_fp16": float(result["coordinate_amplitude"]),
                "coordinate_rice_b": result["rice_b"],
                "coordinate_bits": result["coordinate_bits"],
                "coordinate_bytes": len(result["coordinate_payload"]),
                "coordinate_gain": result["coordinate_gain"],
                "sparc_stages": stage_count,
                "stage_record_bytes": STAGE_RECORD_BYTES,
                "stage_amplitudes_fp16": [
                    float(value) for value in result["stage_amplitudes"][:stage_count]
                ],
                "sse_savings": savings,
                "corrected_sse": corrected_sse,
                "relative_mse": corrected_sse / float(result["energy"]),
                "gap_db": predicted_gap,
                "sources": result["source_rows"],
            }
        )
    extension_crc = zlib.crc32(extension) & 0xFFFFFFFF
    extension.extend(struct.pack("<I", extension_crc))
    extension_payload = bytes(extension)
    extension_path = args.output_dir / "joint_sparc4.extension.bin"
    atomic_write(extension_path, extension_payload)
    manifest_hash = sha256_path(args.manifest)
    evaluation_hash = sha256_path(args.evaluation)
    reconstruction_hash = sha256_path(args.reconstruction)
    encoder_hash = sha256_path(Path(__file__).resolve())
    wrapper = build_wrapper(
        base_payload,
        extension_payload,
        manifest_hash,
        evaluation_hash,
        reconstruction_hash,
        encoder_hash,
    )
    wrapper_path = args.output_dir / "vorpal_joint_sparc4.vjwrap"
    atomic_write(wrapper_path, wrapper)
    decoded_base, decoded_extension, decoded_bindings = parse_wrapper(
        wrapper_path.read_bytes()
    )
    if decoded_base != base_payload or decoded_extension != extension_payload:
        raise AssertionError("wrapper round trip failed")
    if decoded_bindings != {
        "manifest_sha256": manifest_hash,
        "evaluation_sha256": evaluation_hash,
        "reconstruction_sha256": reconstruction_hash,
        "encoder_sha256": encoder_hash,
    }:
        raise AssertionError("wrapper provenance binding round trip failed")
    if len(wrapper) != total_candidate_bytes:
        raise AssertionError((len(wrapper), total_candidate_bytes))
    if len(wrapper) > STRICT_MAX_BYTES:
        raise AssertionError("strict physical rate limit exceeded")

    corrected_global_sse = float(evaluation["sse"]) - math.fsum(
        row["sse_savings"] for row in selected_rows
    )
    corrected_global_relative_mse = corrected_global_sse / float(
        evaluation["source_energy"]
    )
    corrected_global_gap = 10.0 * math.log10(
        corrected_global_relative_mse / reference
    )
    receipt = {
        "format": "VORPAL-preserving procedural multi-basis SPARC wrapper v2",
        "status": "passed" if best["worst_gap"] < 0.0 and corrected_global_gap < 0.0 else "failed",
        "strict_ptq": True,
        "training_or_retraining": False,
        "calibration_or_activations": False,
        "base_preserved_byte_for_byte": True,
        "method": {
            "coordinate_pulses_per_role": COORDINATE_PULSES,
            "transform": "normalized Walsh-Hadamard",
            "transform_values": TRANSFORM_VALUES,
            "groups_per_role": GROUPS_PER_ROLE,
            "procedural_signed_bases_per_stage": PROCEDURAL_BANKS,
            "stage_symbol_bits_per_group": STAGE_SYMBOL_BITS,
            "stage_code_bytes": STAGE_CODE_BYTES,
            "stage_amplitude": "one IEEE binary16 value shared across the role/stage",
            "stage_record_bytes": STAGE_RECORD_BYTES,
            "procedural_hash": "uint32 mix 0x7FEB352D/0x846CA68B",
            "seed_role_stride": SEED_ROLE_STRIDE,
        },
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash,
            "evaluation": str(args.evaluation),
            "evaluation_sha256": evaluation_hash,
            "reconstruction": str(args.reconstruction),
            "reconstruction_sha256": reconstruction_hash,
            "base_bundle": str(args.base_bundle),
            "base_bundle_bytes": len(base_payload),
            "base_bundle_sha256": sha256_bytes(base_payload),
        },
        "artifacts": {
            "extension": str(extension_path),
            "extension_bytes": len(extension_payload),
            "extension_sha256": sha256_bytes(extension_payload),
            "emitted_wrapper": str(wrapper_path),
            "emitted_wrapper_bytes": len(wrapper),
            "emitted_wrapper_sha256": sha256_bytes(wrapper),
            "wrapper_header_bytes": WRAPPER_HEADER.size,
            "wrapper_roundtrip_verified": True,
            "wrapper_binds_base_extension_manifest_evaluation_reconstruction_and_encoder": True,
        },
        "accounting": {
            "strict_max_bytes_below_2p5": STRICT_MAX_BYTES,
            "base_bytes": len(base_payload),
            "wrapper_header_bytes": WRAPPER_HEADER.size,
            "extension_bytes": len(extension_payload),
            "physical_all_in_bytes": len(wrapper),
            "physical_all_in_rate_bpw": len(wrapper) * 8.0 / PANEL_VALUES,
            "rate_headroom_bytes": STRICT_MAX_BYTES - len(wrapper),
            "gaussian_reference_at_actual_rate": reference,
            "total_selected_sparc_stages": maximum_total_stages,
            "fixed_wrapper_and_extension_bytes_before_stages": fixed_wrapper_and_extension,
        },
        "roles": selected_rows,
        "worst_role_gap_db": best["worst_gap"],
        "all_three_roles_below_zero": best["worst_gap"] < 0.0,
        "global": {
            "base_sse": evaluation["sse"],
            "corrected_sse": corrected_global_sse,
            "source_energy": evaluation["source_energy"],
            "relative_mse": corrected_global_relative_mse,
            "gap_db": corrected_global_gap,
            "remains_below_zero": corrected_global_gap < 0.0,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": encoder_hash,
        },
    }
    receipt_path = args.output_dir / "joint_sparc4.receipt.json"
    atomic_write(
        receipt_path,
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "wrapper": str(wrapper_path),
                "wrapper_bytes": len(wrapper),
                "wrapper_sha256": sha256_bytes(wrapper),
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_path(receipt_path),
                "rate_bpw": receipt["accounting"]["physical_all_in_rate_bpw"],
                "role_gaps_db": {row["role"]: row["gap_db"] for row in selected_rows},
                "worst_role_gap_db": best["worst_gap"],
                "global_gap_db": corrected_global_gap,
                "selected_stage_counts": best["counts"],
                "headroom_bytes": receipt["accounting"]["rate_headroom_bytes"],
            },
            indent=2,
        )
    )
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
