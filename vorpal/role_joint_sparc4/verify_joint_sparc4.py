#!/usr/bin/env python3
"""Independent decoder and exact-source verifier for VJWRAP42/VJSPRC41.

This module does not import the encoder.  It structurally parses the emitted
wrapper and applies every transmitted correction before opening source weights
for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
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
GROUPS_PER_ROLE = 96
PROCEDURAL_BANKS = 4
STAGE_SYMBOL_BITS = 20
STAGE_CODE_BYTES = 240
STAGE_RECORD_BYTES = 242
SEED_ROLE_STRIDE = 1_000_000
EXT_HEADER = struct.Struct("<8sBBBBIII")
ROLE_DESCRIPTOR = struct.Struct("<BBHHHeIII")
WRAPPER_HEADER = struct.Struct("<8sIIIII32s32s32s32s32s32sI")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BitReader:
    def __init__(self, payload: bytes, meaningful: int) -> None:
        self.payload = payload
        self.meaningful = meaningful
        self.offset = 0

    def read(self, width: int) -> int:
        if self.offset + width > self.meaningful:
            raise ValueError("truncated bits")
        value = 0
        for _ in range(width):
            value = (value << 1) | (
                (self.payload[self.offset >> 3] >> (7 - (self.offset & 7))) & 1
            )
            self.offset += 1
        return value


def decode_rice(
    payload: bytes, meaningful: int, count: int, rice_b: int
) -> tuple[np.ndarray, np.ndarray]:
    reader = BitReader(payload, meaningful)
    positions = np.empty(count, dtype=np.int64)
    signs = np.empty(count, dtype=np.int8)
    previous = -1
    for item in range(count):
        quotient = 0
        while reader.read(1) == 0:
            quotient += 1
        gap = (quotient << rice_b) | reader.read(rice_b)
        position = previous + gap + 1
        if position <= previous or position >= ROLE_VALUES:
            raise ValueError("Rice support is noncanonical")
        positions[item] = position
        signs[item] = 1 if reader.read(1) else -1
        previous = position
    if reader.offset != meaningful:
        raise ValueError("unused meaningful Rice bits")
    if meaningful % 8:
        padding_mask = (1 << (8 - meaningful % 8)) - 1
        if payload[-1] & padding_mask:
            raise ValueError("nonzero Rice padding")
    return positions, signs


def rice_bit_count(positions: np.ndarray, rice_b: int) -> int:
    gaps = np.diff(np.concatenate((np.asarray([-1]), positions))) - 1
    return int(np.sum(gaps >> rice_b, dtype=np.int64)) + len(positions) * (
        rice_b + 2
    )


def decode_stage(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(payload) != STAGE_CODE_BYTES:
        raise ValueError("stage payload length")
    reader = BitReader(payload, GROUPS_PER_ROLE * STAGE_SYMBOL_BITS)
    banks = np.empty(GROUPS_PER_ROLE, dtype=np.uint32)
    indices = np.empty(GROUPS_PER_ROLE, dtype=np.uint32)
    positive = np.empty(GROUPS_PER_ROLE, dtype=bool)
    for group in range(GROUPS_PER_ROLE):
        code = reader.read(STAGE_SYMBOL_BITS)
        positive[group] = bool(code & 1)
        indices[group] = (code >> 1) & 0x1FFFF
        banks[group] = code >> 18
    if reader.offset != reader.meaningful or np.any(banks >= PROCEDURAL_BANKS):
        raise ValueError("stage symbol invalid")
    return banks, indices, positive


def parse_mask(payload: bytes) -> list[int]:
    if len(payload) != 50:
        raise ValueError("mask bytes")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if np.any(bits[PANEL_BLOCKS:]):
        raise ValueError("mask padding")
    ordinals = np.flatnonzero(bits[:PANEL_BLOCKS]).tolist()
    if len(ordinals) != ROLE_BLOCKS:
        raise ValueError("mask census")
    return ordinals


def parse_wrapper(payload: bytes) -> tuple[bytes, bytes, dict[str, object]]:
    if len(payload) < WRAPPER_HEADER.size:
        raise ValueError("truncated wrapper")
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
        magic != b"VJWRAP42"
        or version != 2
        or header_bytes != WRAPPER_HEADER.size
        or panel_values != PANEL_VALUES
    ):
        raise ValueError("wrapper constants")
    if zlib.crc32(payload[:WRAPPER_HEADER.size - 4]) & 0xFFFFFFFF != crc:
        raise ValueError("wrapper header CRC")
    if len(payload) != WRAPPER_HEADER.size + base_bytes + ext_bytes:
        raise ValueError("wrapper section sizes")
    base = payload[WRAPPER_HEADER.size:WRAPPER_HEADER.size + base_bytes]
    extension = payload[WRAPPER_HEADER.size + base_bytes:]
    if sha256_bytes(base) != base_hash.hex() or sha256_bytes(extension) != ext_hash.hex():
        raise ValueError("wrapper section hash")
    return base, extension, {
        "wrapper_header_crc_verified": True,
        "base_hash_verified": True,
        "extension_hash_verified": True,
        "wrapper_lengths_verified": True,
        "manifest_sha256": manifest_hash.hex(),
        "evaluation_sha256": evaluation_hash.hex(),
        "reconstruction_sha256": reconstruction_hash.hex(),
        "encoder_sha256": encoder_hash.hex(),
    }


def parse_extension(extension: bytes) -> list[dict[str, object]]:
    if len(extension) < EXT_HEADER.size + 4:
        raise ValueError("extension truncated")
    if zlib.crc32(extension[:-4]) & 0xFFFFFFFF != struct.unpack("<I", extension[-4:])[0]:
        raise ValueError("extension CRC")
    fields = EXT_HEADER.unpack(extension[:EXT_HEADER.size])
    magic, version, role_count, log_values, banks, panel_blocks, block_values, groups = fields
    if (
        magic != b"VJSPRC41"
        or version != 1
        or role_count != 3
        or log_values != 17
        or banks != 4
        or panel_blocks != PANEL_BLOCKS
        or block_values != BLOCK_VALUES
        or groups != GROUPS_PER_ROLE
    ):
        raise ValueError("extension constants")
    cursor = EXT_HEADER.size
    rows: list[dict[str, object]] = []
    masks: set[int] = set()
    for expected_role in range(3):
        if cursor + ROLE_DESCRIPTOR.size + 50 > len(extension) - 4:
            raise ValueError("role descriptor truncated")
        descriptor = ROLE_DESCRIPTOR.unpack(
            extension[cursor:cursor + ROLE_DESCRIPTOR.size]
        )
        cursor += ROLE_DESCRIPTOR.size
        role_id, rice_b, block_count, coordinate_count, stage_count, coordinate_amp, coordinate_bits, coordinate_bytes, stage_bytes = descriptor
        if (
            role_id != expected_role
            or rice_b > 20
            or block_count != ROLE_BLOCKS
            or coordinate_count != 100
            or stage_count > 450
            or coordinate_bytes != (coordinate_bits + 7) // 8
            or stage_bytes != STAGE_RECORD_BYTES
        ):
            raise ValueError("role descriptor noncanonical")
        ordinals = parse_mask(extension[cursor:cursor + 50])
        cursor += 50
        if masks.intersection(ordinals):
            raise ValueError("role masks overlap")
        masks.update(ordinals)
        coordinate_payload = extension[cursor:cursor + coordinate_bytes]
        cursor += coordinate_bytes
        positions, signs = decode_rice(
            coordinate_payload, coordinate_bits, coordinate_count, rice_b
        )
        if not math.isfinite(coordinate_amp) or coordinate_amp <= 0.0:
            raise ValueError("coordinate amplitude invalid")
        recomputed_rice_bits = rice_bit_count(positions, rice_b)
        optimal_rice_b = min(range(21), key=lambda value: rice_bit_count(positions, value))
        if coordinate_bits != recomputed_rice_bits or rice_b != optimal_rice_b:
            raise ValueError("Rice payload is not exact and minimal")
        stages = []
        for _ in range(stage_count):
            if cursor + STAGE_RECORD_BYTES > len(extension) - 4:
                raise ValueError("stage truncated")
            amplitude = struct.unpack("<e", extension[cursor:cursor + 2])[0]
            banks_decoded, indices, positive = decode_stage(
                extension[cursor + 2:cursor + STAGE_RECORD_BYTES]
            )
            cursor += STAGE_RECORD_BYTES
            if not math.isfinite(amplitude) or amplitude <= 0.0:
                raise ValueError("stage amplitude invalid")
            stages.append(
                {
                    "amplitude": float(amplitude),
                    "banks": banks_decoded,
                    "indices": indices,
                    "positive": positive,
                }
            )
        rows.append(
            {
                "role_id": role_id,
                "ordinals": ordinals,
                "coordinate_amplitude": float(coordinate_amp),
                "coordinate_positions": positions,
                "coordinate_signs": signs,
                "rice_b": rice_b,
                "coordinate_bits": coordinate_bits,
                "coordinate_bytes": coordinate_bytes,
                "stages": stages,
            }
        )
    if cursor != len(extension) - 4:
        raise ValueError("trailing extension bytes")
    return rows


def mix_signs(linear: cp.ndarray, seeds: cp.ndarray) -> cp.ndarray:
    value = (
        linear
        + (seeds[:, None] + cp.uint32(1)) * cp.uint32(0x9E3779B9)
    ).astype(cp.uint32)
    value ^= value >> cp.uint32(16)
    value *= cp.uint32(0x7FEB352D)
    value ^= value >> cp.uint32(15)
    value *= cp.uint32(0x846CA68B)
    value ^= value >> cp.uint32(16)
    return 1.0 - 2.0 * (value & cp.uint32(1)).astype(cp.float64)


def hadamard_signs(indices: cp.ndarray, coordinates: cp.ndarray) -> cp.ndarray:
    value = indices[:, None].astype(cp.uint32) & coordinates[None, :]
    value ^= value >> cp.uint32(16)
    value ^= value >> cp.uint32(8)
    value ^= value >> cp.uint32(4)
    parity = (cp.uint32(0x6996) >> (value & cp.uint32(15))) & cp.uint32(1)
    return 1.0 - 2.0 * parity.astype(cp.float64)


def bf16(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_hash or len(payload) != BLOCK_VALUES * 2:
        raise ValueError(f"source validation failed: {path}")
    words = np.frombuffer(payload, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-receipt", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".partial").exists():
        raise FileExistsError(args.output)

    wrapper_payload = args.wrapper.read_bytes()
    base, extension, wrapper_checks = parse_wrapper(wrapper_payload)
    if base != args.base_bundle.read_bytes():
        raise ValueError("embedded base is not the normative base byte-for-byte")
    external_bindings = {
        "manifest_sha256": sha256_path(args.manifest),
        "evaluation_sha256": sha256_path(args.evaluation),
        "reconstruction_sha256": sha256_path(args.reconstruction),
        "encoder_sha256": sha256_path(args.encoder),
    }
    if any(wrapper_checks[key] != value for key, value in external_bindings.items()):
        raise ValueError("wrapper external provenance binding mismatch")
    decoded_roles = parse_extension(extension)

    # Structural tamper tests are performed before any source is opened.
    tampered = bytearray(wrapper_payload)
    tampered[-17] ^= 1
    payload_tamper_rejected = False
    try:
        parse_wrapper(bytes(tampered))
    except ValueError:
        payload_tamper_rejected = True
    tampered_extension = bytearray(extension)
    tampered_extension[-5] ^= 1
    extension_crc_tamper_rejected = False
    try:
        parse_extension(bytes(tampered_extension))
    except ValueError:
        extension_crc_tamper_rejected = True
    provenance_fields = list(WRAPPER_HEADER.unpack(wrapper_payload[:WRAPPER_HEADER.size]))
    mutated_encoder_hash = bytearray(provenance_fields[11])
    mutated_encoder_hash[0] ^= 1
    provenance_fields[11] = bytes(mutated_encoder_hash)
    provenance_fields[-1] = 0
    mutated_header = WRAPPER_HEADER.pack(*provenance_fields)
    mutated_header = mutated_header[:-4] + struct.pack(
        "<I", zlib.crc32(mutated_header[:-4]) & 0xFFFFFFFF
    )
    _, _, mutated_checks = parse_wrapper(
        mutated_header + wrapper_payload[WRAPPER_HEADER.size:]
    )
    provenance_tamper_rejected = any(
        mutated_checks[key] != value for key, value in external_bindings.items()
    )
    if (
        not payload_tamper_rejected
        or not extension_crc_tamper_rejected
        or not provenance_tamper_rejected
    ):
        raise AssertionError("tamper tests did not fail closed")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    receipt = json.loads(args.experiment_receipt.read_text(encoding="utf-8"))
    if not (
        receipt.get("status") == "passed"
        and receipt.get("strict_ptq") is True
        and receipt.get("training_or_retraining") is False
        and receipt.get("calibration_or_activations") is False
    ):
        raise ValueError("experiment receipt status/PTQ assertions are invalid")
    if (
        int(receipt["artifacts"]["emitted_wrapper_bytes"]) != len(wrapper_payload)
        or receipt["artifacts"]["emitted_wrapper_sha256"] != sha256_bytes(wrapper_payload)
        or int(receipt["artifacts"]["extension_bytes"]) != len(extension)
        or receipt["artifacts"]["extension_sha256"] != sha256_bytes(extension)
        or int(receipt["artifacts"]["wrapper_header_bytes"]) != WRAPPER_HEADER.size
        or receipt["inputs"]["base_bundle_sha256"] != sha256_bytes(base)
        or any(receipt["inputs"][key] != value for key, value in external_bindings.items() if key != "encoder_sha256")
        or receipt["environment"]["script_sha256"] != external_bindings["encoder_sha256"]
    ):
        raise ValueError("receipt artifact/provenance binding mismatch")
    expected_method = {
        "coordinate_pulses_per_role": 100,
        "transform": "normalized Walsh-Hadamard",
        "transform_values": TRANSFORM_VALUES,
        "groups_per_role": GROUPS_PER_ROLE,
        "procedural_signed_bases_per_stage": PROCEDURAL_BANKS,
        "stage_symbol_bits_per_group": STAGE_SYMBOL_BITS,
        "stage_code_bytes": STAGE_CODE_BYTES,
        "stage_record_bytes": STAGE_RECORD_BYTES,
        "seed_role_stride": SEED_ROLE_STRIDE,
    }
    if any(receipt["method"].get(key) != value for key, value in expected_method.items()):
        raise ValueError("receipt method constants disagree with decoder")
    for decoded, claimed in zip(decoded_roles, receipt["roles"], strict=True):
        if (
            int(decoded["role_id"]) != int(claimed["role_id"])
            or len(decoded["coordinate_positions"]) != int(claimed["coordinate_pulses"])
            or len(decoded["stages"]) != int(claimed["sparc_stages"])
            or int(decoded["rice_b"]) != int(claimed["coordinate_rice_b"])
            or int(decoded["coordinate_bits"]) != int(claimed["coordinate_bits"])
            or int(decoded["coordinate_bytes"]) != int(claimed["coordinate_bytes"])
            or float(decoded["coordinate_amplitude"])
            != float(claimed["coordinate_amplitude_fp16"])
        ):
            raise ValueError("receipt role fields disagree with decoded extension")
    if sha256_path(args.reconstruction) != evaluation["reconstruction_sha256"]:
        raise ValueError("reconstruction binding")
    reconstruction = np.memmap(
        args.reconstruction,
        dtype="<f8",
        mode="r",
        shape=(PANEL_BLOCKS, BLOCK_VALUES),
    )
    linear = cp.arange(ROLE_VALUES, dtype=cp.uint32).reshape(
        GROUPS_PER_ROLE, TRANSFORM_VALUES
    )
    coordinates = cp.arange(TRANSFORM_VALUES, dtype=cp.uint32)
    scored = []
    corrected_digest = hashlib.sha256()
    decoded_reconstructions: list[cp.ndarray] = []
    for decoded, role in zip(decoded_roles, ROLES, strict=True):
        role_id = int(decoded["role_id"])
        expected_ordinals = [
            ordinal
            for ordinal, block in enumerate(manifest["blocks"])
            if block["role"] == role
        ]
        if decoded["ordinals"] != expected_ordinals:
            raise ValueError("decoded role mask disagrees with manifest")
        corrected = cp.empty(
            (GROUPS_PER_ROLE, TRANSFORM_VALUES), dtype=cp.float64
        )
        for local_block, ordinal in enumerate(expected_ordinals):
            corrected[local_block * 2:(local_block + 1) * 2] = cp.asarray(
                np.asarray(reconstruction[ordinal]), dtype=cp.float64
            ).reshape(2, TRANSFORM_VALUES)
        flat = corrected.ravel()
        positions = cp.asarray(decoded["coordinate_positions"], dtype=cp.int64)
        signs = cp.asarray(decoded["coordinate_signs"], dtype=cp.float64)
        flat[positions] += signs * float(decoded["coordinate_amplitude"])
        for stage_index, stage in enumerate(decoded["stages"]):
            banks = cp.asarray(stage["banks"], dtype=cp.uint32)
            indices = cp.asarray(stage["indices"], dtype=cp.uint32)
            positive = cp.asarray(stage["positive"], dtype=cp.float64)
            signs = 2.0 * positive - 1.0
            seeds = (
                cp.uint32(role_id * SEED_ROLE_STRIDE + stage_index * PROCEDURAL_BANKS)
                + banks
            )
            diagonal = mix_signs(linear, seeds)
            atom = hadamard_signs(indices, coordinates) * diagonal
            corrected += (
                signs[:, None]
                * (float(stage["amplitude"]) / math.sqrt(TRANSFORM_VALUES))
                * atom
            )
        decoded_reconstructions.append(corrected)

    # No exact source payload is opened until all three role reconstructions
    # have been fully decoded from the wrapper.
    for decoded, role, corrected in zip(
        decoded_roles, ROLES, decoded_reconstructions, strict=True
    ):
        role_id = int(decoded["role_id"])
        expected_ordinals = [
            ordinal
            for ordinal, block in enumerate(manifest["blocks"])
            if block["role"] == role
        ]
        source = cp.empty_like(corrected)
        energy = 0.0
        source_hashes = []
        for local_block, ordinal in enumerate(expected_ordinals):
            block = manifest["blocks"][ordinal]
            source_path = Path(block["source_path"])
            if not source_path.is_absolute():
                source_path = args.source_root / source_path
            source_cpu = bf16(source_path, block["source_sha256"])
            source[local_block * 2:(local_block + 1) * 2] = cp.asarray(
                source_cpu, dtype=cp.float64
            ).reshape(2, TRANSFORM_VALUES)
            source_hashes.append(block["source_sha256"])
        energy = float(cp.sum(source * source, dtype=cp.float64).get())
        residual = source - corrected
        sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
        corrected_cpu = cp.asnumpy(corrected).astype("<f8", copy=False)
        corrected_digest.update(corrected_cpu.tobytes())
        scored.append(
            {
                "role": role,
                "role_id": role_id,
                "blocks": ROLE_BLOCKS,
                "source_hashes_verified": len(source_hashes),
                "coordinate_pulses_decoded": len(decoded["coordinate_positions"]),
                "sparc_stages_decoded": len(decoded["stages"]),
                "source_energy": energy,
                "corrected_sse": sse,
                "relative_mse": sse / energy,
            }
        )
        del corrected, source, residual, corrected_cpu
        cp.get_default_memory_pool().free_all_blocks()
    decoded_reconstructions.clear()
    del reconstruction

    physical_bytes = len(wrapper_payload)
    rate = physical_bytes * 8.0 / PANEL_VALUES
    reference = 2.0 ** (-2.0 * rate)
    for row in scored:
        row["gap_db"] = 10.0 * math.log10(row["relative_mse"] / reference)
    global_sse = float(evaluation["sse"])
    for row in scored:
        baseline = next(item for item in evaluation["by_role"] if item["role"] == row["role"])
        global_sse += row["corrected_sse"] - float(baseline["sse"])
    global_relative = global_sse / float(evaluation["source_energy"])
    global_gap = 10.0 * math.log10(global_relative / reference)

    for actual, claimed in zip(scored, receipt["roles"], strict=True):
        for actual_key, claimed_key in (
            ("source_energy", "source_energy"),
            ("corrected_sse", "corrected_sse"),
            ("relative_mse", "relative_mse"),
            ("gap_db", "gap_db"),
        ):
            if not math.isclose(
                float(actual[actual_key]),
                float(claimed[claimed_key]),
                rel_tol=8e-14,
                abs_tol=3e-11,
            ):
                raise ValueError(f"receipt mismatch: {actual['role']} {actual_key}")
    if not math.isclose(global_sse, receipt["global"]["corrected_sse"], rel_tol=8e-14, abs_tol=3e-11):
        raise ValueError("global SSE receipt mismatch")

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    result = {
        "format": "independent VJWRAP42/VJSPRC41 decoder verification v2",
        "status": "passed",
        "encoder_module_imported": False,
        "strict_ptq": True,
        "source_free_decode_completed_before_source_scoring": True,
        "wrapper": str(args.wrapper),
        "wrapper_bytes": physical_bytes,
        "wrapper_sha256": sha256_path(args.wrapper),
        "base_bundle_sha256": sha256_path(args.base_bundle),
        "embedded_base_preserved_byte_for_byte": True,
        "wrapper_external_manifest_evaluation_reconstruction_and_encoder_hashes_verified_before_source_access": True,
        "extension_sha256": sha256_bytes(extension),
        "experiment_receipt_sha256": sha256_path(args.experiment_receipt),
        "manifest_sha256": sha256_path(args.manifest),
        "evaluation_sha256": sha256_path(args.evaluation),
        "reconstruction_sha256": sha256_path(args.reconstruction),
        **wrapper_checks,
        "extension_crc_verified": True,
        "coordinate_amplitudes_finite_positive": True,
        "rice_payload_lengths_and_parameters_exact_minimal": True,
        "experiment_receipt_status_ptq_and_method_constants_verified": True,
        "role_masks_disjoint_and_verified": True,
        "payload_tamper_rejected": payload_tamper_rejected,
        "extension_crc_tamper_rejected": extension_crc_tamper_rejected,
        "recrc_encoder_provenance_tamper_rejected": provenance_tamper_rejected,
        "physical_rate_bpw": rate,
        "strictly_below_2p5_bpw": rate < 2.5,
        "gaussian_reference": reference,
        "roles": scored,
        "worst_role_gap_db": max(row["gap_db"] for row in scored),
        "all_three_roles_below_zero": all(row["gap_db"] < 0.0 for row in scored),
        "global_corrected_sse": global_sse,
        "global_relative_mse": global_relative,
        "global_gap_db": global_gap,
        "global_remains_below_zero": global_gap < 0.0,
        "corrected_three_role_reconstruction_f64_sha256": corrected_digest.hexdigest(),
        "receipt_metrics_reproduced": True,
        "cupy": cp.__version__,
        "gpu": gpu_name,
        "python": sys.version,
        "verifier_script_sha256": sha256_path(Path(__file__).resolve()),
    }
    if not result["all_three_roles_below_zero"] or not result["global_remains_below_zero"]:
        raise AssertionError("negative-gap claim failed")
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
