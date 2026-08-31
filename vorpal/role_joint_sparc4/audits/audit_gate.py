#!/usr/bin/env python3
"""Independent gate-role replay/audit for VJWRAP41 + VJSPRC41.

The module imports neither the SPARC4 builder nor its verifier.  It parses the
physical wire image, checks the embedded VORPAL base byte-for-byte, recomputes
the gate coordinate support, and replays every emitted greedy four-bank
Walsh-Hadamard stage against the frozen gate residual on CuPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
GATE_ID = 2
PANEL_BLOCKS = 400
BLOCK_VALUES = 262_144
PANEL_VALUES = PANEL_BLOCKS * BLOCK_VALUES
ROLE_BLOCKS = 48
ROLE_VALUES = ROLE_BLOCKS * BLOCK_VALUES
TRANSFORM_VALUES = 131_072
GROUPS = ROLE_VALUES // TRANSFORM_VALUES
BANKS = 4
STAGE_BITS = 20
STAGE_CODE_BYTES = 240
STAGE_RECORD_BYTES = 242
ROLE_STRIDE = 1_000_000
STRICT_MAX_BYTES = 32_767_999

WRAPPER = struct.Struct("<8sIIIII32s32s32s32s32s32sI")
EXT_HEADER = struct.Struct("<8sBBBBIII")
ROLE_DESC = struct.Struct("<BBHHHeIII")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(payload: bytes) -> np.ndarray:
    if len(payload) != BLOCK_VALUES * 2:
        raise ValueError("BF16 source byte length")
    words = np.frombuffer(payload, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


class Bits:
    def __init__(self, payload: bytes, meaningful: int):
        if meaningful < 0 or meaningful > len(payload) * 8:
            raise ValueError("bit count exceeds payload")
        self.payload = payload
        self.meaningful = meaningful
        self.offset = 0

    def read(self, width: int) -> int:
        if width < 0 or self.offset + width > self.meaningful:
            raise ValueError("truncated bitstream")
        value = 0
        for _ in range(width):
            value = (value << 1) | (
                (self.payload[self.offset >> 3] >> (7 - (self.offset & 7))) & 1
            )
            self.offset += 1
        return value


def rice_decode(payload: bytes, meaningful: int, count: int, rice_b: int):
    if not (0 <= rice_b <= 20) or len(payload) != (meaningful + 7) // 8:
        raise ValueError("noncanonical Rice dimensions")
    reader = Bits(payload, meaningful)
    positions = np.empty(count, dtype=np.int64)
    signs = np.empty(count, dtype=np.int8)
    previous = -1
    for item in range(count):
        quotient = 0
        while reader.read(1) == 0:
            quotient += 1
            if quotient > ROLE_VALUES:
                raise ValueError("unbounded Rice quotient")
        gap = (quotient << rice_b) | reader.read(rice_b)
        position = previous + gap + 1
        if position <= previous or position >= ROLE_VALUES:
            raise ValueError("Rice support noncanonical/range")
        positions[item] = position
        signs[item] = 1 if reader.read(1) else -1
        previous = position
    if reader.offset != meaningful:
        raise ValueError("unused meaningful Rice bits")
    padding = len(payload) * 8 - meaningful
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("nonzero Rice padding")
    return positions, signs, padding


def decode_stage(payload: bytes):
    if len(payload) != STAGE_CODE_BYTES:
        raise ValueError("stage byte length")
    reader = Bits(payload, GROUPS * STAGE_BITS)
    banks = np.empty(GROUPS, dtype=np.uint8)
    indices = np.empty(GROUPS, dtype=np.uint32)
    signs = np.empty(GROUPS, dtype=np.int8)
    for group in range(GROUPS):
        code = reader.read(STAGE_BITS)
        signs[group] = 1 if code & 1 else -1
        indices[group] = (code >> 1) & ((1 << 17) - 1)
        banks[group] = code >> 18
    if reader.offset != reader.meaningful or np.any(banks >= BANKS):
        raise ValueError("stage symbols noncanonical")
    return banks, indices, signs


def parse_mask(payload: bytes) -> list[int]:
    if len(payload) != 50:
        raise ValueError("role mask byte length")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if bits.size != PANEL_BLOCKS or int(bits.sum()) != ROLE_BLOCKS:
        raise ValueError("role mask census")
    return np.flatnonzero(bits).tolist()


def parse_wrapper(payload: bytes):
    if len(payload) < WRAPPER.size:
        raise ValueError("wrapper truncated")
    fields = WRAPPER.unpack_from(payload)
    (
        magic, version, header_bytes, panel_values, base_bytes, ext_bytes,
        base_hash, ext_hash, manifest_hash, evaluation_hash,
        reconstruction_hash, encoder_hash, crc,
    ) = fields
    if magic != b"VJWRAP42" or version != 2 or header_bytes != WRAPPER.size or panel_values != PANEL_VALUES:
        raise ValueError("wrapper constants")
    if zlib.crc32(payload[: WRAPPER.size - 4]) & 0xFFFFFFFF != crc:
        raise ValueError("wrapper header CRC")
    if len(payload) != WRAPPER.size + base_bytes + ext_bytes:
        raise ValueError("wrapper exact EOF/section lengths")
    base = payload[WRAPPER.size : WRAPPER.size + base_bytes]
    extension = payload[WRAPPER.size + base_bytes :]
    if sha_bytes(base) != base_hash.hex() or sha_bytes(extension) != ext_hash.hex():
        raise ValueError("wrapper section SHA-256")
    return base, extension, {
        "header_bytes": header_bytes,
        "header_crc32": f"{crc:08x}",
        "base_bytes": base_bytes,
        "base_sha256": base_hash.hex(),
        "extension_bytes": ext_bytes,
        "extension_sha256": ext_hash.hex(),
        "manifest_sha256": manifest_hash.hex(),
        "evaluation_sha256": evaluation_hash.hex(),
        "reconstruction_sha256": reconstruction_hash.hex(),
        "encoder_sha256": encoder_hash.hex(),
    }


def parse_extension(payload: bytes):
    if len(payload) < EXT_HEADER.size + 4:
        raise ValueError("extension truncated")
    expected_crc = struct.unpack_from("<I", payload, len(payload) - 4)[0]
    if zlib.crc32(payload[:-4]) & 0xFFFFFFFF != expected_crc:
        raise ValueError("extension CRC")
    header = EXT_HEADER.unpack_from(payload)
    magic, version, roles, log_n, banks, panel_blocks, block_values, groups = header
    if (
        magic != b"VJSPRC41"
        or version != 1
        or roles != 3
        or log_n != 17
        or banks != BANKS
        or panel_blocks != PANEL_BLOCKS
        or block_values != BLOCK_VALUES
        or groups != GROUPS
    ):
        raise ValueError("extension constants")
    cursor = EXT_HEADER.size
    rows = []
    masks: set[int] = set()
    for expected_role in range(3):
        if cursor + ROLE_DESC.size + 50 > len(payload) - 4:
            raise ValueError("role descriptor truncated")
        desc_offset = cursor
        role_id, rice_b, block_count, coord_count, stage_count, coord_amp, coord_bits, coord_bytes, stage_bytes = ROLE_DESC.unpack_from(payload, cursor)
        cursor += ROLE_DESC.size
        if (
            role_id != expected_role
            or rice_b > 20
            or block_count != ROLE_BLOCKS
            or coord_count != 100
            or stage_count > 450
            or not math.isfinite(float(coord_amp))
            or coord_amp <= 0
            or coord_bytes != (coord_bits + 7) // 8
            or stage_bytes != STAGE_RECORD_BYTES
        ):
            raise ValueError("role descriptor noncanonical")
        mask_offset = cursor
        ordinals = parse_mask(payload[cursor : cursor + 50])
        cursor += 50
        if masks.intersection(ordinals):
            raise ValueError("overlapping role masks")
        masks.update(ordinals)
        coord_offset = cursor
        coord_end = cursor + coord_bytes
        if coord_end > len(payload) - 4:
            raise ValueError("coordinate payload truncated")
        positions, signs, padding = rice_decode(
            payload[coord_offset:coord_end], coord_bits, coord_count, rice_b
        )
        cursor = coord_end
        stages = []
        for stage_index in range(stage_count):
            if cursor + STAGE_RECORD_BYTES > len(payload) - 4:
                raise ValueError("stage truncated")
            offset = cursor
            amp_bits = payload[cursor : cursor + 2]
            amplitude = struct.unpack("<e", amp_bits)[0]
            stage_payload = payload[cursor + 2 : cursor + STAGE_RECORD_BYTES]
            cursor += STAGE_RECORD_BYTES
            if not math.isfinite(amplitude) or amplitude <= 0:
                raise ValueError("stage amplitude invalid")
            stage_banks, stage_indices, stage_signs = decode_stage(stage_payload)
            stages.append(
                {
                    "stage": stage_index,
                    "offset": offset,
                    "amplitude": float(amplitude),
                    "amplitude_bits_hex": amp_bits.hex(),
                    "banks": stage_banks,
                    "indices": stage_indices,
                    "signs": stage_signs,
                    "payload_sha256": sha_bytes(stage_payload),
                }
            )
        rows.append(
            {
                "role_id": role_id,
                "desc_offset": desc_offset,
                "mask_offset": mask_offset,
                "ordinals": ordinals,
                "rice_b": rice_b,
                "coordinate_count": coord_count,
                "coordinate_amplitude": float(coord_amp),
                "coordinate_bits": coord_bits,
                "coordinate_bytes": coord_bytes,
                "coordinate_padding_bits": padding,
                "coordinate_offset": coord_offset,
                "coordinate_positions": positions,
                "coordinate_signs": signs,
                "stage_count": stage_count,
                "stage_bytes": stage_bytes,
                "stages": stages,
            }
        )
    if cursor != len(payload) - 4:
        raise ValueError("extension trailing bytes")
    if len(masks) != 3 * ROLE_BLOCKS:
        raise ValueError("role mask union census")
    return rows, f"{expected_crc:08x}"


def fwht(values: cp.ndarray) -> cp.ndarray:
    result = values.copy()
    width = 1
    while width < TRANSFORM_VALUES:
        view = result.reshape(result.shape[0], -1, 2, width)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        result = view.reshape(result.shape)
        width *= 2
    return result / math.sqrt(TRANSFORM_VALUES)


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


def hadamard_signs(indices: cp.ndarray, coordinates: cp.ndarray) -> cp.ndarray:
    value = indices[:, None].astype(cp.uint32) & coordinates[None, :]
    value ^= value >> cp.uint32(16)
    value ^= value >> cp.uint32(8)
    value ^= value >> cp.uint32(4)
    parity = (cp.uint32(0x6996) >> (value & cp.uint32(15))) & cp.uint32(1)
    return 1.0 - 2.0 * parity.astype(cp.float64)


def expect_failure(name: str, function, fragment: str) -> dict:
    try:
        function()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "case": name,
            "passed": fragment in message,
            "expected_fragment": fragment,
            "observed": message,
        }
    return {"case": name, "passed": False, "expected_fragment": fragment, "observed": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-receipt", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    wrapper_payload = args.wrapper.read_bytes()
    base, extension, wrapper_wire = parse_wrapper(wrapper_payload)
    if base != args.base.read_bytes():
        raise ValueError("embedded base differs byte-for-byte from normative VORPAL")
    if extension != args.extension.read_bytes():
        raise ValueError("embedded extension differs from standalone artifact")
    roles, extension_crc = parse_extension(extension)
    gate = roles[GATE_ID]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    receipt = json.loads(args.experiment_receipt.read_text(encoding="utf-8"))
    independent = json.loads(args.independent_receipt.read_text(encoding="utf-8"))
    identities = {
        "wrapper": {"bytes": len(wrapper_payload), "sha256": sha_bytes(wrapper_payload)},
        "extension": {"bytes": len(extension), "sha256": sha_bytes(extension)},
        "base": {"bytes": len(base), "sha256": sha_bytes(base)},
        "manifest": {"bytes": args.manifest.stat().st_size, "sha256": sha_path(args.manifest)},
        "evaluation": {"bytes": args.evaluation.stat().st_size, "sha256": sha_path(args.evaluation)},
        "reconstruction": {"bytes": args.reconstruction.stat().st_size, "sha256": sha_path(args.reconstruction)},
        "experiment_receipt": {"bytes": args.experiment_receipt.stat().st_size, "sha256": sha_path(args.experiment_receipt)},
        "independent_receipt": {"bytes": args.independent_receipt.stat().st_size, "sha256": sha_path(args.independent_receipt)},
        "builder": {"bytes": args.builder.stat().st_size, "sha256": sha_path(args.builder)},
        "verifier": {"bytes": args.verifier.stat().st_size, "sha256": sha_path(args.verifier)},
    }
    input_claims = receipt["inputs"]
    artifact_claims = receipt["artifacts"]
    expected_equalities = (
        (identities["manifest"]["sha256"], input_claims["manifest_sha256"], "manifest"),
        (identities["evaluation"]["sha256"], input_claims["evaluation_sha256"], "evaluation"),
        (identities["reconstruction"]["sha256"], input_claims["reconstruction_sha256"], "reconstruction"),
        (identities["base"]["sha256"], input_claims["base_bundle_sha256"], "base"),
        (identities["extension"]["sha256"], artifact_claims["extension_sha256"], "extension"),
        (identities["wrapper"]["sha256"], artifact_claims["emitted_wrapper_sha256"], "wrapper"),
        (identities["builder"]["sha256"], receipt["environment"]["script_sha256"], "builder"),
        (identities["verifier"]["sha256"], independent["verifier_script_sha256"], "verifier"),
    )
    for actual, claimed, label in expected_equalities:
        if actual != claimed:
            raise ValueError(f"{label} provenance hash mismatch")
    if evaluation["encoded_sha256"] != identities["base"]["sha256"]:
        raise ValueError("evaluation does not bind normative base")
    if evaluation["reconstruction_sha256"] != identities["reconstruction"]["sha256"]:
        raise ValueError("evaluation does not bind reconstruction")
    for key, identity_key in (
        ("manifest_sha256", "manifest"),
        ("evaluation_sha256", "evaluation"),
        ("reconstruction_sha256", "reconstruction"),
        ("encoder_sha256", "builder"),
    ):
        if wrapper_wire[key] != identities[identity_key]["sha256"]:
            raise ValueError(f"wrapper {key} provenance binding mismatch")

    expected_ordinals = [
        ordinal for ordinal, block in enumerate(manifest["blocks"])
        if block["role"] == ROLES[GATE_ID]
    ]
    if gate["ordinals"] != expected_ordinals or len(expected_ordinals) != ROLE_BLOCKS:
        raise ValueError("gate mask disagrees with manifest")
    layers = []
    source_rows = []
    source_gpu = cp.empty((GROUPS, TRANSFORM_VALUES), dtype=cp.float64)
    reconstruction = np.memmap(
        args.reconstruction, dtype="<f8", mode="r", shape=(PANEL_BLOCKS, BLOCK_VALUES)
    )
    reconstruction_gpu = cp.empty_like(source_gpu)
    for local_block, ordinal in enumerate(expected_ordinals):
        block = manifest["blocks"][ordinal]
        source_path = Path(block["source_path"])
        if not source_path.is_absolute():
            source_path = args.source_root / source_path
        payload = source_path.read_bytes()
        actual_hash = sha_bytes(payload)
        if actual_hash != block["source_sha256"]:
            raise ValueError(f"source hash mismatch at gate ordinal {ordinal}")
        match = re.search(r"model\.layers\.(\d+)\.", block["tensor"])
        if match is None:
            match = re.match(r"l(\d+)-", block["id"])
        if match is None:
            raise ValueError("gate layer identity")
        layer = int(match.group(1))
        layers.append(layer)
        source_rows.append(
            {"ordinal": ordinal, "id": block["id"], "layer": layer, "bytes": len(payload), "sha256": actual_hash}
        )
        source_gpu[local_block * 2 : (local_block + 1) * 2] = cp.asarray(
            bf16(payload), dtype=cp.float64
        ).reshape(2, TRANSFORM_VALUES)
        reconstruction_gpu[local_block * 2 : (local_block + 1) * 2] = cp.asarray(
            np.asarray(reconstruction[ordinal]), dtype=cp.float64
        ).reshape(2, TRANSFORM_VALUES)
    del reconstruction
    if layers != list(range(48)):
        raise ValueError("gate role is not exactly one block from layers 0..47")

    residual = source_gpu - reconstruction_gpu
    energy = float(cp.sum(source_gpu * source_gpu, dtype=cp.float64).get())
    base_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
    flat = residual.ravel()
    absolute = cp.abs(flat)
    k = gate["coordinate_count"]
    partition = cp.argpartition(absolute, -k)[-k:]
    emitted_positions_gpu = cp.asarray(gate["coordinate_positions"], dtype=cp.int64)
    cutoff = absolute[partition].min()
    greater = cp.flatnonzero(absolute > cutoff)
    tied = cp.flatnonzero(absolute == cutoff)
    selected_ties = k - int(greater.size)
    canonical = cp.sort(cp.concatenate((greater, tied[:selected_ties])))
    canonical_positions = cp.asnumpy(canonical).astype(np.int64)
    support_is_canonical = np.array_equal(canonical_positions, gate["coordinate_positions"])
    # Also replay the encoder's exact CuPy argpartition/argsort sequence.
    ordered = partition[cp.argsort(-absolute[partition])]
    encoder_positions = np.sort(cp.asnumpy(ordered).astype(np.int64), kind="stable")
    if not np.array_equal(encoder_positions, gate["coordinate_positions"]):
        raise ValueError("gate coordinate support differs from exact encoder replay")
    emitted_signs_gpu = cp.asarray(gate["coordinate_signs"], dtype=cp.float64)
    expected_signs = cp.asnumpy(cp.sign(flat[emitted_positions_gpu])).astype(np.int8)
    if not np.array_equal(expected_signs, gate["coordinate_signs"]):
        raise ValueError("gate coordinate signs differ from residual")
    expected_coord_amp = np.float16(
        float(cp.mean(absolute[emitted_positions_gpu], dtype=cp.float64).get())
    )
    if struct.pack("<e", float(expected_coord_amp)) != struct.pack("<e", gate["coordinate_amplitude"]):
        raise ValueError("gate coordinate amplitude differs from exact replay")
    before_coordinate = base_sse
    flat[emitted_positions_gpu] -= emitted_signs_gpu * float(gate["coordinate_amplitude"])
    after_coordinate = float(cp.sum(residual * residual, dtype=cp.float64).get())

    linear = cp.arange(ROLE_VALUES, dtype=cp.uint32).reshape(GROUPS, TRANSFORM_VALUES)
    coordinates = cp.arange(TRANSFORM_VALUES, dtype=cp.uint32)
    group_rows = cp.arange(GROUPS)
    mismatch_stages = []
    stage_gain_sum = 0.0
    stage_amp_digest = hashlib.sha256()
    stage_code_digest = hashlib.sha256()
    replay_started = time.time()
    for stage_index, emitted in enumerate(gate["stages"]):
        best_abs = cp.zeros(GROUPS, dtype=cp.float64)
        best_coefficient = cp.zeros(GROUPS, dtype=cp.float64)
        best_index = cp.zeros(GROUPS, dtype=cp.uint32)
        best_bank = cp.zeros(GROUPS, dtype=cp.uint32)
        for bank in range(BANKS):
            seed = GATE_ID * ROLE_STRIDE + stage_index * BANKS + bank
            diagonal = mix_signs(linear, seed)
            transformed = fwht(residual * diagonal)
            index = cp.argmax(cp.abs(transformed), axis=1).astype(cp.uint32)
            coefficient = transformed[group_rows, index]
            magnitude = cp.abs(coefficient)
            replace = magnitude > best_abs
            best_abs = cp.where(replace, magnitude, best_abs)
            best_coefficient = cp.where(replace, coefficient, best_coefficient)
            best_index = cp.where(replace, index, best_index)
            best_bank = cp.where(replace, bank, best_bank)
        expected_amp = np.float16(float(cp.mean(best_abs, dtype=cp.float64).get()))
        actual_banks = cp.asnumpy(best_bank).astype(np.uint8)
        actual_indices = cp.asnumpy(best_index).astype(np.uint32)
        actual_signs = cp.asnumpy(cp.where(best_coefficient > 0, 1, -1)).astype(np.int8)
        fields_equal = (
            np.array_equal(actual_banks, emitted["banks"])
            and np.array_equal(actual_indices, emitted["indices"])
            and np.array_equal(actual_signs, emitted["signs"])
            and struct.pack("<e", float(expected_amp)) == bytes.fromhex(emitted["amplitude_bits_hex"])
        )
        if not fields_equal:
            mismatch_stages.append(
                {
                    "stage": stage_index,
                    "bank_differences": int(np.count_nonzero(actual_banks != emitted["banks"])),
                    "index_differences": int(np.count_nonzero(actual_indices != emitted["indices"])),
                    "sign_differences": int(np.count_nonzero(actual_signs != emitted["signs"])),
                    "expected_amplitude": float(expected_amp),
                    "emitted_amplitude": emitted["amplitude"],
                }
            )
            raise ValueError(f"gate SPARC stage {stage_index} differs from greedy replay")
        amplitude = float(emitted["amplitude"])
        gain = float(cp.sum(2.0 * amplitude * best_abs - amplitude * amplitude, dtype=cp.float64).get())
        stage_gain_sum += gain
        seeds = cp.uint32(GATE_ID * ROLE_STRIDE + stage_index * BANKS) + best_bank
        atom = hadamard_signs(best_index, coordinates) * mix_signs(linear, seeds)
        residual -= cp.sign(best_coefficient)[:, None] * (
            amplitude / math.sqrt(TRANSFORM_VALUES)
        ) * atom
        stage_amp_digest.update(bytes.fromhex(emitted["amplitude_bits_hex"]))
        # Hash the decoded logical fields, independent of builder serialization.
        stage_code_digest.update(actual_banks.tobytes())
        stage_code_digest.update(actual_indices.astype("<u4", copy=False).tobytes())
        stage_code_digest.update(actual_signs.tobytes())
        if (stage_index + 1) % 25 == 0 or stage_index + 1 == gate["stage_count"]:
            current_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
            print(
                f"gate replay {stage_index + 1}/{gate['stage_count']} "
                f"sse={current_sse:.15f} elapsed={time.time() - replay_started:.1f}s",
                flush=True,
            )

    corrected_sse = float(cp.sum(residual * residual, dtype=cp.float64).get())
    corrected_relative = corrected_sse / energy
    physical_bytes = len(wrapper_payload)
    rate = physical_bytes * 8.0 / PANEL_VALUES
    gaussian = 2.0 ** (-2.0 * rate)
    gap = 10.0 * math.log10(corrected_relative / gaussian)
    gate_claim = receipt["roles"][GATE_ID]
    independent_gate = independent["roles"][GATE_ID]
    for label, actual, claimed, tolerance in (
        ("energy receipt", energy, gate_claim["source_energy"], 3e-11),
        ("base SSE receipt", base_sse, gate_claim["base_sse"], 3e-11),
        ("corrected SSE receipt", corrected_sse, gate_claim["corrected_sse"], 3e-11),
        ("gate gap receipt", gap, gate_claim["gap_db"], 3e-11),
        ("corrected SSE independent", corrected_sse, independent_gate["corrected_sse"], 3e-11),
        ("gate gap independent", gap, independent_gate["gap_db"], 3e-11),
    ):
        if not math.isclose(float(actual), float(claimed), rel_tol=8e-14, abs_tol=tolerance):
            raise ValueError(f"metric mismatch: {label}: {actual} vs {claimed}")

    tamper_tests = []
    changed = bytearray(wrapper_payload)
    changed[0] ^= 1
    tamper_tests.append(expect_failure("wrapper_header_bit", lambda: parse_wrapper(bytes(changed)), "wrapper constants"))
    changed = bytearray(wrapper_payload)
    changed[WRAPPER.size] ^= 1
    tamper_tests.append(expect_failure("embedded_base_bit", lambda: parse_wrapper(bytes(changed)), "section SHA-256"))
    changed = bytearray(wrapper_payload)
    changed[-17] ^= 1
    tamper_tests.append(expect_failure("embedded_extension_bit", lambda: parse_wrapper(bytes(changed)), "section SHA-256"))
    tamper_tests.append(expect_failure("truncated_EOF", lambda: parse_wrapper(wrapper_payload[:-1]), "exact EOF"))
    tamper_tests.append(expect_failure("appended_EOF", lambda: parse_wrapper(wrapper_payload + b"\0"), "exact EOF"))
    # Change the encoder identity and recompute the header CRC.  Structural
    # parsing succeeds, but external provenance validation must fail.
    changed_fields = list(WRAPPER.unpack_from(wrapper_payload))
    changed_encoder = bytearray(changed_fields[11])
    changed_encoder[0] ^= 1
    changed_fields[11] = bytes(changed_encoder)
    changed_fields[12] = 0
    provisional = WRAPPER.pack(*changed_fields)
    changed_fields[12] = zlib.crc32(provisional[:-4]) & 0xFFFFFFFF
    changed_header = WRAPPER.pack(*changed_fields)
    changed_wrapper = changed_header + wrapper_payload[WRAPPER.size:]

    def parse_and_validate_changed_encoder() -> None:
        _, _, changed_wire = parse_wrapper(changed_wrapper)
        if changed_wire["encoder_sha256"] != identities["builder"]["sha256"]:
            raise ValueError("encoder provenance binding mismatch")

    tamper_tests.append(expect_failure(
        "encoder_provenance_bit_with_valid_crc",
        parse_and_validate_changed_encoder,
        "encoder provenance binding mismatch",
    ))
    changed_ext = bytearray(extension)
    changed_ext[-5] ^= 1
    tamper_tests.append(expect_failure("extension_crc_bit", lambda: parse_extension(bytes(changed_ext)), "extension CRC"))
    # Recompute extension CRC after setting a meaningful padding bit: syntax,
    # rather than only integrity, must reject it.
    if gate["coordinate_padding_bits"]:
        changed_ext = bytearray(extension)
        coordinate_last = gate["coordinate_offset"] + gate["coordinate_bytes"] - 1
        changed_ext[coordinate_last] ^= 1
        changed_ext[-4:] = struct.pack("<I", zlib.crc32(changed_ext[:-4]) & 0xFFFFFFFF)
        tamper_tests.append(expect_failure("coordinate_nonzero_padding_with_valid_crc", lambda: parse_extension(bytes(changed_ext)), "nonzero Rice padding"))
    # Make gate mask overlap up while keeping its population fixed and CRC valid.
    changed_ext = bytearray(extension)
    gate_mask_offset = gate["mask_offset"]
    gate_first = gate["ordinals"][0]
    up_first = roles[0]["ordinals"][0]
    changed_ext[gate_mask_offset + gate_first // 8] &= ~(1 << (gate_first & 7))
    changed_ext[gate_mask_offset + up_first // 8] |= 1 << (up_first & 7)
    changed_ext[-4:] = struct.pack("<I", zlib.crc32(changed_ext[:-4]) & 0xFFFFFFFF)
    tamper_tests.append(expect_failure("overlapping_role_mask_with_valid_crc", lambda: parse_extension(bytes(changed_ext)), "overlapping role masks"))
    if not all(row["passed"] for row in tamper_tests):
        raise AssertionError("one or more tamper cases failed open")

    audit = {
        "format": "independent gate SPARC4 exact-selection replay audit v1",
        "status": "passed" if gap < 0 and not mismatch_stages else "failed",
        "strict_ptq": True,
        "training_or_retraining": False,
        "calibration_or_activations": False,
        "encoder_module_imported": False,
        "identities": identities,
        "wrapper_wire": {
            **wrapper_wire,
            "wrapper_sha256": sha_bytes(wrapper_payload),
            "exact_eof": True,
            "embedded_base_byte_for_byte_equal": True,
            "extension_byte_for_byte_equal": True,
            "strict_max_bytes": STRICT_MAX_BYTES,
            "headroom_bytes": STRICT_MAX_BYTES - physical_bytes,
            "physical_rate_bpw": rate,
        },
        "extension_wire": {
            "crc32": extension_crc,
            "roles": 3,
            "role_masks_disjoint": True,
            "total_stages": sum(row["stage_count"] for row in roles),
            "stage_counts": [row["stage_count"] for row in roles],
            "no_trailing_bytes": True,
        },
        "gate": {
            "role": ROLES[GATE_ID],
            "ordinals": expected_ordinals,
            "layers": layers,
            "source_hashes_verified": len(source_rows),
            "source_energy": energy,
            "base_sse": base_sse,
            "coordinate": {
                "count": k,
                "rice_b": gate["rice_b"],
                "meaningful_bits": gate["coordinate_bits"],
                "payload_bytes": gate["coordinate_bytes"],
                "zero_padding_bits": gate["coordinate_padding_bits"],
                "cutoff_magnitude_f64": float(cutoff.get()),
                "strictly_greater_count": int(greater.size),
                "total_cutoff_ties": int(tied.size),
                "selected_cutoff_ties": selected_ties,
                "canonical_index_tie_rule_equal": support_is_canonical,
                "exact_encoder_sequence_support_equal": True,
                "signs_equal": True,
                "amplitude_fp16": gate["coordinate_amplitude"],
                "amplitude_exact_replay_equal": True,
                "sse_gain": before_coordinate - after_coordinate,
            },
            "sparc": {
                "stages": gate["stage_count"],
                "groups_per_stage": GROUPS,
                "procedural_banks_per_group": BANKS,
                "all_bank_index_sign_fields_exact": True,
                "all_fp16_amplitudes_exact": True,
                "mismatch_stages": mismatch_stages,
                "stage_gain_sum": stage_gain_sum,
                "amplitude_bits_sha256": stage_amp_digest.hexdigest(),
                "decoded_logical_codes_sha256": stage_code_digest.hexdigest(),
                "elapsed_seconds": time.time() - replay_started,
            },
            "corrected_sse": corrected_sse,
            "corrected_relative_mse": corrected_relative,
            "gaussian_reference": gaussian,
            "signed_gaussian_gap_db": gap,
            "below_zero_db": gap < 0,
        },
        "source_audit": source_rows,
        "tamper_tests": tamper_tests,
        "provenance": {
            "all_receipt_input_and_script_hashes_recomputed": True,
            "evaluation_binds_base_and_reconstruction": True,
            "note": (
                "Manifest/evaluation/reconstruction/script identities are external receipt evidence; "
                "the physical wrapper directly binds only the embedded base and extension."
            ),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "output": str(args.output),
        "output_sha256": sha_path(args.output),
        "wrapper_wire": audit["wrapper_wire"],
        "extension_wire": audit["extension_wire"],
        "gate_coordinate": audit["gate"]["coordinate"],
        "gate_sparc": audit["gate"]["sparc"],
        "gate_corrected_sse": corrected_sse,
        "gate_gap_db": gap,
        "tamper_tests": tamper_tests,
    }, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
