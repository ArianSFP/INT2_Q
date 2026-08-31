#!/usr/bin/env python3
"""Independent up-role audit of the VORPAL-preserving SPARC4 wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import cupy as cp
import numpy as np


ROLE = "mlp.experts.{expert}.up_proj.weight"
PANEL_BLOCKS = 400
BLOCK_VALUES = 262_144
PANEL_VALUES = PANEL_BLOCKS * BLOCK_VALUES
ROLE_BLOCKS = 48
ROLE_VALUES = ROLE_BLOCKS * BLOCK_VALUES
TRANSFORM_VALUES = 131_072
GROUPS = 96
BANKS = 4
STAGE_CODE_BYTES = 240
STAGE_RECORD_BYTES = 242
WRAPPER_V1 = struct.Struct("<8sIIIII32s32sI")
WRAPPER_V2 = struct.Struct("<8sIIIII32s32s32s32s32s32sI")
EXT_HEADER = struct.Struct("<8sBBBBIII")
ROLE_DESCRIPTOR = struct.Struct("<BBHHHeIII")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Bits:
    def __init__(self, payload: bytes, meaningful: int) -> None:
        if meaningful < 0 or len(payload) != (meaningful + 7) // 8:
            raise ValueError("nonminimal bit payload")
        self.payload = payload
        self.meaningful = meaningful
        self.offset = 0

    def read(self, width: int) -> int:
        if width < 0 or self.offset + width > self.meaningful:
            raise ValueError("truncated bit payload")
        value = 0
        for _ in range(width):
            value = (value << 1) | (
                (self.payload[self.offset >> 3] >> (7 - (self.offset & 7))) & 1
            )
            self.offset += 1
        return value

    def finish(self) -> None:
        if self.offset != self.meaningful:
            raise ValueError("unused meaningful bits")
        if self.meaningful % 8:
            mask = (1 << (8 - self.meaningful % 8)) - 1
            if self.payload[-1] & mask:
                raise ValueError("nonzero padding bits")


def parse_wrapper(payload: bytes) -> tuple[bytes, bytes, dict]:
    if len(payload) < WRAPPER_V1.size:
        raise ValueError("truncated wrapper")
    magic = payload[:8]
    wrapper_struct = WRAPPER_V2 if magic == b"VJWRAP42" else WRAPPER_V1
    fields = wrapper_struct.unpack_from(payload)
    magic, version, header_bytes, panel_values, base_bytes, extension_bytes = fields[:6]
    base_hash, extension_hash = fields[6:8]
    external_hashes = fields[8:-1]
    crc = fields[-1]
    if (
        magic not in (b"VJWRAP41", b"VJWRAP42")
        or version != (2 if magic == b"VJWRAP42" else 1)
        or header_bytes != wrapper_struct.size
        or panel_values != PANEL_VALUES
    ):
        raise ValueError("wrapper constants")
    if zlib.crc32(payload[: wrapper_struct.size - 4]) & 0xFFFFFFFF != crc:
        raise ValueError("wrapper CRC")
    if len(payload) != wrapper_struct.size + base_bytes + extension_bytes:
        raise ValueError("wrapper EOF/length")
    base = payload[wrapper_struct.size : wrapper_struct.size + base_bytes]
    extension = payload[wrapper_struct.size + base_bytes :]
    if sha_bytes(base) != base_hash.hex() or sha_bytes(extension) != extension_hash.hex():
        raise ValueError("wrapper member hash")
    return base, extension, {
        "header_bytes": header_bytes,
        "base_bytes": base_bytes,
        "extension_bytes": extension_bytes,
        "base_sha256": base_hash.hex(),
        "extension_sha256": extension_hash.hex(),
        "header_crc32": f"{crc:08x}",
        "manifest_sha256": external_hashes[0].hex() if len(external_hashes) == 4 else None,
        "evaluation_sha256": external_hashes[1].hex() if len(external_hashes) == 4 else None,
        "reconstruction_sha256": external_hashes[2].hex() if len(external_hashes) == 4 else None,
        "encoder_sha256": external_hashes[3].hex() if len(external_hashes) == 4 else None,
    }


def decode_rice(payload: bytes, meaningful: int, count: int, rice_b: int):
    if not (0 <= rice_b <= 20) or count <= 0:
        raise ValueError("Rice descriptor")
    reader = Bits(payload, meaningful)
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
            raise ValueError("Rice support")
        positions[index] = position
        signs[index] = 1 if reader.read(1) else -1
        previous = position
    reader.finish()
    return positions, signs


def parse_stage(payload: bytes):
    if len(payload) != STAGE_CODE_BYTES:
        raise ValueError("stage bytes")
    reader = Bits(payload, len(payload) * 8)
    banks = np.empty(GROUPS, dtype=np.uint32)
    indices = np.empty(GROUPS, dtype=np.uint32)
    positive = np.empty(GROUPS, dtype=bool)
    for group in range(GROUPS):
        code = reader.read(20)
        positive[group] = bool(code & 1)
        indices[group] = (code >> 1) & 0x1FFFF
        banks[group] = code >> 18
    reader.finish()
    if np.any(banks >= BANKS) or np.any(indices >= TRANSFORM_VALUES):
        raise ValueError("stage symbol")
    return banks, indices, positive


def parse_extension(extension: bytes) -> list[dict]:
    if len(extension) < EXT_HEADER.size + 4:
        raise ValueError("extension truncation")
    if zlib.crc32(extension[:-4]) & 0xFFFFFFFF != struct.unpack_from(
        "<I", extension, len(extension) - 4
    )[0]:
        raise ValueError("extension CRC")
    if EXT_HEADER.unpack_from(extension) != (
        b"VJSPRC41", 1, 3, 17, 4, PANEL_BLOCKS, BLOCK_VALUES, GROUPS
    ):
        raise ValueError("extension constants")
    cursor = EXT_HEADER.size
    rows = []
    occupied: set[int] = set()
    for expected_role in range(3):
        if cursor + ROLE_DESCRIPTOR.size + 50 > len(extension) - 4:
            raise ValueError("role truncation")
        (
            role_id,
            rice_b,
            block_count,
            coordinate_count,
            stage_count,
            coordinate_amplitude,
            coordinate_bits,
            coordinate_bytes,
            stage_bytes,
        ) = ROLE_DESCRIPTOR.unpack_from(extension, cursor)
        cursor += ROLE_DESCRIPTOR.size
        if (
            role_id != expected_role
            or rice_b > 20
            or block_count != ROLE_BLOCKS
            or coordinate_count != 100
            or stage_count > 450
            or not math.isfinite(coordinate_amplitude)
            or coordinate_amplitude <= 0.0
            or coordinate_bytes != (coordinate_bits + 7) // 8
            or stage_bytes != STAGE_RECORD_BYTES
        ):
            raise ValueError("role descriptor")
        mask_payload = extension[cursor : cursor + 50]
        cursor += 50
        mask = np.unpackbits(
            np.frombuffer(mask_payload, dtype=np.uint8), bitorder="little"
        )
        ordinals = np.flatnonzero(mask[:PANEL_BLOCKS]).astype(int).tolist()
        if len(ordinals) != ROLE_BLOCKS or np.any(mask[PANEL_BLOCKS:]):
            raise ValueError("role mask")
        if occupied.intersection(ordinals):
            raise ValueError("overlapping role masks")
        occupied.update(ordinals)
        end = cursor + coordinate_bytes
        if end > len(extension) - 4:
            raise ValueError("coordinate truncation")
        coordinate_payload = extension[cursor:end]
        cursor = end
        positions, signs = decode_rice(
            coordinate_payload, coordinate_bits, coordinate_count, rice_b
        )
        stages = []
        stage_code_hasher = hashlib.sha256()
        for _ in range(stage_count):
            if cursor + STAGE_RECORD_BYTES > len(extension) - 4:
                raise ValueError("stage truncation")
            amplitude = struct.unpack_from("<e", extension, cursor)[0]
            stage_payload = extension[cursor + 2 : cursor + STAGE_RECORD_BYTES]
            cursor += STAGE_RECORD_BYTES
            if not math.isfinite(amplitude) or amplitude <= 0.0:
                raise ValueError("stage amplitude")
            banks, indices, positive = parse_stage(stage_payload)
            stage_code_hasher.update(stage_payload)
            stages.append(
                {
                    "amplitude": float(amplitude),
                    "banks": banks,
                    "indices": indices,
                    "positive": positive,
                }
            )
        rows.append(
            {
                "role_id": role_id,
                "ordinals": ordinals,
                "coordinate_amplitude": float(coordinate_amplitude),
                "coordinate_bits": coordinate_bits,
                "coordinate_bytes": coordinate_bytes,
                "coordinate_payload_sha256": sha_bytes(coordinate_payload),
                "coordinate_positions": positions,
                "coordinate_signs": signs,
                "stages": stages,
                "stage_codes_sha256": stage_code_hasher.hexdigest(),
            }
        )
    if cursor != len(extension) - 4:
        raise ValueError("extension trailing bytes")
    return rows


def bf16(path: Path, expected_sha: str) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) != BLOCK_VALUES * 2 or sha_bytes(payload) != expected_sha:
        raise ValueError(f"source mismatch: {path}")
    words = np.frombuffer(payload, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def mix_signs(linear: cp.ndarray, seeds: cp.ndarray | int) -> cp.ndarray:
    if isinstance(seeds, int):
        offset = cp.uint32(((seeds + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    else:
        offset = (seeds[:, None] + cp.uint32(1)) * cp.uint32(0x9E3779B9)
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


def fwht(values: cp.ndarray) -> cp.ndarray:
    transformed = values.copy()
    width = 1
    while width < TRANSFORM_VALUES:
        view = transformed.reshape(transformed.shape[0], -1, 2, width)
        left = view[:, :, 0].copy()
        right = view[:, :, 1].copy()
        view[:, :, 0] = left + right
        view[:, :, 1] = left - right
        transformed = view.reshape(transformed.shape)
        width *= 2
    return transformed / math.sqrt(TRANSFORM_VALUES)


def scalar_mix_sign(linear: int, seed: int) -> int:
    mask = 0xFFFFFFFF
    value = (linear + ((seed + 1) * 0x9E3779B9 & mask)) & mask
    value ^= value >> 16
    value = value * 0x7FEB352D & mask
    value ^= value >> 15
    value = value * 0x846CA68B & mask
    value ^= value >> 16
    return 1 if (value & 1) == 0 else -1


def canonical_top(magnitudes: cp.ndarray, count: int) -> cp.ndarray:
    provisional = cp.argpartition(magnitudes, -count)[-count:]
    cutoff = cp.min(magnitudes[provisional])
    greater = cp.flatnonzero(magnitudes > cutoff)
    needed = count - int(greater.size)
    equals = cp.flatnonzero(magnitudes == cutoff)[:needed]
    selected = cp.concatenate((greater, equals))
    order = cp.lexsort(
        cp.stack((selected.astype(cp.float64), -magnitudes[selected]))
    )
    return selected[order]


def close(left: float, right: float, tolerance: float = 3e-10) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError((left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-receipt", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wrapper_payload = args.wrapper.read_bytes()
    base, extension, wrapper = parse_wrapper(wrapper_payload)
    roles = parse_extension(extension)
    up = roles[0]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    receipt = json.loads(args.experiment_receipt.read_text(encoding="utf-8"))
    independent = json.loads(args.independent_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("strict_ptq") is not True
        or receipt.get("training_or_retraining") is not False
        or receipt.get("calibration_or_activations") is not False
        or receipt["inputs"]["manifest_sha256"] != sha(args.manifest)
        or receipt["inputs"]["evaluation_sha256"] != sha(args.evaluation)
        or receipt["inputs"]["reconstruction_sha256"] != sha(args.reconstruction)
        or receipt["inputs"]["base_bundle_sha256"] != sha(args.base_bundle)
        or receipt["inputs"]["base_bundle_bytes"] != len(base)
        or receipt["artifacts"]["extension_sha256"] != sha_bytes(extension)
        or receipt["artifacts"]["extension_bytes"] != len(extension)
        or receipt["artifacts"]["emitted_wrapper_sha256"] != sha_bytes(wrapper_payload)
        or receipt["artifacts"]["emitted_wrapper_bytes"] != len(wrapper_payload)
        or receipt["environment"]["script_sha256"] != sha(args.encoder)
        or independent["verifier_script_sha256"] != sha(args.verifier)
        or independent["wrapper_sha256"] != sha_bytes(wrapper_payload)
    ):
        raise AssertionError("provenance mismatch")
    if wrapper["manifest_sha256"] is not None and (
        wrapper["manifest_sha256"] != sha(args.manifest)
        or wrapper["evaluation_sha256"] != sha(args.evaluation)
        or wrapper["reconstruction_sha256"] != sha(args.reconstruction)
        or wrapper["encoder_sha256"] != sha(args.encoder)
    ):
        raise AssertionError("physical wrapper provenance mismatch")
    if base != args.base_bundle.read_bytes() or sha_bytes(base) != wrapper["base_sha256"]:
        raise AssertionError("embedded base mismatch")
    if sha(args.reconstruction) != evaluation["reconstruction_sha256"]:
        raise AssertionError("published reconstruction mismatch")
    expected_ordinals = [
        index for index, block in enumerate(manifest["blocks"]) if block["role"] == ROLE
    ]
    if up["ordinals"] != expected_ordinals:
        raise AssertionError("up mask mismatch")

    panel = np.memmap(
        args.reconstruction,
        dtype="<f8",
        mode="r",
        shape=(PANEL_BLOCKS, BLOCK_VALUES),
    )
    source = cp.empty((GROUPS, TRANSFORM_VALUES), dtype=cp.float64)
    baseline = cp.empty_like(source)
    verified_sources = []
    receipt_up = next(row for row in receipt["roles"] if row["role"] == ROLE)
    receipt_sources = {int(row["ordinal"]): row for row in receipt_up["sources"]}
    for local_block, ordinal in enumerate(expected_ordinals):
        block = manifest["blocks"][ordinal]
        source_path = Path(block["source_path"])
        if not source_path.is_absolute():
            source_path = args.source_root / source_path
        source_cpu = bf16(source_path, block["source_sha256"])
        if receipt_sources[ordinal]["sha256"] != block["source_sha256"]:
            raise AssertionError("receipt source mismatch")
        source[local_block * 2 : (local_block + 1) * 2] = cp.asarray(
            source_cpu, dtype=cp.float64
        ).reshape(2, TRANSFORM_VALUES)
        baseline[local_block * 2 : (local_block + 1) * 2] = cp.asarray(
            np.asarray(panel[ordinal]), dtype=cp.float64
        ).reshape(2, TRANSFORM_VALUES)
        verified_sources.append(
            {"ordinal": ordinal, "id": block["id"], "sha256": block["source_sha256"]}
        )
    residual = source - baseline
    base_sse = float(cp.asnumpy(cp.sum(residual * residual, dtype=cp.float64)))
    energy = float(cp.asnumpy(cp.sum(source * source, dtype=cp.float64)))
    close(base_sse, receipt_up["base_sse"])
    close(energy, receipt_up["source_energy"])

    magnitudes = cp.abs(residual.ravel())
    ranked = canonical_top(magnitudes, 101)
    top100 = ranked[:100]
    canonical_positions = cp.asnumpy(cp.sort(top100)).astype(np.int64)
    sort_order = cp.argsort(top100)
    canonical_signs = cp.asnumpy(cp.sign(residual.ravel()[top100][sort_order])).astype(
        np.int8
    )
    if not np.array_equal(canonical_positions, up["coordinate_positions"]):
        raise AssertionError("up coordinate support is not canonical top-100")
    if not np.array_equal(canonical_signs, up["coordinate_signs"]):
        raise AssertionError("up coordinate signs mismatch")
    expected_coordinate_amplitude = float(
        np.float16(float(cp.asnumpy(cp.mean(magnitudes[top100], dtype=cp.float64))))
    )
    if expected_coordinate_amplitude != up["coordinate_amplitude"]:
        raise AssertionError("up coordinate amplitude mismatch")
    cutoff_margin = float(
        cp.asnumpy(magnitudes[ranked[99]] - magnitudes[ranked[100]])
    )
    if cutoff_margin <= 0.0:
        raise AssertionError("coordinate top-100 cutoff is tied")
    residual.ravel()[cp.asarray(up["coordinate_positions"])] -= (
        cp.asarray(up["coordinate_signs"], dtype=cp.float64)
        * up["coordinate_amplitude"]
    )

    linear = cp.arange(ROLE_VALUES, dtype=cp.uint32).reshape(GROUPS, TRANSFORM_VALUES)
    coordinates = cp.arange(TRANSFORM_VALUES, dtype=cp.uint32)
    checkpoints = {0, 1, 2, 17, 63, 127, len(up["stages"]) - 1}
    checkpoint_rows = []
    probe_coordinates = (0, 1, 2, 3, 17, 255, 1024, 65535, 131071)
    basis_signature = hashlib.sha256()
    for stage_index, stage in enumerate(up["stages"]):
        if stage_index in checkpoints:
            best_abs = cp.zeros(GROUPS, dtype=cp.float64)
            best_coefficient = cp.zeros(GROUPS, dtype=cp.float64)
            best_index = cp.zeros(GROUPS, dtype=cp.uint32)
            best_bank = cp.zeros(GROUPS, dtype=cp.uint32)
            for bank in range(BANKS):
                diagonal = mix_signs(linear, stage_index * BANKS + bank)
                transformed = fwht(residual * diagonal)
                indices = cp.argmax(cp.abs(transformed), axis=1).astype(cp.uint32)
                coefficients = transformed[cp.arange(GROUPS), indices]
                absolute = cp.abs(coefficients)
                replace = absolute > best_abs
                best_abs = cp.where(replace, absolute, best_abs)
                best_coefficient = cp.where(replace, coefficients, best_coefficient)
                best_index = cp.where(replace, indices, best_index)
                best_bank = cp.where(replace, bank, best_bank)
            expected_amplitude = float(
                np.float16(float(cp.asnumpy(cp.mean(best_abs, dtype=cp.float64))))
            )
            if not np.array_equal(cp.asnumpy(best_bank), stage["banks"]):
                raise AssertionError(f"bank search mismatch at stage {stage_index}")
            if not np.array_equal(cp.asnumpy(best_index), stage["indices"]):
                raise AssertionError(f"Hadamard index mismatch at stage {stage_index}")
            if not np.array_equal(
                cp.asnumpy(best_coefficient > 0.0), stage["positive"]
            ):
                raise AssertionError(f"coefficient sign mismatch at stage {stage_index}")
            if expected_amplitude != stage["amplitude"]:
                raise AssertionError(f"stage amplitude mismatch at stage {stage_index}")
            checkpoint_rows.append(
                {
                    "stage": stage_index,
                    "groups_verified": GROUPS,
                    "amplitude_fp16": stage["amplitude"],
                }
            )

        for group in range(GROUPS):
            bank = int(stage["banks"][group])
            index = int(stage["indices"][group])
            coefficient_sign = 1 if bool(stage["positive"][group]) else -1
            seed = stage_index * BANKS + bank
            signs = []
            for coordinate in probe_coordinates:
                diagonal = scalar_mix_sign(group * TRANSFORM_VALUES + coordinate, seed)
                hadamard = 1 if ((index & coordinate).bit_count() & 1) == 0 else -1
                signs.append(coefficient_sign * diagonal * hadamard)
            basis_signature.update(
                struct.pack("<IIBI", stage_index, group, bank, index)
                + bytes(1 if value > 0 else 0 for value in signs)
            )

        banks_gpu = cp.asarray(stage["banks"], dtype=cp.uint32)
        indices_gpu = cp.asarray(stage["indices"], dtype=cp.uint32)
        coefficient_signs = cp.where(
            cp.asarray(stage["positive"]), cp.float64(1.0), cp.float64(-1.0)
        )
        seeds = cp.uint32(stage_index * BANKS) + banks_gpu
        diagonal = mix_signs(linear, seeds)
        atom = hadamard_signs(indices_gpu, coordinates) * diagonal
        residual -= (
            coefficient_signs[:, None]
            * (stage["amplitude"] / math.sqrt(TRANSFORM_VALUES))
            * atom
        )

    corrected_sse = float(cp.asnumpy(cp.sum(residual * residual, dtype=cp.float64)))
    corrected = source - residual
    corrected_sha = sha_bytes(cp.asnumpy(corrected).astype("<f8", copy=False).tobytes())
    relative_mse = corrected_sse / energy
    rate = len(wrapper_payload) * 8.0 / PANEL_VALUES
    gaussian = 2.0 ** (-2.0 * rate)
    gap = 10.0 * math.log10(relative_mse / gaussian)
    independent_up = next(row for row in independent["roles"] if row["role"] == ROLE)
    close(corrected_sse, receipt_up["corrected_sse"])
    close(corrected_sse, independent_up["corrected_sse"])
    close(relative_mse, receipt_up["relative_mse"])
    close(gap, independent_up["gap_db"], 2e-12)
    if gap >= 0.0 or len(wrapper_payload) > 32_767_999:
        raise AssertionError("rate/gap target failed")

    result = {
        "format": "independent up-role SPARC4 audit v1",
        "status": "passed",
        "strict_ptq": True,
        "training_or_retraining": False,
        "calibration_or_activations": False,
        "command": sys.argv,
        "auditor_sha256": sha(Path(__file__)),
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "wrapper": {
            **wrapper,
            "bytes": len(wrapper_payload),
            "sha256": sha_bytes(wrapper_payload),
            "physical_rate_bpw": rate,
            "strict_headroom_bytes": 32_767_999 - len(wrapper_payload),
            "embedded_base_preserved_byte_for_byte": True,
        },
        "bindings": {
            "manifest_sha256": sha(args.manifest),
            "evaluation_sha256": sha(args.evaluation),
            "reconstruction_sha256": sha(args.reconstruction),
            "base_bundle_sha256": sha(args.base_bundle),
            "experiment_receipt_sha256": sha(args.experiment_receipt),
            "independent_receipt_sha256": sha(args.independent_receipt),
            "encoder_sha256": sha(args.encoder),
            "verifier_sha256": sha(args.verifier),
        },
        "up_role": {
            "blocks": ROLE_BLOCKS,
            "values": ROLE_VALUES,
            "source_energy": energy,
            "baseline_sse": base_sse,
            "coordinate_pulses": len(up["coordinate_positions"]),
            "coordinate_positions_sha256": sha_bytes(
                up["coordinate_positions"].astype("<i8").tobytes()
            ),
            "coordinate_signs_sha256": sha_bytes(up["coordinate_signs"].tobytes()),
            "coordinate_amplitude_fp16": up["coordinate_amplitude"],
            "coordinate_cutoff_margin": cutoff_margin,
            "canonical_untied_coordinate_support_verified": True,
            "sparc_stages": len(up["stages"]),
            "stage_codes_sha256": up["stage_codes_sha256"],
            "procedural_basis_probe_signature_sha256": basis_signature.hexdigest(),
            "matching_pursuit_checkpoints": checkpoint_rows,
            "all_transmitted_stages_decoded": True,
            "corrected_reconstruction_f64_sha256": corrected_sha,
            "corrected_sse": corrected_sse,
            "relative_mse": relative_mse,
            "gaussian_reference": gaussian,
            "signed_gap_db": gap,
            "passes_below_zero_db": True,
        },
        "verified_sources": verified_sources,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "up_role": result["up_role"]}, indent=2))


if __name__ == "__main__":
    main()
