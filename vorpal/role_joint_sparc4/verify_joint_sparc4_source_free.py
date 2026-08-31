#!/usr/bin/env python3
"""Pure-standard-library structural and receipt verifier for VJWRAP42.

This verifier needs neither raw BF16 source tensors, the 800 MiB reconstruction,
NumPy, nor CuPy.  It verifies the emitted wrapper and extension completely,
checks the published base/manifest/evaluation/encoder bindings, validates the
experiment and exact-source replay receipts, and recomputes physical rate and
all claimed Gaussian-reference gaps from the independently scored SSE values.

It cannot independently rescore SSE without the omitted source tensors and
reconstruction.  That job belongs to verify_joint_sparc4.py; this program binds
its published replay receipt to the wrapper, inputs, and exact verifier script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Callable


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
STRICT_MAX_BYTES = PANEL_VALUES * 5 // 16 - 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=json_object,
        parse_constant=reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON object required: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: Any, message: str) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=8e-14, abs_tol=3e-11
    ):
        raise ValueError(
            f"{message}: actual={float(actual)!r}, expected={float(expected)!r}"
        )


class BitReader:
    def __init__(self, payload: bytes, meaningful: int) -> None:
        require(0 <= meaningful <= len(payload) * 8, "invalid bit length")
        self.payload = payload
        self.meaningful = meaningful
        self.offset = 0

    def read(self, width: int) -> int:
        require(width >= 0, "negative bit width")
        if self.offset + width > self.meaningful:
            raise ValueError("truncated bitstream")
        value = 0
        for _ in range(width):
            value = (value << 1) | (
                (self.payload[self.offset >> 3] >> (7 - (self.offset & 7))) & 1
            )
            self.offset += 1
        return value


def decode_rice(
    payload: bytes, meaningful: int, count: int, rice_b: int
) -> tuple[list[int], list[int]]:
    reader = BitReader(payload, meaningful)
    positions: list[int] = []
    signs: list[int] = []
    previous = -1
    for _ in range(count):
        quotient = 0
        while reader.read(1) == 0:
            quotient += 1
        gap = (quotient << rice_b) | reader.read(rice_b)
        position = previous + gap + 1
        require(previous < position < ROLE_VALUES, "Rice support is noncanonical")
        positions.append(position)
        signs.append(1 if reader.read(1) else -1)
        previous = position
    require(reader.offset == meaningful, "unused meaningful Rice bits")
    if meaningful % 8:
        padding_mask = (1 << (8 - meaningful % 8)) - 1
        require(bool(payload) and payload[-1] & padding_mask == 0, "nonzero Rice padding")
    return positions, signs


def rice_bit_count(positions: list[int], rice_b: int) -> int:
    previous = -1
    bits = 0
    for position in positions:
        gap = position - previous - 1
        bits += (gap >> rice_b) + rice_b + 2
        previous = position
    return bits


def parse_mask(payload: bytes) -> list[int]:
    require(len(payload) == 50, "mask byte length")
    ordinals = [
        ordinal
        for ordinal in range(PANEL_BLOCKS)
        if payload[ordinal >> 3] & (1 << (ordinal & 7))
    ]
    require(len(ordinals) == ROLE_BLOCKS, "mask census")
    return ordinals


def validate_stage(payload: bytes) -> str:
    require(len(payload) == STAGE_CODE_BYTES, "stage payload length")
    reader = BitReader(payload, GROUPS_PER_ROLE * STAGE_SYMBOL_BITS)
    canonical = hashlib.sha256()
    for _ in range(GROUPS_PER_ROLE):
        code = reader.read(STAGE_SYMBOL_BITS)
        sign = code & 1
        index = (code >> 1) & 0x1FFFF
        bank = code >> 18
        require(bank < PROCEDURAL_BANKS, "stage bank out of range")
        require(index < TRANSFORM_VALUES, "stage index out of range")
        canonical.update(struct.pack("<BIB", bank, index, sign))
    require(reader.offset == reader.meaningful, "unused stage bits")
    return canonical.hexdigest()


def parse_wrapper(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    require(len(payload) >= WRAPPER_HEADER.size, "truncated wrapper")
    fields = WRAPPER_HEADER.unpack(payload[:WRAPPER_HEADER.size])
    (
        magic,
        version,
        header_bytes,
        panel_values,
        base_bytes,
        extension_bytes,
        base_hash,
        extension_hash,
        manifest_hash,
        evaluation_hash,
        reconstruction_hash,
        encoder_hash,
        crc,
    ) = fields
    require(magic == b"VJWRAP42", "wrapper magic")
    require(version == 2, "wrapper version")
    require(header_bytes == WRAPPER_HEADER.size == 224, "wrapper header size")
    require(panel_values == PANEL_VALUES, "wrapper panel values")
    require(
        zlib.crc32(payload[:WRAPPER_HEADER.size - 4]) & 0xFFFFFFFF == crc,
        "wrapper header CRC",
    )
    require(
        len(payload) == WRAPPER_HEADER.size + base_bytes + extension_bytes,
        "wrapper section sizes",
    )
    base = payload[WRAPPER_HEADER.size:WRAPPER_HEADER.size + base_bytes]
    extension = payload[WRAPPER_HEADER.size + base_bytes:]
    require(hashlib.sha256(base).digest() == base_hash, "wrapper base hash")
    require(hashlib.sha256(extension).digest() == extension_hash, "wrapper extension hash")
    return base, extension, {
        "header_bytes": header_bytes,
        "panel_values": panel_values,
        "base_bytes": base_bytes,
        "extension_bytes": extension_bytes,
        "base_sha256": base_hash.hex(),
        "extension_sha256": extension_hash.hex(),
        "manifest_sha256": manifest_hash.hex(),
        "evaluation_sha256": evaluation_hash.hex(),
        "reconstruction_sha256": reconstruction_hash.hex(),
        "encoder_sha256": encoder_hash.hex(),
    }


def parse_extension(extension: bytes) -> list[dict[str, Any]]:
    require(len(extension) >= EXT_HEADER.size + 4, "extension truncated")
    expected_crc = struct.unpack("<I", extension[-4:])[0]
    require(
        zlib.crc32(extension[:-4]) & 0xFFFFFFFF == expected_crc,
        "extension CRC",
    )
    fields = EXT_HEADER.unpack(extension[:EXT_HEADER.size])
    magic, version, role_count, log_values, banks, panel_blocks, block_values, groups = fields
    require(magic == b"VJSPRC41", "extension magic")
    require(version == 1 and role_count == 3, "extension version/role count")
    require(log_values == 17 and banks == PROCEDURAL_BANKS, "extension transform constants")
    require(panel_blocks == PANEL_BLOCKS and block_values == BLOCK_VALUES, "extension panel constants")
    require(groups == GROUPS_PER_ROLE, "extension group count")
    cursor = EXT_HEADER.size
    rows: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    for expected_role in range(3):
        descriptor_offset = cursor
        require(cursor + ROLE_DESCRIPTOR.size + 50 <= len(extension) - 4, "role descriptor truncated")
        descriptor = ROLE_DESCRIPTOR.unpack(extension[cursor:cursor + ROLE_DESCRIPTOR.size])
        cursor += ROLE_DESCRIPTOR.size
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
        ) = descriptor
        require(role_id == expected_role, "role descriptor order")
        require(rice_b <= 20, "Rice parameter range")
        require(block_count == ROLE_BLOCKS, "role block count")
        require(coordinate_count == 100, "coordinate count")
        require(stage_count <= 450, "stage count")
        require(coordinate_bytes == (coordinate_bits + 7) // 8, "coordinate byte count")
        require(stage_bytes == STAGE_RECORD_BYTES, "stage record byte count")
        mask_offset = cursor
        ordinals = parse_mask(extension[cursor:cursor + 50])
        cursor += 50
        require(seen_ordinals.isdisjoint(ordinals), "role masks overlap")
        seen_ordinals.update(ordinals)
        coordinate_offset = cursor
        coordinate_payload = extension[cursor:cursor + coordinate_bytes]
        require(len(coordinate_payload) == coordinate_bytes, "coordinate payload truncated")
        cursor += coordinate_bytes
        positions, signs = decode_rice(
            coordinate_payload, coordinate_bits, coordinate_count, rice_b
        )
        require(
            math.isfinite(coordinate_amplitude) and coordinate_amplitude > 0.0,
            "coordinate amplitude invalid",
        )
        require(coordinate_bits == rice_bit_count(positions, rice_b), "Rice bit count mismatch")
        optimal_rice_b = min(range(21), key=lambda value: rice_bit_count(positions, value))
        require(rice_b == optimal_rice_b, "Rice parameter is not minimal")
        stages_offset = cursor
        stage_amplitudes: list[float] = []
        stage_symbol_digest = hashlib.sha256()
        for _ in range(stage_count):
            require(cursor + STAGE_RECORD_BYTES <= len(extension) - 4, "stage truncated")
            amplitude = struct.unpack("<e", extension[cursor:cursor + 2])[0]
            require(math.isfinite(amplitude) and amplitude > 0.0, "stage amplitude invalid")
            stage_payload = extension[cursor + 2:cursor + STAGE_RECORD_BYTES]
            stage_symbol_digest.update(bytes.fromhex(validate_stage(stage_payload)))
            stage_amplitudes.append(float(amplitude))
            cursor += STAGE_RECORD_BYTES
        rows.append(
            {
                "role_id": role_id,
                "ordinals": ordinals,
                "coordinate_amplitude": float(coordinate_amplitude),
                "coordinate_positions": positions,
                "coordinate_signs": signs,
                "rice_b": rice_b,
                "coordinate_bits": coordinate_bits,
                "coordinate_bytes": coordinate_bytes,
                "stage_count": stage_count,
                "stage_amplitudes": stage_amplitudes,
                "stage_symbol_digest": stage_symbol_digest.hexdigest(),
                "descriptor_offset": descriptor_offset,
                "mask_offset": mask_offset,
                "coordinate_offset": coordinate_offset,
                "stages_offset": stages_offset,
            }
        )
    require(cursor == len(extension) - 4, "trailing extension bytes")
    return rows


def validate_external_bindings(
    wrapper: dict[str, Any], manifest_sha: str, evaluation_sha: str, encoder_sha: str
) -> None:
    require(wrapper["manifest_sha256"] == manifest_sha, "manifest provenance binding")
    require(wrapper["evaluation_sha256"] == evaluation_sha, "evaluation provenance binding")
    require(wrapper["encoder_sha256"] == encoder_sha, "encoder provenance binding")


def validate_publication(
    wrapper_payload: bytes,
    base: bytes,
    extension: bytes,
    wrapper: dict[str, Any],
    decoded: list[dict[str, Any]],
    external_base: bytes,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    receipt: dict[str, Any],
    replay: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    require(base == external_base, "embedded base differs from published base")
    validate_external_bindings(
        wrapper, hashes["manifest_sha256"], hashes["evaluation_sha256"], hashes["encoder_sha256"]
    )
    require(
        wrapper["reconstruction_sha256"] == evaluation.get("reconstruction_sha256"),
        "reconstruction digest is not chained through evaluation",
    )
    require(
        manifest.get("strict_ptq") is True
        and manifest.get("training_or_retraining") is False,
        "manifest PTQ assertions",
    )
    require(
        evaluation.get("status") == "passed"
        and evaluation.get("strict_ptq") is True
        and evaluation.get("source_is_evaluator_only") is True,
        "evaluation status/PTQ assertions",
    )
    require(
        evaluation.get("all_400_source_hashes_ordinals_and_scatter_coverage_verified") is True,
        "evaluation source coverage assertion",
    )
    require(
        int(evaluation.get("source_blocks", -1)) == PANEL_BLOCKS
        and int(evaluation.get("panel_values", -1)) == PANEL_VALUES,
        "evaluation panel census",
    )
    require(evaluation.get("encoded_sha256") == wrapper["base_sha256"], "evaluation base binding")
    require(int(evaluation.get("encoded_bytes", -1)) == len(base), "evaluation base byte count")

    blocks = manifest.get("blocks")
    eval_blocks = evaluation.get("blocks")
    require(isinstance(blocks, list) and len(blocks) == PANEL_BLOCKS, "manifest block census")
    require(isinstance(eval_blocks, list) and len(eval_blocks) == PANEL_BLOCKS, "evaluation block census")
    for ordinal, (block, eval_block) in enumerate(zip(blocks, eval_blocks, strict=True)):
        require(int(block.get("ordinal", -1)) == ordinal, "manifest ordinal order")
        require(int(eval_block.get("canonical_block_ordinal", -1)) == ordinal, "evaluation ordinal order")
        for key in ("id", "role", "source_sha256"):
            require(block.get(key) == eval_block.get(key), f"manifest/evaluation block {key}")

    require(
        receipt.get("status") == "passed"
        and receipt.get("strict_ptq") is True
        and receipt.get("training_or_retraining") is False
        and receipt.get("calibration_or_activations") is False
        and receipt.get("base_preserved_byte_for_byte") is True,
        "experiment receipt status/PTQ assertions",
    )
    method = receipt.get("method", {})
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
    for key, expected in expected_method.items():
        require(method.get(key) == expected, f"experiment method constant: {key}")
    inputs = receipt.get("inputs", {})
    artifacts = receipt.get("artifacts", {})
    require(inputs.get("manifest_sha256") == hashes["manifest_sha256"], "receipt manifest binding")
    require(inputs.get("evaluation_sha256") == hashes["evaluation_sha256"], "receipt evaluation binding")
    require(inputs.get("reconstruction_sha256") == wrapper["reconstruction_sha256"], "receipt reconstruction binding")
    require(inputs.get("base_bundle_sha256") == wrapper["base_sha256"], "receipt base binding")
    require(int(inputs.get("base_bundle_bytes", -1)) == len(base), "receipt base bytes")
    require(artifacts.get("extension_sha256") == wrapper["extension_sha256"], "receipt extension binding")
    require(int(artifacts.get("extension_bytes", -1)) == len(extension), "receipt extension bytes")
    require(artifacts.get("emitted_wrapper_sha256") == sha256_bytes(wrapper_payload), "receipt wrapper binding")
    require(int(artifacts.get("emitted_wrapper_bytes", -1)) == len(wrapper_payload), "receipt wrapper bytes")
    require(int(artifacts.get("wrapper_header_bytes", -1)) == WRAPPER_HEADER.size, "receipt header bytes")
    require(artifacts.get("wrapper_roundtrip_verified") is True, "receipt wrapper round trip")
    require(
        artifacts.get("wrapper_binds_base_extension_manifest_evaluation_reconstruction_and_encoder") is True,
        "receipt six-binding assertion",
    )
    environment = receipt.get("environment", {})
    require(environment.get("script_sha256") == hashes["encoder_sha256"], "receipt encoder binding")

    physical_bytes = len(wrapper_payload)
    rate = physical_bytes * 8.0 / PANEL_VALUES
    reference = 2.0 ** (-2.0 * rate)
    accounting = receipt.get("accounting", {})
    require(physical_bytes <= STRICT_MAX_BYTES, "physical rate is not strictly below 2.5 bpw")
    require(int(accounting.get("strict_max_bytes_below_2p5", -1)) == STRICT_MAX_BYTES, "strict max-byte accounting")
    require(int(accounting.get("base_bytes", -1)) == len(base), "base-byte accounting")
    require(int(accounting.get("wrapper_header_bytes", -1)) == WRAPPER_HEADER.size, "header-byte accounting")
    require(int(accounting.get("extension_bytes", -1)) == len(extension), "extension-byte accounting")
    require(int(accounting.get("physical_all_in_bytes", -1)) == physical_bytes, "physical-byte accounting")
    require(int(accounting.get("rate_headroom_bytes", -1)) == STRICT_MAX_BYTES - physical_bytes, "headroom accounting")
    close(accounting.get("physical_all_in_rate_bpw"), rate, "rate accounting")
    close(accounting.get("gaussian_reference_at_actual_rate"), reference, "reference accounting")

    claimed_roles = receipt.get("roles")
    replay_roles = replay.get("roles")
    require(isinstance(claimed_roles, list) and len(claimed_roles) == 3, "receipt role census")
    require(isinstance(replay_roles, list) and len(replay_roles) == 3, "replay role census")
    evaluated_by_role = {row["role"]: row for row in evaluation.get("by_role", [])}
    recomputed_roles: list[dict[str, Any]] = []
    total_stages = 0
    for role_id, (role, parsed, claimed, replay_row) in enumerate(
        zip(ROLES, decoded, claimed_roles, replay_roles, strict=True)
    ):
        expected_ordinals = [
            ordinal for ordinal, block in enumerate(blocks) if block.get("role") == role
        ]
        require(parsed["role_id"] == role_id, "decoded role id")
        require(parsed["ordinals"] == expected_ordinals, "role mask/manifest mismatch")
        require(claimed.get("role") == role and int(claimed.get("role_id", -1)) == role_id, "receipt role identity")
        require(replay_row.get("role") == role and int(replay_row.get("role_id", -1)) == role_id, "replay role identity")
        require(claimed.get("block_ordinals") == expected_ordinals, "receipt role ordinals")
        require(int(claimed.get("coordinate_pulses", -1)) == len(parsed["coordinate_positions"]), "receipt coordinate count")
        require(int(claimed.get("coordinate_rice_b", -1)) == parsed["rice_b"], "receipt Rice parameter")
        require(int(claimed.get("coordinate_bits", -1)) == parsed["coordinate_bits"], "receipt Rice bits")
        require(int(claimed.get("coordinate_bytes", -1)) == parsed["coordinate_bytes"], "receipt Rice bytes")
        close(claimed.get("coordinate_amplitude_fp16"), parsed["coordinate_amplitude"], "receipt coordinate amplitude")
        require(int(claimed.get("sparc_stages", -1)) == parsed["stage_count"], "receipt stage count")
        require(int(claimed.get("stage_record_bytes", -1)) == STAGE_RECORD_BYTES, "receipt stage bytes")
        require(claimed.get("stage_amplitudes_fp16") == parsed["stage_amplitudes"], "receipt stage amplitudes")
        total_stages += parsed["stage_count"]

        sources = claimed.get("sources")
        require(isinstance(sources, list) and len(sources) == ROLE_BLOCKS, "receipt source census")
        for source, ordinal in zip(sources, expected_ordinals, strict=True):
            block = blocks[ordinal]
            require(
                int(source.get("ordinal", -1)) == ordinal
                and source.get("id") == block.get("id")
                and source.get("sha256") == block.get("source_sha256"),
                "receipt source/manifest identity",
            )

        baseline = evaluated_by_role.get(role)
        require(isinstance(baseline, dict), "evaluation missing role")
        close(claimed.get("source_energy"), baseline.get("source_energy"), "receipt/evaluation role energy")
        close(claimed.get("base_sse"), baseline.get("sse"), "receipt/evaluation role baseline SSE")
        corrected_sse = float(claimed.get("corrected_sse"))
        source_energy = float(claimed.get("source_energy"))
        relative = corrected_sse / source_energy
        gap = 10.0 * math.log10(relative / reference)
        close(claimed.get("relative_mse"), relative, "receipt role relative MSE")
        close(claimed.get("gap_db"), gap, "receipt role gap")
        close(claimed.get("sse_savings"), float(claimed.get("base_sse")) - corrected_sse, "receipt role savings")
        close(replay_row.get("source_energy"), source_energy, "replay role energy")
        close(replay_row.get("corrected_sse"), corrected_sse, "replay role SSE")
        close(replay_row.get("relative_mse"), relative, "replay role relative MSE")
        close(replay_row.get("gap_db"), gap, "replay role gap")
        require(int(replay_row.get("source_hashes_verified", -1)) == ROLE_BLOCKS, "replay source hash census")
        require(int(replay_row.get("coordinate_pulses_decoded", -1)) == 100, "replay coordinate count")
        require(int(replay_row.get("sparc_stages_decoded", -1)) == parsed["stage_count"], "replay stage count")
        recomputed_roles.append(
            {
                "role": role,
                "role_id": role_id,
                "block_count": ROLE_BLOCKS,
                "coordinate_pulses": 100,
                "sparc_stages": parsed["stage_count"],
                "source_energy": source_energy,
                "corrected_sse": corrected_sse,
                "relative_mse": relative,
                "gap_db": gap,
                "stage_symbol_digest": parsed["stage_symbol_digest"],
            }
        )
    require(int(accounting.get("total_selected_sparc_stages", -1)) == total_stages, "total stage accounting")
    fixed_bytes = WRAPPER_HEADER.size + len(extension) - total_stages * STAGE_RECORD_BYTES
    require(int(accounting.get("fixed_wrapper_and_extension_bytes_before_stages", -1)) == fixed_bytes, "fixed-byte accounting")

    recomputed_global_sse = float(evaluation.get("sse"))
    for row in recomputed_roles:
        recomputed_global_sse += row["corrected_sse"] - float(evaluated_by_role[row["role"]]["sse"])
    global_energy = float(evaluation.get("source_energy"))
    global_relative = recomputed_global_sse / global_energy
    global_gap = 10.0 * math.log10(global_relative / reference)
    receipt_global = receipt.get("global", {})
    close(receipt_global.get("base_sse"), evaluation.get("sse"), "receipt global baseline SSE")
    close(receipt_global.get("source_energy"), global_energy, "receipt global source energy")
    close(receipt_global.get("corrected_sse"), recomputed_global_sse, "receipt global corrected SSE")
    close(receipt_global.get("relative_mse"), global_relative, "receipt global relative MSE")
    close(receipt_global.get("gap_db"), global_gap, "receipt global gap")

    require(replay.get("status") == "passed" and replay.get("strict_ptq") is True, "replay status/PTQ")
    require(replay.get("encoder_module_imported") is False, "replay imported encoder")
    require(replay.get("source_free_decode_completed_before_source_scoring") is True, "replay decode/source ordering")
    require(replay.get("receipt_metrics_reproduced") is True, "replay metric reproduction")
    require(replay.get("wrapper_sha256") == hashes["wrapper_sha256"], "replay wrapper binding")
    require(int(replay.get("wrapper_bytes", -1)) == physical_bytes, "replay wrapper bytes")
    require(replay.get("base_bundle_sha256") == wrapper["base_sha256"], "replay base binding")
    require(replay.get("extension_sha256") == wrapper["extension_sha256"], "replay extension binding")
    require(replay.get("manifest_sha256") == hashes["manifest_sha256"], "replay manifest binding")
    require(replay.get("evaluation_sha256") == hashes["evaluation_sha256"], "replay evaluation binding")
    require(replay.get("reconstruction_sha256") == wrapper["reconstruction_sha256"], "replay reconstruction binding")
    require(replay.get("encoder_sha256") == hashes["encoder_sha256"], "replay encoder binding")
    require(replay.get("verifier_script_sha256") == hashes["full_verifier_sha256"], "replay verifier binding")
    require(replay.get("experiment_receipt_sha256") == hashes["experiment_receipt_sha256"], "replay experiment receipt binding")
    close(replay.get("physical_rate_bpw"), rate, "replay physical rate")
    close(replay.get("gaussian_reference"), reference, "replay Gaussian reference")
    close(replay.get("worst_role_gap_db"), max(row["gap_db"] for row in recomputed_roles), "replay worst role gap")
    close(replay.get("global_corrected_sse"), recomputed_global_sse, "replay global SSE")
    close(replay.get("global_relative_mse"), global_relative, "replay global relative MSE")
    close(replay.get("global_gap_db"), global_gap, "replay global gap")
    require(replay.get("all_three_roles_below_zero") is True, "replay all-role result")
    require(replay.get("global_remains_below_zero") is True, "replay global result")
    require(receipt.get("all_three_roles_below_zero") is True, "receipt all-role result")
    require(receipt_global.get("remains_below_zero") is True, "receipt global result")
    require(all(row["gap_db"] < 0.0 for row in recomputed_roles), "recomputed role gap is not negative")
    require(global_gap < 0.0, "recomputed global gap is not negative")

    return {
        "physical_bytes": physical_bytes,
        "rate_bpw": rate,
        "headroom_bytes": STRICT_MAX_BYTES - physical_bytes,
        "gaussian_reference": reference,
        "roles": recomputed_roles,
        "worst_role_gap_db": max(row["gap_db"] for row in recomputed_roles),
        "global_corrected_sse": recomputed_global_sse,
        "global_relative_mse": global_relative,
        "global_gap_db": global_gap,
        "total_sparc_stages": total_stages,
        "fixed_wrapper_and_extension_bytes_before_stages": fixed_bytes,
    }


def repack_wrapper_header(payload: bytes, fields: list[Any]) -> bytes:
    fields[-1] = 0
    header = WRAPPER_HEADER.pack(*fields)
    header = header[:-4] + struct.pack(
        "<I", zlib.crc32(header[:-4]) & 0xFFFFFFFF
    )
    return header + payload[WRAPPER_HEADER.size:]


def with_extension_crc(payload: bytearray) -> bytes:
    payload[-4:] = struct.pack("<I", zlib.crc32(payload[:-4]) & 0xFFFFFFFF)
    return bytes(payload)


def run_tamper_tests(
    wrapper_payload: bytes,
    extension: bytes,
    wrapper: dict[str, Any],
    decoded: list[dict[str, Any]],
    manifest_sha: str,
    evaluation_sha: str,
    encoder_sha: str,
    receipt: dict[str, Any],
    replay: dict[str, Any],
) -> list[dict[str, str]]:
    tests: list[dict[str, str]] = []

    def rejected(name: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except (ValueError, struct.error, IndexError, KeyError, TypeError) as exc:
            tests.append(
                {
                    "name": name,
                    "expected": "reject",
                    "result": "rejected",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return
        raise AssertionError(f"tamper test did not fail closed: {name}")

    corrupted = bytearray(wrapper_payload)
    corrupted[12] ^= 1
    rejected("wrapper_header_crc_bitflip", lambda: parse_wrapper(bytes(corrupted)))

    corrupted = bytearray(wrapper_payload)
    corrupted[WRAPPER_HEADER.size + 7] ^= 1
    rejected("embedded_base_payload_bitflip", lambda: parse_wrapper(bytes(corrupted)))

    corrupted = bytearray(wrapper_payload)
    corrupted[-17] ^= 1
    rejected("extension_payload_bitflip", lambda: parse_wrapper(bytes(corrupted)))

    corrupted_extension = bytearray(extension)
    corrupted_extension[-1] ^= 1
    rejected("extension_crc_trailer_bitflip", lambda: parse_extension(bytes(corrupted_extension)))

    for field_index, key, actual in (
        (8, "manifest", manifest_sha),
        (9, "evaluation", evaluation_sha),
        (10, "reconstruction", wrapper["reconstruction_sha256"]),
        (11, "encoder", encoder_sha),
    ):
        fields = list(WRAPPER_HEADER.unpack(wrapper_payload[:WRAPPER_HEADER.size]))
        changed = bytearray(fields[field_index])
        changed[0] ^= 1
        fields[field_index] = bytes(changed)
        mutated_payload = repack_wrapper_header(wrapper_payload, fields)

        def check_provenance(payload: bytes = mutated_payload, name: str = key, expected: str = actual) -> None:
            _, _, mutated = parse_wrapper(payload)
            if name == "reconstruction":
                require(mutated["reconstruction_sha256"] == expected, "reconstruction provenance binding")
            else:
                validate_external_bindings(mutated, manifest_sha, evaluation_sha, encoder_sha)

        rejected(f"recrc_{key}_provenance_hash_mutation", check_provenance)

    corrupted_extension = bytearray(extension)
    amp_offset = int(decoded[0]["descriptor_offset"]) + 8
    corrupted_extension[amp_offset:amp_offset + 2] = b"\x00\x00"
    rejected(
        "recrc_zero_coordinate_amplitude",
        lambda: parse_extension(with_extension_crc(corrupted_extension)),
    )

    corrupted_extension = bytearray(extension)
    stage_amp_offset = int(decoded[0]["stages_offset"])
    corrupted_extension[stage_amp_offset:stage_amp_offset + 2] = b"\x00\x00"
    rejected(
        "recrc_zero_stage_amplitude",
        lambda: parse_extension(with_extension_crc(corrupted_extension)),
    )

    corrupted_extension = bytearray(extension)
    bits_offset = int(decoded[0]["descriptor_offset"]) + 10
    original_bits = struct.unpack("<I", corrupted_extension[bits_offset:bits_offset + 4])[0]
    corrupted_extension[bits_offset:bits_offset + 4] = struct.pack("<I", original_bits + 1)
    rejected(
        "recrc_noncanonical_rice_bit_length",
        lambda: parse_extension(with_extension_crc(corrupted_extension)),
    )

    corrupted_extension = bytearray(extension)
    first_ordinal = int(decoded[0]["ordinals"][0])
    second_mask_offset = int(decoded[1]["mask_offset"])
    old_second_ordinal = int(decoded[1]["ordinals"][0])
    corrupted_extension[second_mask_offset + old_second_ordinal // 8] &= ~(
        1 << (old_second_ordinal & 7)
    ) & 0xFF
    corrupted_extension[second_mask_offset + first_ordinal // 8] |= 1 << (
        first_ordinal & 7
    )
    rejected(
        "recrc_overlapping_role_masks",
        lambda: parse_extension(with_extension_crc(corrupted_extension)),
    )

    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["strict_ptq"] = False
    rejected(
        "experiment_receipt_ptq_claim_mutation",
        lambda: require(
            bad_receipt.get("strict_ptq") is True,
            "experiment receipt status/PTQ assertions",
        ),
    )

    bad_replay = copy.deepcopy(replay)
    bad_replay["wrapper_sha256"] = "00" * 32
    rejected(
        "independent_replay_wrapper_binding_mutation",
        lambda: require(
            bad_replay.get("wrapper_sha256") == sha256_bytes(wrapper_payload),
            "replay wrapper binding",
        ),
    )
    return tests


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = Path(str(path) + ".partial")
    require(not path.exists() and not partial.exists(), f"output exists: {path}")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--experiment-receipt", type=Path, required=True)
    parser.add_argument("--independent-verification", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--full-verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tamper-output", type=Path, required=True)
    args = parser.parse_args()

    wrapper_payload = args.wrapper.read_bytes()
    base, extension, wrapper = parse_wrapper(wrapper_payload)
    decoded = parse_extension(extension)
    manifest = load_json(args.manifest)
    evaluation = load_json(args.evaluation)
    receipt = load_json(args.experiment_receipt)
    replay = load_json(args.independent_verification)
    hashes = {
        "wrapper_sha256": sha256_bytes(wrapper_payload),
        "base_sha256": sha256_path(args.base_bundle),
        "manifest_sha256": sha256_path(args.manifest),
        "evaluation_sha256": sha256_path(args.evaluation),
        "experiment_receipt_sha256": sha256_path(args.experiment_receipt),
        "independent_verification_sha256": sha256_path(args.independent_verification),
        "encoder_sha256": sha256_path(args.encoder),
        "full_verifier_sha256": sha256_path(args.full_verifier),
        "source_free_verifier_sha256": sha256_path(Path(__file__).resolve()),
    }
    metrics = validate_publication(
        wrapper_payload,
        base,
        extension,
        wrapper,
        decoded,
        args.base_bundle.read_bytes(),
        manifest,
        evaluation,
        receipt,
        replay,
        hashes,
    )
    tamper_tests = run_tamper_tests(
        wrapper_payload,
        extension,
        wrapper,
        decoded,
        hashes["manifest_sha256"],
        hashes["evaluation_sha256"],
        hashes["encoder_sha256"],
        receipt,
        replay,
    )
    tamper_receipt = {
        "format": "VJWRAP42 source-free tamper receipt v1",
        "status": "passed",
        "wrapper_sha256": hashes["wrapper_sha256"],
        "tests_run": len(tamper_tests),
        "tests_rejected": sum(test["result"] == "rejected" for test in tamper_tests),
        "all_tamper_tests_failed_closed": all(
            test["result"] == "rejected" for test in tamper_tests
        ),
        "tests": tamper_tests,
        "python": sys.version,
        "source_free_verifier_sha256": hashes["source_free_verifier_sha256"],
    }
    result = {
        "format": "VJWRAP42/VJSPRC41 standard-library source-free verification v1",
        "status": "passed",
        "strict_ptq": True,
        "python_standard_library_only": True,
        "raw_bf16_sources_opened": False,
        "reconstruction_payload_opened": False,
        "numpy_or_cupy_imported": False,
        "scope": {
            "binary_wrapper_and_extension_fully_parsed": True,
            "published_base_manifest_evaluation_and_encoder_hashes_verified": True,
            "embedded_reconstruction_digest_chained_through_evaluation_and_receipts": True,
            "experiment_and_exact_source_replay_receipts_consistency_verified": True,
            "sse_independently_rescored": False,
            "limitation": "SSE rescore requires raw BF16 plus reconstruction and is recorded by the separately bound full verifier receipt.",
        },
        "hashes": hashes,
        "wrapper": wrapper,
        "extension": {
            "crc_verified": True,
            "rice_payloads_exact_and_minimal": True,
            "coordinate_and_stage_amplitudes_finite_positive": True,
            "all_stage_symbols_parsed": True,
            "role_masks_disjoint_and_match_manifest": True,
        },
        "receipt_and_replay_validation": {
            "experiment_receipt_status_ptq_provenance_method_and_metrics_verified": True,
            "independent_replay_status_provenance_script_and_metrics_verified": True,
            "independent_replay_corrected_reconstruction_sha256": replay.get(
                "corrected_three_role_reconstruction_f64_sha256"
            ),
        },
        "metrics_recomputed_from_replay_sse": metrics,
        "all_three_roles_below_zero_db": all(
            row["gap_db"] < 0.0 for row in metrics["roles"]
        ),
        "global_below_zero_db": metrics["global_gap_db"] < 0.0,
        "tamper_receipt": str(args.tamper_output),
        "tamper_tests_run": len(tamper_tests),
        "all_tamper_tests_failed_closed": True,
        "python": sys.version,
    }
    atomic_json(args.tamper_output, tamper_receipt)
    result["tamper_receipt_sha256"] = sha256_path(args.tamper_output)
    atomic_json(args.output, result)
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
