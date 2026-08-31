#!/usr/bin/env python3
"""End-to-end synthetic and tamper tests for the VORPAL verifier."""

from __future__ import annotations

import bz2
import copy
import hashlib
import json
import lzma
import math
import shutil
import struct
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

import verify_vorpal_publication as v


def digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def role_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, count in v.GLOBAL_ROLE_COUNTS.items():
        for block_index in range(count):
            rows.append({"role": role, "layer": None, "block_index": block_index})
    for role in v.LAYERWISE_ROLES:
        for layer in range(48):
            rows.append({"role": role, "layer": layer, "block_index": layer})
    assert len(rows) == v.CHUNK_COUNT
    return rows


def build_side(selected_codes: list[int]) -> tuple[bytes, bytes, bytes, bytes]:
    labels = bytes((v.LABEL_COUNT * v.LABEL_BITS + 7) // 8)
    header = v.SIDE_HEADER.pack(
        v.SIDE_MAGIC,
        v.WIRE_VERSION,
        v.CHUNK_COUNT,
        v.GROUPS_PER_CHUNK,
        v.GROUP_VALUES,
        v.CHUNK_COUNT,
        v.LABEL_COUNT,
        0.001,
        64,
    )
    prefix = header + struct.pack("<64d", *([1.0] * 64)) + struct.pack("<400f", *([1.0] * 400)) + labels
    canonical = prefix + b"".join(v.PROFILE.pack(0.5, 1.0, 0) for _ in range(v.CHUNK_COUNT))
    literal = bytearray(canonical)
    route = bytearray(v.ROUTE_BYTES)
    profile_offset = len(prefix)
    for index, code in enumerate(selected_codes):
        literal[profile_offset + index * v.PROFILE.size + 16] = code
        if code:
            route[index >> 3] |= 1 << (index & 7)
    canonical_xz = lzma.compress(canonical, format=lzma.FORMAT_XZ, preset=9)
    return bytes(literal), canonical, canonical_xz, bytes(route)


def make_frame(logical_bits: int = 0, payload: bytes = b"") -> bytes:
    assert len(payload) == (logical_bits + 7) // 8
    return struct.pack("<If", logical_bits, 1.0) + payload


def strata(blocks: list[dict[str, object]], fields: tuple[str, ...], bpw: float, reference: float) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in blocks:
        buckets.setdefault(tuple(row[field] for field in fields), []).append(row)
    result = []
    for key in sorted(buckets, key=lambda item: tuple("<global>" if value is None else str(value) for value in item)):
        members = buckets[key]
        energy = 0.0
        sse = 0.0
        for row in members:
            energy += float(row["source_energy"])
            sse += float(row["sse"])
        relative = sse / energy
        output = {field: value for field, value in zip(fields, key, strict=True)}
        output.update(
            blocks=len(members),
            values=len(members) * v.BLOCK_VALUES,
            source_energy=energy,
            sse=sse,
            relative_mse=relative,
            effective_charged_panel_bpw=bpw,
            gaussian_reference_mse_at_effective_panel_rate=reference,
            diagnostic_gap_db_at_effective_panel_rate=10.0 * math.log10(relative / reference),
            ids=[str(row["id"]) for row in members],
        )
        result.append(output)
    return result


def build_fixture(root: Path, *, relative_mse: float = 0.9) -> None:
    containers = root / "containers"
    containers.mkdir(parents=True)
    base_frame = make_frame()
    selected_frame = make_frame(8, b"\0")
    base_hash = digest(base_frame)
    selected_hash = digest(selected_frame)

    selected_codes = [1] + [0] * (v.CHUNK_COUNT - 1)
    literal_side, canonical_side, canonical_xz, route = build_side(selected_codes)
    (root / "side.bin").write_bytes(literal_side)
    frames = [selected_frame] + [base_frame] * (v.CHUNK_COUNT - 1)
    for index, frame in enumerate(frames):
        (containers / f"wf-{index:03d}.polar.bin").write_bytes(frame)

    raw_mask = bytes(v.MASK_RAW_BYTES)
    compressed_mask = bz2.compress(raw_mask, compresslevel=9)
    payload = canonical_xz + route
    header = v.OUTER_HEADER.pack(
        v.OUTER_MAGIC,
        v.WIRE_VERSION,
        v.OUTER_HEADER.size,
        v.SIDE_CODEC_FIXED_ROUTE,
        len(literal_side),
        len(payload),
        v.MASK_CODEC_BZ2,
        len(raw_mask),
        len(compressed_mask),
        bytes.fromhex(digest(literal_side)),
        bytes.fromhex(digest(payload)),
        bytes.fromhex(digest(raw_mask)),
        bytes.fromhex(digest(compressed_mask)),
    )
    bundle = header + payload + compressed_mask + b"".join(frames)
    (root / "selected.wfouter").write_bytes(bundle)
    physical_prelude = len(header) + len(payload) + len(compressed_mask)
    bundle_bytes = len(bundle)
    bpw = bundle_bytes * 8.0 / v.PANEL_VALUES
    reference = 2.0 ** (-2.0 * bpw)
    signed_gap = 10.0 * math.log10(relative_mse / reference)

    identity_rows = role_rows()
    blocks = []
    chunks = []
    for index, identity in enumerate(identity_rows):
        block_id = f"synthetic-{index:03d}"
        tensor = f"layers.{identity['layer']}.{identity['role']}" if identity["layer"] is not None else str(identity["role"])
        blocks.append(
            {
                "ordinal": index,
                "block_index": identity["block_index"],
                "id": block_id,
                "tensor": tensor,
                "role": identity["role"],
                "source_path": f"private/sources/{block_id}.bf16.bin",
                "source_sha256": digest(f"source-{index}"),
                "labels_i8": [0],
                "serialized_rms_fp32": 1.0,
            }
        )
        chunks.append(
            {
                "chunk_index": index,
                "alphabet_size": 64,
                "eta": 1.0,
                "test_distortion": 0.5,
                "normalized_source": f"private/normalized_sources/wf-{index:03d}.bf16.bin",
                "normalized_source_sha256": digest(f"normalized-{index}"),
                "nominal_rate": 2.0,
                "unconstrained_rate": 2.0,
                "mean_qvariance": 1.0,
                "members": [index],
            }
        )
    construction = {
        "format": "continuous reverse-waterfilled PLTE exploratory manifest v1",
        "strict_ptq": True,
        "training_or_retraining": False,
        "parameters": {
            "block_values": v.BLOCK_VALUES,
            "group_values": v.GROUP_VALUES,
            "groups_per_polar_block": v.GROUPS_PER_CHUNK,
            "sigma_source": 3.0,
        },
        "census": {
            "source_blocks": v.CHUNK_COUNT,
            "groups": v.LABEL_COUNT,
            "polar_chunks": v.CHUNK_COUNT,
            "values": v.PANEL_VALUES,
        },
        "blocks": blocks,
        "chunks": chunks,
        "side_ledger": {},
        "ideal_projection": {},
    }
    dump(root / "construction.manifest.json", construction)
    selected_manifest = copy.deepcopy(construction)
    selected_manifest["chunks"][0]["alphabet_size"] = 128
    dump(root / "selected.manifest.json", selected_manifest)

    ledger_rows = []
    for index, (identity, block) in enumerate(zip(identity_rows, blocks, strict=True)):
        ledger_rows.append(
            {
                "canonical_block_ordinal": index,
                "id": block["id"],
                "tensor": block["tensor"],
                "role": identity["role"],
                "layer": identity["layer"],
                "block_index": identity["block_index"],
                "path": block["source_path"],
                "sha256": block["source_sha256"],
            }
        )
    ledger = {
        "format": "canonical BF16 source ledger v1",
        "evaluator_only": True,
        "checkpoint": v.CHECKPOINT,
        "selection_manifest_sha256": digest("upstream-selection-manifest"),
        "construction_manifest_sha256": v.sha256_path(root / "construction.manifest.json"),
        "blocks": ledger_rows,
    }
    dump(root / "source_ledger.json", ledger)

    base_receipt = {
        "format": "continuous reverse-waterfilled PLTE full panel encode receipt v1",
        "status": "complete",
        "all_internal_roundtrips_passed": True,
        "chunks": v.CHUNK_COUNT,
        "manifest_sha256": v.sha256_path(root / "construction.manifest.json"),
        "encoder_sha256": digest("encoder"),
        "failures": [],
        "actual_container_bytes": len(base_frame) * v.CHUNK_COUNT,
        "rows": [
            {"chunk_index": index, "status": "resumed", "container_bytes": len(base_frame), "relative_mse": 0.1}
            for index in range(v.CHUNK_COUNT)
        ],
    }
    dump(root / "base.run.receipt.json", base_receipt)

    tail_frame_hash = digest("synthetic-tail-k1")
    candidate = {
        "format": "continuous PLTE all-base adaptive candidate receipt v3",
        "status": "complete",
        "strict_ptq": True,
        "training_or_retraining": False,
        "manifest_sha256": v.sha256_path(root / "construction.manifest.json"),
        "base_receipt_sha256": v.sha256_path(root / "base.run.receipt.json"),
        "base_receipt_status": "complete",
        "raw_mask_sha256": digest(raw_mask),
        "implementation_sha256": digest("candidate-wrapper"),
        "pinned_runner_core_sha256": digest("runner-core"),
        "pinned_repacker_core_sha256": digest("repacker-core"),
        "encoder_sha256": digest("encoder"),
        "repacker_sha256": digest("repacker"),
        "scorer_sha256": digest("scorer"),
        "decoder_sha256": v.PINNED["clean_decoder"],
        "scanned_chunk_indices": list(range(v.CHUNK_COUNT)),
        "base_reports_scanned": v.CHUNK_COUNT,
        "trigger_predicate_universe": "all 400 canonical validated base gaps",
        "row_schema": {
            "base_alphabet_size": "required: 64 or 128",
            "base": "required and explicitly carries alphabet_size",
            "upgrade": "A64: required A128 object; A128: null",
            "tails": "required prefixes against the base alphabet",
        },
        "trigger_gap_db_strictly_greater_than": 0.1,
        "triggered_chunk_indices": [0],
        "triggered_base_alphabet_counts": {"64": 1, "128": 0},
        "tail_prefixes": [1],
        "tail_ranking": "synthetic",
        "failures": [],
        "rows": [
            {
                "chunk_index": 0,
                "trigger_gap_db": 0.2,
                "base_alphabet_size": 64,
                "available_option_kinds": ["base", "alphabet-upgrade", "tail"],
                "base": {
                    "alphabet_size": 64,
                    "report": "private/base.json",
                    "container": "private/base.polar.bin",
                    "container_bytes": len(base_frame),
                    "container_sha256": base_hash,
                    "raw_source_energy": 1000.0,
                    "raw_sse": 100.0,
                },
                "upgrade": {
                    "report": "private/upgrade.json",
                    "container": "private/upgrade.polar.bin",
                    "decode": "private/upgrade.decode.json",
                    "container_bytes": len(selected_frame),
                    "container_sha256": selected_hash,
                    "raw_source_energy": 1000.0,
                    "raw_sse": 90.0,
                    "independent_decode_passed": True,
                    "kind": "alphabet-upgrade",
                    "from_alphabet_size": 64,
                    "to_alphabet_size": 128,
                },
                "tails": [
                    {
                        "chunk_index": 0,
                        "escape_count": 1,
                        "container_bytes": len(base_frame) + 5,
                        "incremental_tail_bytes": 5,
                        "meaningful_tail_bits": 34,
                        "tail_padding_bits": 6,
                        "container_sha256": tail_frame_hash,
                        "container_path": "private/tail-k1.polar.bin",
                        "payload_unchanged": True,
                        "independent_physical_reparse_passed": True,
                        "parsed_tail_applied_for_scoring": True,
                        "raw_gain_identity_passed": True,
                        "raw_source_energy": 1000.0,
                        "raw_sse": 101.0,
                        "raw_sse_reduction": -1.0,
                        "raw_relative_mse": 0.101,
                        "actual_container_bpw": (len(base_frame) + 5) * 8.0 / v.BLOCK_VALUES,
                    }
                ],
            }
        ],
    }
    dump(root / "candidate.receipt.json", candidate)

    side_receipt = {
        "format": "continuous reverse-waterfilled PLTE side receipt v1",
        "status": "exact round-trip passed",
        "side_path": "side.bin",
        "side_bytes": len(literal_side),
        "side_bits": len(literal_side) * 8,
        "side_bpw_over_panel": len(literal_side) * 8.0 / v.PANEL_VALUES,
        "side_sha256": digest(literal_side),
        "header_bytes": v.SIDE_HEADER.size,
        "exp2_lut_bytes": 512,
        "block_scale_bytes": 1600,
        "packed_label_bytes": (v.LABEL_COUNT * v.LABEL_BITS + 7) // 8,
        "profile_bytes": v.CHUNK_COUNT * v.PROFILE.size,
        "stable_membership_reconstructed_from_side_only": True,
        "exact_eof": True,
        "profile_binary64_roundtrip": True,
    }
    dump(root / "side.receipt.json", side_receipt)

    bundle_bindings = {
        "fixed_route_packer_wrapper_sha256": v.PINNED["fixed_route_bundle_packer"],
        "fixed_route_codec_sha256": v.PINNED["fixed_route_codec"],
        "audited_v1_outer_decoder_sha256": v.PINNED["v1_outer_decoder"],
        "pinned_v1_outer_decoder_sha256": v.PINNED["v1_outer_decoder"],
    }
    bundle_receipt = {
        "format": "continuous PLTE WFOUTR fixed-route bundle experiment v2",
        "status": "passed",
        "experimental_not_v1": True,
        "dependency_bindings": bundle_bindings,
        "source_free_reparse_passed": True,
        "bundle_path": "selected.wfouter",
        "bundle_bytes": bundle_bytes,
        "bundle_sha256": digest(bundle),
        "physical_all_in_bpw": bpw,
        "panel_values": v.PANEL_VALUES,
        "header_bytes": len(header),
        "header_sha256": digest(header),
        "physical_prelude_bytes": physical_prelude,
        "side": {
            "codec_id": v.SIDE_CODEC_FIXED_ROUTE,
            "codec": "XZ(canonical all-A64 WFPLTE01) + fixed LSB-first route400",
            "raw_bytes": len(literal_side),
            "raw_sha256": digest(literal_side),
            "canonical_raw_bytes": len(canonical_side),
            "canonical_raw_sha256": digest(canonical_side),
            "canonical_xz_bytes": len(canonical_xz),
            "canonical_xz_sha256": digest(canonical_xz),
            "route_bits": v.ROUTE_BITS,
            "route_bytes": len(route),
            "route_sha256": digest(route),
            "compressed_bytes": len(payload),
            "compressed_sha256": digest(payload),
            "literal_side_reconstructed_hash_verified": True,
            "canonical_profile_offsets_verified": True,
            "alphabet_domain": [64, 128],
        },
        "mask": {
            "codec": "BZ2 level 9",
            "raw_bytes": len(raw_mask),
            "compressed_bytes": len(compressed_mask),
            "raw_sha256": digest(raw_mask),
            "compressed_sha256": digest(compressed_mask),
        },
        "containers": {
            "count": v.CHUNK_COUNT,
            "bytes": sum(map(len, frames)),
            "ordered_sha256": [digest(frame) for frame in frames],
            "all_arithmetic_padding_zero": True,
            "all_sparse_tail_padding_zero": True,
            "exact_eof": True,
        },
    }
    dump(root / "bundle.receipt.json", bundle_receipt)

    base_total_sse = Decimal(str(relative_mse * v.CHUNK_COUNT * 1000.0)) + Decimal(10)
    selected_total_sse = base_total_sse - Decimal(10)
    max_bpw = Decimal("2.5")
    option_rows = [
        {
            "option_id": "base",
            "kind": "base",
            "alphabet_size": 64,
            "container_path": "private/base.polar.bin",
            "container_bytes": len(base_frame),
            "byte_delta_from_base": 0,
            "container_sha256": base_hash,
            "escape_count": 0,
            "base_raw_sse_decimal": "100",
            "raw_sse_decimal": "100",
            "raw_sse_savings_decimal": "0",
        },
        {
            "option_id": "upgrade-a128",
            "kind": "alphabet-upgrade",
            "alphabet_size": 128,
            "container_path": "private/upgrade.polar.bin",
            "container_bytes": len(selected_frame),
            "byte_delta_from_base": 1,
            "container_sha256": selected_hash,
            "escape_count": 0,
            "base_raw_sse_decimal": "100",
            "raw_sse_decimal": "90",
            "raw_sse_savings_decimal": "10",
        },
        {
            "option_id": "tail-k1",
            "kind": "tail",
            "alphabet_size": 64,
            "container_path": "private/tail-k1.polar.bin",
            "container_bytes": len(base_frame) + 5,
            "byte_delta_from_base": 5,
            "container_sha256": tail_frame_hash,
            "escape_count": 1,
            "base_raw_sse_decimal": "100",
            "raw_sse_decimal": "101",
            "raw_sse_savings_decimal": "-1",
        },
    ]
    frontier_rows = [
        {"alphabet_signature_hex": "0x0", "byte_delta": 0, "savings_decimal": "0", "choice_ids": ["base"]},
        {"alphabet_signature_hex": "0x1", "byte_delta": 1, "savings_decimal": "10", "choice_ids": ["upgrade-a128"]},
    ]
    with localcontext() as context:
        context.prec = 80
        objective_log = selected_total_sse.ln() + Decimal(2).ln() * Decimal(16 * bundle_bytes) / Decimal(v.PANEL_VALUES)
        objective = objective_log.exp()
        reference_decimal = (-(Decimal(2) * Decimal(8 * bundle_bytes) / Decimal(v.PANEL_VALUES)) * Decimal(2).ln()).exp()
        gap_decimal = Decimal(10) * ((selected_total_sse / Decimal(400000)) / reference_decimal).ln() / Decimal(10).ln()
    selection_map = []
    for index, frame in enumerate(frames):
        triggered = index == 0
        selection_map.append(
            {
                "chunk_index": index,
                "option_id": "upgrade-a128" if triggered else "base",
                "kind": "alphabet-upgrade" if triggered else "base",
                "base_alphabet_size": 64,
                "selected_alphabet_size": 128 if triggered else 64,
                "source_container_path": "private/selected.polar.bin",
                "staged_container": f"containers/wf-{index:03d}.polar.bin",
                "container_bytes": len(frame),
                "container_sha256": digest(frame),
                "escape_count": 0,
                "triggered": triggered,
                "selected_raw_sse_decimal_if_triggered": "90" if triggered else None,
                "raw_sse_savings_decimal": "10" if triggered else "0",
            }
        )
    selection = {
        "format": "continuous PLTE exact adaptive selection receipt fixed-route v2",
        "status": "passed",
        "strict_ptq": True,
        "write_once_output": True,
        "objective": "(base_raw_sse - savings) * 2**(16 * WFOUTR01_bundle_bytes / panel_values)",
        "physical_selection_mode": "fixed-route",
        "selection_guarantee": "global exact: side codec 3 has one fixed 50-byte route and an alphabet-invariant canonical all-A64 XZ member",
        "selection_arithmetic": {
            "byte_axis": "exact integer physical bytes",
            "sse_axis": "exact Decimal values parsed from serialized candidate JSON",
            "transcendental_comparison": "Python Decimal ln at 80-digit precision",
            "tie_break": "objective, then total bytes, then lexicographic option IDs",
            "pareto_rule_exact_mode": "synthetic",
            "side_physical_model": "codec 3: one XZ(canonical all-A64 side) plus fixed 50-byte route400; selected bytes verified against isolated v2 packer",
            "mask_physical_model": "Python bz2 level 9 once; selected bytes verified",
        },
        "inputs": {
            "manifest_path": "construction.manifest.json",
            "manifest_sha256": v.sha256_path(root / "construction.manifest.json"),
            "base_dir": "private/base",
            "base_run_receipt_sha256": v.sha256_path(root / "base.run.receipt.json"),
            "candidate_receipt_path": "candidate.receipt.json",
            "candidate_receipt_sha256": v.sha256_path(root / "candidate.receipt.json"),
            "candidate_v3_declared_dependency_hashes": {},
            "packer_path": "pack_continuous_side.py",
            "packer_sha256": v.PINNED["side_packer"],
            "bundle_packer_path": "pack_bundle_fixed_route_v2.py",
            "bundle_packer_sha256": v.PINNED["fixed_route_bundle_packer"],
            "raw_mask_path": "private/raw_mask.bin",
            "raw_mask_bytes": len(raw_mask),
            "raw_mask_sha256": digest(raw_mask),
            "python": "python",
            "base_total_raw_sse_decimal": str(base_total_sse),
            "total_raw_energy_decimal": "400000",
            "total_raw_energy_origin": "synthetic",
        },
        "validation": {
            "canonical_chunks": v.CHUNK_COUNT,
            "one_staged_option_per_chunk": True,
            "all_base_reports_and_containers_hash_validated": True,
            "all_adaptive_reports_decodes_and_container_hashes_validated": True,
            "tail_payload_identity_validated": True,
            "original_raw_side_bytes": len(literal_side),
            "selected_raw_side_bytes": len(literal_side),
            "raw_side_length_invariant_after_alphabet_regeneration": True,
            "selected_raw_side_matches_reranked_signature_byte_for_byte": True,
            "selected_physical_side_payload_matches_rerank_hash_and_length": True,
            "selected_bz2_mask_matches_rerank_hash_and_length": True,
            "selected_manifest_side_roundtrip": "exact round-trip passed",
            "wfoutr01_source_free_reparse": True,
        },
        "dp": {
            "triggered_chunks": [0],
            "triggered_count": 1,
            "option_count": 3,
            "proxy_frontier_states": 2,
            "physical_frontier_states": 2,
            "peak_exact_signature_states": 2,
            "distinct_physical_side_signatures": 2,
            "fixed_route_bits": v.ROUTE_BITS,
            "frontier_sha256": v.canonical_sha256(frontier_rows),
            "selected_alphabet_signature_hex": "0x1",
            "selected_choice_ids": ["upgrade-a128"],
            "selected_byte_delta": 1,
            "selected_raw_sse_savings_decimal": "10",
            "raw_side_constant_proxy_selected_choice_ids": ["upgrade-a128"],
            "raw_side_constant_proxy_log_objective_decimal": str(objective_log),
            "raw_side_constant_proxy_is_claim_accounting": False,
        },
        "artifacts": {
            "selected_manifest": "selected.manifest.json",
            "selected_manifest_sha256": v.sha256_path(root / "selected.manifest.json"),
            "literal_side": "side.bin",
            "literal_side_raw_bytes": len(literal_side),
            "literal_side_sha256": digest(literal_side),
            "side_receipt": "side.receipt.json",
            "side_receipt_sha256": v.sha256_path(root / "side.receipt.json"),
            "container_directory": "containers",
            "physical_bundle": "selected.wfouter",
            "physical_bundle_bytes": bundle_bytes,
            "physical_bundle_sha256": digest(bundle),
            "bundle_receipt": "bundle.receipt.json",
            "bundle_receipt_sha256": v.sha256_path(root / "bundle.receipt.json"),
        },
        "accounting": {
            "panel_values": v.PANEL_VALUES,
            "base_container_bytes": len(base_frame) * v.CHUNK_COUNT,
            "base_original_side_physical_payload_bytes": len(payload),
            "base_modeled_physical_bundle_bytes": bundle_bytes - 1,
            "selected_container_bytes": sum(map(len, frames)),
            "raw_side_bytes_not_directly_charged": len(literal_side),
            "wfoutr01_header_bytes": len(header),
            "side_physical_payload_bytes": len(payload),
            "side_canonical_xz_bytes": len(canonical_xz),
            "side_fixed_route_bytes": len(route),
            "mask_bz2_compressed_bytes": len(compressed_mask),
            "physical_prelude_bytes": physical_prelude,
            "physical_bundle_bytes": bundle_bytes,
            "strict_max_bpw_decimal": str(max_bpw),
            "physical_all_in_bpw_decimal": str(Decimal(8 * bundle_bytes) / Decimal(v.PANEL_VALUES)),
            "strict_rate_pass": True,
            "selected_total_raw_sse_decimal": str(selected_total_sse),
            "raw_relative_mse_decimal": str(selected_total_sse / Decimal(400000)),
            "gaussian_reference_gap_db_decimal": str(gap_decimal),
            "objective_decimal": str(objective),
            "objective_change_from_base_db_decimal": "0",
        },
        "options_considered": {"0": option_rows},
        "selection_map": selection_map,
    }
    dump(root / "selection.receipt.json", selection)
    (root / "selection.receipt.sha256").write_text(
        f"{v.sha256_path(root / 'selection.receipt.json')}  selection.receipt.json\n",
        encoding="ascii",
    )

    decode_dependencies = {
        "fixed_route_decoder_wrapper_sha256": v.PINNED["fixed_route_outer_decoder"],
        "fixed_route_codec_sha256": v.PINNED["fixed_route_codec"],
        "audited_v1_outer_decoder_sha256": v.PINNED["v1_outer_decoder"],
        "pinned_v1_outer_decoder_sha256": v.PINNED["v1_outer_decoder"],
    }
    decode_chunks = []
    for index, frame in enumerate(frames):
        decode_chunks.append(
            {
                "chunk_index": index,
                "container_bytes": len(frame),
                "container_sha256": digest(frame),
                "logical_bits": 8 if index == 0 else 0,
                "escape_count": 0,
                "fp32_decoder_scale": 1.0,
                "arithmetic_padding_bits": 0,
                "arithmetic_padding_zero": True,
                "tail_padding_bits": 0,
                "tail_padding_zero": True,
                "alphabet_size": 128 if index == 0 else 64,
                "arithmetic_decoder_bits_read": 8 if index == 0 else 0,
                "frequency_u16_sha256": digest(f"frequency-{index}"),
                "reconstruction_indices_i16_sha256": digest(f"indices-{index}"),
            }
        )
    reconstruction_hash = digest("synthetic-reconstruction")
    decode = {
        "format": "continuous reverse-waterfilled PLTE independent outer decode fixed-route experiment v2",
        "status": "passed",
        "strict_ptq": True,
        "experimental_not_v1": True,
        "independence": {
            "read_encoder_json": False,
            "read_exploratory_manifest": False,
            "read_normalized_or_raw_source": False,
            "encoder_probability_arrays_used": False,
            "membership_derived_from_literal_side_only": True,
        },
        "geometry": {
            "canonical_blocks": v.CHUNK_COUNT,
            "groups_per_canonical_block": v.GROUPS_PER_CHUNK,
            "group_values": v.GROUP_VALUES,
            "polar_chunks": v.CHUNK_COUNT,
            "polar_block_values": v.BLOCK_VALUES,
            "panel_values": v.PANEL_VALUES,
        },
        "side": {"bytes": len(literal_side), "sha256": digest(literal_side), "exact_eof": True, "packed_label_count": v.LABEL_COUNT},
        "encoded_stream": {
            "layout": "fixed route",
            "self_contained_decoder_side_and_mask": True,
            "physical_prelude_bytes": physical_prelude,
            "container_stream_bytes": sum(map(len, frames)),
            "container_stream_sha256": digest(b"".join(frames)),
            "combined_encoded_bytes": bundle_bytes,
            "combined_encoded_sha256": digest(bundle),
            "exact_eof_after_declared_chunk_count": True,
            "all_arithmetic_padding_zero": True,
            "all_sparse_tail_padding_zero": True,
            "actual_all_in_bpw": bpw,
        },
        "decoder_assets": {
            "raw_mask_bytes": len(raw_mask),
            "raw_mask_sha256": digest(raw_mask),
            "raw_mask_embedded_and_physically_charged": True,
            "clean_decoder_sha256": v.PINNED["clean_decoder"],
            "outer_decoder_sha256": v.PINNED["v1_outer_decoder"],
            **decode_dependencies,
        },
        "outer_bundle": {
            "side_codec_id": v.SIDE_CODEC_FIXED_ROUTE,
            "canonical_side_raw_bytes": len(canonical_side),
            "canonical_side_raw_sha256": digest(canonical_side),
            "canonical_side_xz_bytes": len(canonical_xz),
            "canonical_side_xz_sha256": digest(canonical_xz),
            "route_bits": v.ROUTE_BITS,
            "route_bytes": len(route),
            "route_sha256": digest(route),
            "reconstructed_literal_side_sha256": digest(literal_side),
            "reconstructed_literal_side_hash_verified": True,
            "canonical_profile_offsets_verified": True,
            "alphabet_domain": [64, 128],
            "decompressor_exact_eof": True,
        },
        "dependency_bindings": decode_dependencies,
        "alphabet_census": {"64": 399, "128": 1, "256": 0},
        "chunks": decode_chunks,
        "reconstruction": {
            "path": "private/reconstruction.f32",
            "shape": [v.CHUNK_COUNT, v.BLOCK_VALUES],
            "dtype": "<f4",
            "bytes": v.PANEL_VALUES * 4,
            "sha256": reconstruction_hash,
            "every_canonical_group_written_exactly_once": True,
        },
    }
    dump(root / "decode.receipt.json", decode)

    evaluation_blocks = []
    block_sse = relative_mse * 1000.0
    for index, ledger_row in enumerate(ledger_rows):
        evaluation_blocks.append(
            {
                "canonical_block_ordinal": index,
                "id": ledger_row["id"],
                "tensor": ledger_row["tensor"],
                "role": ledger_row["role"],
                "layer": ledger_row["layer"],
                "source_sha256": ledger_row["sha256"],
                "source_energy": 1000.0,
                "sse": block_sse,
                "relative_mse": relative_mse,
                "effective_charged_panel_bpw": bpw,
                "gaussian_reference_mse_at_effective_panel_rate": reference,
                "diagnostic_gap_db_at_effective_panel_rate": signed_gap,
            }
        )
    mixed = []
    shared_bpw = physical_prelude * 8.0 / v.PANEL_VALUES
    for index, frame in enumerate(frames):
        container_bpw = len(frame) * 8.0 / v.BLOCK_VALUES
        effective = container_bpw + shared_bpw
        chunk_reference = 2.0 ** (-2.0 * effective)
        mixed.append(
            {
                "chunk_index": index,
                "mixed_groups_from_canonical_blocks": 1,
                "source_energy": 1000.0,
                "sse": block_sse,
                "relative_mse": relative_mse,
                "container_bytes": len(frame),
                "container_bpw": container_bpw,
                "allocated_shared_outer_bpw": shared_bpw,
                "effective_charged_chunk_bpw": effective,
                "gaussian_reference_mse_at_effective_chunk_rate": chunk_reference,
                "diagnostic_gap_db": 10.0 * math.log10(relative_mse / chunk_reference),
                "diagnostic_only": True,
            }
        )
    evaluation_dependencies = {
        "fixed_route_evaluator_wrapper_sha256": v.PINNED["fixed_route_evaluator"],
        "delegated_v1_evaluator_sha256": v.PINNED["v1_evaluator"],
        "pinned_v1_evaluator_sha256": v.PINNED["v1_evaluator"],
        **decode_dependencies,
    }
    total_energy = 400000.0
    total_sse = block_sse * v.CHUNK_COUNT
    evaluation = {
        "format": "continuous reverse-waterfilled PLTE exact-source evaluation fixed-route experiment v2",
        "status": "passed",
        "strict_ptq": True,
        "experimental_not_v1": True,
        "source_is_evaluator_only": True,
        "source_ledger_sha256": v.sha256_path(root / "source_ledger.json"),
        "ordered_source_identity_sha256": v.ordered_source_identity(ledger_rows),
        "source_blocks": v.CHUNK_COUNT,
        "panel_values": v.PANEL_VALUES,
        "all_source_hashes_and_ordinals_verified": True,
        "all_400_source_hashes_ordinals_and_scatter_coverage_verified": True,
        "decode_receipt_sha256": v.sha256_path(root / "decode.receipt.json"),
        "reconstruction_sha256": reconstruction_hash,
        "encoded_sha256": digest(bundle),
        "loaded_outer_decoder_sha256": v.PINNED["fixed_route_outer_decoder"],
        "normative_clean_decoder_sha256": v.PINNED["clean_decoder"],
        "normative_raw_mask_sha256": digest(raw_mask),
        "encoded_bytes": bundle_bytes,
        "actual_all_in_bpw": bpw,
        "source_energy": total_energy,
        "sse": total_sse,
        "relative_mse": relative_mse,
        "gaussian_reference_mse_at_actual_rate": reference,
        "gaussian_reference_gap_db": signed_gap,
        "target_gap_db": -0.10,
        "target_gap_le_negative_0p10_db_passed": signed_gap <= -0.10,
        "rate_limit_bpw": 2.5,
        "strict_rate_below_2p5_bpw_passed": True,
        "blocks": evaluation_blocks,
        "by_role": strata(evaluation_blocks, ("role",), bpw, reference),
        "by_layer": strata(evaluation_blocks, ("layer",), bpw, reference),
        "by_role_and_layer": strata(evaluation_blocks, ("role", "layer"), bpw, reference),
        "mixed_chunk_diagnostics": mixed,
        "evaluation_dependency_bindings": evaluation_dependencies,
        "fixed_route_codec": {
            "side_codec_id": v.SIDE_CODEC_FIXED_ROUTE,
            "route_bits": v.ROUTE_BITS,
            "literal_side_hash_verified_before_source_scoring": True,
        },
    }
    dump(root / "evaluation.json", evaluation)


def rewrite_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    dump(path, value)


def refresh_selection_checksum(root: Path) -> None:
    (root / "selection.receipt.sha256").write_text(
        f"{v.sha256_path(root / 'selection.receipt.json')}  selection.receipt.json\n",
        encoding="ascii",
    )


class PublicationVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_mask_hash = v.PINNED["raw_mask"]
        v.PINNED["raw_mask"] = digest(bytes(v.MASK_RAW_BYTES))

    @classmethod
    def tearDownClass(cls) -> None:
        v.PINNED["raw_mask"] = cls.original_mask_hash

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "publication"
        self.root.mkdir()
        build_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_rejected(self) -> None:
        with self.assertRaises((v.VerificationError, KeyError, TypeError, ValueError, OSError, struct.error)):
            v.verify_publication(self.root)

    def test_valid_source_free_fixture(self) -> None:
        report = v.verify_publication(self.root)
        self.assertEqual(report["status"], "passed")
        self.assertLessEqual(Decimal(report["signed_gaussian_gap_db_decimal"]), Decimal("-0.10"))
        self.assertEqual(report["canonical_blocks"], 400)

    def test_bundle_byte_tamper_rejected(self) -> None:
        path = self.root / "selected.wfouter"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        self.assert_rejected()

    def test_physical_bpw_tamper_rejected(self) -> None:
        rewrite_json(self.root / "evaluation.json", lambda row: row.__setitem__("actual_all_in_bpw", 1.0))
        self.assert_rejected()

    def test_role_layer_coverage_tamper_rejected(self) -> None:
        rewrite_json(self.root / "source_ledger.json", lambda row: row["blocks"][64].__setitem__("layer", 47))
        self.assert_rejected()

    def test_dp_choice_tamper_rejected(self) -> None:
        rewrite_json(self.root / "selection.receipt.json", lambda row: row["dp"].__setitem__("selected_byte_delta", 0))
        refresh_selection_checksum(self.root)
        self.assert_rejected()

    def test_independent_decode_chunk_tamper_rejected(self) -> None:
        rewrite_json(self.root / "decode.receipt.json", lambda row: row["chunks"][3].__setitem__("container_sha256", digest("wrong")))
        self.assert_rejected()

    def test_strict_ptq_tamper_rejected(self) -> None:
        rewrite_json(self.root / "candidate.receipt.json", lambda row: row.__setitem__("strict_ptq", False))
        self.assert_rejected()

    def test_raw_source_hash_hidden_under_innocent_name_rejected(self) -> None:
        (self.root / "innocent.dat").write_bytes(b"source-0")
        self.assert_rejected()

    def test_staged_container_tamper_rejected(self) -> None:
        (self.root / "containers" / "wf-099.polar.bin").write_bytes(b"tampered")
        self.assert_rejected()

    def test_candidate_option_omission_rejected(self) -> None:
        rewrite_json(self.root / "candidate.receipt.json", lambda row: row["rows"][0].__setitem__("tails", []))
        self.assert_rejected()

    def test_signed_gap_above_negative_point_one_rejected(self) -> None:
        shutil.rmtree(self.root)
        self.root.mkdir()
        build_fixture(self.root, relative_mse=0.99)
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main(verbosity=2)
