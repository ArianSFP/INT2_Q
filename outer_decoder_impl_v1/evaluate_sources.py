#!/usr/bin/env python3
"""Separately score a canonical outer reconstruction against exact BF16 sources.

The source ledger is intentionally an evaluator-only input.  It is never read
by ``outer_decode.py``.  Every source is bound to canonical order by an explicit
ordinal and SHA256; no encoder report or group-membership manifest is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cupy as cp
import numpy as np

import outer_decode as dec


TARGET_GAP_DB = -0.10
RATE_LIMIT_BPW = 2.5
EXPECTED_CLEAN_DECODER_SHA256 = (
    "7589f4be6e784d8e5a0067303da389b6d982430eb84fda52f668808f322c25d9"
)
EXPECTED_RAW_MASK_SHA256 = (
    "11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6"
)
EXPECTED_CHECKPOINT_REPO = "Qwen/Qwen3-30B-A3B"
EXPECTED_CHECKPOINT_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = Path(str(path) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-receipt", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-source-ledger-sha256",
        required=True,
        help="published SHA256 of the frozen evaluator-only source ledger",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = load_json(args.decode_receipt)
    if receipt.get("status") != "passed":
        raise ValueError("decode receipt is not passed")
    if receipt.get("format") != "continuous reverse-waterfilled PLTE independent outer decode v1":
        raise ValueError("unsupported decode receipt format")
    reconstruction_row = receipt["reconstruction"]
    geometry = receipt["geometry"]
    encoded_row = receipt["encoded_stream"]
    decoder_assets = receipt["decoder_assets"]
    if encoded_row.get("self_contained_decoder_side_and_mask") is not True:
        raise ValueError("claim evaluation requires a self-contained WFOUTR01 bundle")
    if decoder_assets.get("raw_mask_embedded_and_physically_charged") is not True:
        raise ValueError("decode receipt does not charge the embedded raw mask")
    if str(decoder_assets.get("clean_decoder_sha256")) != EXPECTED_CLEAN_DECODER_SHA256:
        raise ValueError("decode receipt used a non-normative clean decoder")
    if str(decoder_assets.get("raw_mask_sha256")) != EXPECTED_RAW_MASK_SHA256:
        raise ValueError("decode receipt used a non-normative polar mask")
    local_outer_hash = sha256_path(Path(dec.__file__))
    if str(decoder_assets.get("outer_decoder_sha256")) != local_outer_hash:
        raise ValueError("decode receipt does not match the loaded outer decoder")
    expected_reconstruction_hash = str(reconstruction_row["sha256"])
    actual_reconstruction_hash = sha256_path(args.reconstruction)
    if actual_reconstruction_hash != expected_reconstruction_hash:
        raise ValueError("reconstruction SHA256 does not match decode receipt")
    expected_bytes = int(reconstruction_row["bytes"])
    if args.reconstruction.stat().st_size != expected_bytes:
        raise ValueError("reconstruction byte count does not match decode receipt")

    encoded_bytes = args.bundle.stat().st_size
    encoded_hash = sha256_path(args.bundle)
    with args.bundle.open("rb") as handle:
        prelude = dec.read_bundle_prelude(handle)
        frames = dec.read_all_frames(handle, prelude.side.chunk_count)
    side = prelude.side
    if sha256_bytes(prelude.raw_mask) != EXPECTED_RAW_MASK_SHA256:
        raise ValueError("embedded polar mask does not match the normative hash")
    actual_prelude_bytes = (
        len(prelude.header)
        + len(prelude.compressed_side)
        + len(prelude.compressed_mask)
    )
    actual_container_stream_bytes = sum(len(frame.literal) for frame in frames)
    if int(encoded_row.get("physical_prelude_bytes", -1)) != actual_prelude_bytes:
        raise ValueError("decode receipt physical prelude byte count is inconsistent")
    if int(encoded_row.get("container_stream_bytes", -1)) != actual_container_stream_bytes:
        raise ValueError("decode receipt container stream byte count is inconsistent")
    if encoded_bytes != actual_prelude_bytes + actual_container_stream_bytes:
        raise AssertionError("physical bundle partition does not sum to bundle bytes")
    if encoded_bytes != int(encoded_row["combined_encoded_bytes"]):
        raise ValueError("encoded byte count does not match decode receipt")
    if encoded_hash != str(encoded_row["combined_encoded_sha256"]):
        raise ValueError("encoded SHA256 does not match decode receipt")

    ledger = load_json(args.source_ledger)
    actual_ledger_hash = sha256_path(args.source_ledger)
    expected_ledger_hash = args.expected_source_ledger_sha256.lower()
    if len(expected_ledger_hash) != 64 or actual_ledger_hash != expected_ledger_hash:
        raise ValueError("source ledger does not match its published SHA256")
    if ledger.get("format") != "canonical BF16 source ledger v1":
        raise ValueError("unsupported source-ledger format")
    checkpoint = ledger.get("checkpoint")
    if checkpoint != {
        "repo": EXPECTED_CHECKPOINT_REPO,
        "revision": EXPECTED_CHECKPOINT_REVISION,
    }:
        raise ValueError("source ledger does not pin the normative Qwen checkpoint")
    blocks = ledger.get("blocks")
    expected_geometry = {
        "canonical_blocks": side.block_count,
        "groups_per_canonical_block": side.groups_per_chunk,
        "group_values": side.group_values,
        "polar_chunks": side.chunk_count,
        "polar_block_values": dec.NORMATIVE_BLOCK_LENGTH,
        "panel_values": side.label_count * side.group_values,
    }
    if geometry != expected_geometry:
        raise ValueError("decode receipt geometry does not match the reparsed bundle side")
    block_count = side.block_count
    block_values = side.groups_per_chunk * side.group_values
    panel_values = side.label_count * side.group_values
    if block_count != 400:
        raise ValueError(f"claim evaluation requires the frozen 400-block panel, got {block_count}")
    if not isinstance(blocks, list) or len(blocks) != block_count:
        raise ValueError(f"source ledger must contain exactly {block_count} blocks")
    ordinals = [int(row["canonical_block_ordinal"]) for row in blocks]
    if ordinals != list(range(block_count)):
        raise ValueError("source ledger is not in exact canonical ordinal order")
    source_ids = [str(row["id"]) for row in blocks]
    if len(set(source_ids)) != block_count or any(not value for value in source_ids):
        raise ValueError("source ledger IDs must be nonempty and unique")
    for ordinal, row in enumerate(blocks):
        for field in ("tensor", "role", "layer"):
            if field not in row:
                raise ValueError(f"source ledger row {ordinal} omits required field {field!r}")
        if not str(row["tensor"]) or not str(row["role"]):
            raise ValueError(f"source ledger row {ordinal} has an empty tensor/role")
        layer = row["layer"]
        if layer is not None and (not isinstance(layer, int) or isinstance(layer, bool)):
            raise ValueError(f"source ledger row {ordinal} layer must be integer or null")

    dtype = np.dtype(str(reconstruction_row["dtype"]))
    shape = tuple(int(value) for value in reconstruction_row["shape"])
    if shape != (block_count, block_values):
        raise ValueError("decode receipt reconstruction shape is inconsistent")
    reconstruction = np.memmap(
        args.reconstruction, dtype=dtype, mode="r", shape=shape
    )
    total_energy = 0.0
    total_sse = 0.0
    groups_per_block = int(geometry["groups_per_canonical_block"])
    group_values = int(geometry["group_values"])
    group_energy = np.empty(block_count * groups_per_block, dtype=np.float64)
    group_sse = np.empty(block_count * groups_per_block, dtype=np.float64)
    source_identity = hashlib.sha256()
    block_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(blocks):
        source_path = resolve_source(args.source_root, str(row["path"]))
        payload = source_path.read_bytes()
        expected_hash = str(row["sha256"]).lower()
        actual_hash = sha256_bytes(payload)
        if actual_hash != expected_hash:
            raise ValueError(
                f"source ordinal {ordinal} SHA256 mismatch: {actual_hash} != {expected_hash}"
            )
        if len(payload) != block_values * 2:
            raise ValueError(
                f"source ordinal {ordinal} has {len(payload)} bytes, expected {block_values * 2}"
            )
        raw_u16 = np.frombuffer(payload, dtype="<u2")
        source_f32 = (raw_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        source_gpu = cp.asarray(source_f32, dtype=cp.float64)
        reconstruction_gpu = cp.asarray(
            np.asarray(reconstruction[ordinal]), dtype=cp.float64
        )
        source_grouped = source_gpu.reshape(groups_per_block, group_values)
        reconstruction_grouped = reconstruction_gpu.reshape(groups_per_block, group_values)
        energy_by_group = cp.asnumpy(
            cp.sum(cp.square(source_grouped), axis=1, dtype=cp.float64)
        )
        sse_by_group = cp.asnumpy(
            cp.sum(
                cp.square(source_grouped - reconstruction_grouped),
                axis=1,
                dtype=cp.float64,
            )
        )
        group_begin = ordinal * groups_per_block
        group_end = group_begin + groups_per_block
        group_energy[group_begin:group_end] = energy_by_group
        group_sse[group_begin:group_end] = sse_by_group
        energy = float(np.sum(energy_by_group, dtype=np.float64))
        sse = float(np.sum(sse_by_group, dtype=np.float64))
        if not math.isfinite(energy) or energy <= 0.0 or not math.isfinite(sse):
            raise ValueError(f"invalid score at source ordinal {ordinal}")
        total_energy += energy
        total_sse += sse
        source_identity.update(ordinal.to_bytes(8, "little"))
        source_identity.update(len(source_ids[ordinal].encode("utf-8")).to_bytes(4, "little"))
        source_identity.update(source_ids[ordinal].encode("utf-8"))
        source_identity.update(bytes.fromhex(actual_hash))
        for identity_field in ("tensor", "role", "layer"):
            identity_value = json.dumps(
                row[identity_field], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            source_identity.update(len(identity_value).to_bytes(4, "little"))
            source_identity.update(identity_value)
        block_rows.append(
            {
                "canonical_block_ordinal": ordinal,
                "id": source_ids[ordinal],
                "tensor": str(row["tensor"]),
                "role": str(row["role"]),
                "layer": row["layer"],
                "source_sha256": actual_hash,
                "source_energy": energy,
                "sse": sse,
                "relative_mse": sse / energy,
            }
        )
        print(
            f"[{ordinal + 1}/{block_count}] scored canonical block {ordinal:03d}",
            flush=True,
        )
    del reconstruction
    relative_mse = total_sse / total_energy
    actual_bpw = encoded_bytes * 8.0 / panel_values
    gaussian_reference_mse = 2.0 ** (-2.0 * actual_bpw)
    gap_db = 10.0 * math.log10(relative_mse / gaussian_reference_mse)
    for row in block_rows:
        row["effective_charged_panel_bpw"] = actual_bpw
        row["gaussian_reference_mse_at_effective_panel_rate"] = gaussian_reference_mse
        row["diagnostic_gap_db_at_effective_panel_rate"] = 10.0 * math.log10(
            float(row["relative_mse"]) / gaussian_reference_mse
        )

    def aggregate_strata(fields: tuple[str, ...]) -> list[dict[str, object]]:
        buckets: dict[tuple[object, ...], dict[str, object]] = {}
        for row in block_rows:
            key = tuple(row[field] for field in fields)
            bucket = buckets.setdefault(
                key,
                {
                    "blocks": 0,
                    "values": 0,
                    "source_energy": 0.0,
                    "sse": 0.0,
                    "ids": [],
                },
            )
            bucket["blocks"] = int(bucket["blocks"]) + 1
            bucket["values"] = int(bucket["values"]) + block_values
            bucket["source_energy"] = float(bucket["source_energy"]) + float(
                row["source_energy"]
            )
            bucket["sse"] = float(bucket["sse"]) + float(row["sse"])
            bucket["ids"].append(str(row["id"]))
        output: list[dict[str, object]] = []
        for key in sorted(buckets, key=lambda item: tuple("<global>" if value is None else str(value) for value in item)):
            bucket = buckets[key]
            distortion = float(bucket["sse"]) / float(bucket["source_energy"])
            result_row: dict[str, object] = {
                field: value for field, value in zip(fields, key, strict=True)
            }
            result_row.update(
                blocks=int(bucket["blocks"]),
                values=int(bucket["values"]),
                source_energy=float(bucket["source_energy"]),
                sse=float(bucket["sse"]),
                relative_mse=distortion,
                effective_charged_panel_bpw=actual_bpw,
                gaussian_reference_mse_at_effective_panel_rate=gaussian_reference_mse,
                diagnostic_gap_db_at_effective_panel_rate=10.0
                * math.log10(distortion / gaussian_reference_mse),
                ids=bucket["ids"],
            )
            output.append(result_row)
        return output

    if side.label_count != group_energy.size or side.chunk_count != len(frames):
        raise AssertionError("side/evaluator group or chunk census mismatch")
    chunk_rows: list[dict[str, object]] = []
    physical_prelude_bytes = actual_prelude_bytes
    shared_outer_bpw = physical_prelude_bytes * 8.0 / panel_values
    receipt_chunks = receipt["chunks"]
    if len(receipt_chunks) != side.chunk_count:
        raise ValueError("decode receipt chunk census mismatch")
    if [int(row.get("chunk_index", -1)) for row in receipt_chunks] != list(
        range(side.chunk_count)
    ):
        raise ValueError("decode receipt chunks are not in exact canonical order")
    for chunk_index in range(side.chunk_count):
        begin = chunk_index * side.groups_per_chunk
        end = begin + side.groups_per_chunk
        members = side.stable_order[begin:end]
        energy = float(np.sum(group_energy[members], dtype=np.float64))
        sse = float(np.sum(group_sse[members], dtype=np.float64))
        distortion = sse / energy
        container_bytes = len(frames[chunk_index].literal)
        if container_bytes != int(receipt_chunks[chunk_index]["container_bytes"]):
            raise ValueError(f"chunk {chunk_index} byte count disagrees with decode receipt")
        if frames[chunk_index].sha256 != str(receipt_chunks[chunk_index]["container_sha256"]):
            raise ValueError(f"chunk {chunk_index} hash disagrees with decode receipt")
        effective_chunk_bpw = (
            container_bytes * 8.0 / int(geometry["polar_block_values"])
            + shared_outer_bpw
        )
        reference = 2.0 ** (-2.0 * effective_chunk_bpw)
        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "mixed_groups_from_canonical_blocks": int(
                    np.unique(members // groups_per_block).size
                ),
                "source_energy": energy,
                "sse": sse,
                "relative_mse": distortion,
                "container_bytes": container_bytes,
                "container_bpw": container_bytes
                * 8.0
                / int(geometry["polar_block_values"]),
                "allocated_shared_outer_bpw": shared_outer_bpw,
                "effective_charged_chunk_bpw": effective_chunk_bpw,
                "gaussian_reference_mse_at_effective_chunk_rate": reference,
                "diagnostic_gap_db": 10.0 * math.log10(distortion / reference),
                "diagnostic_only": True,
            }
        )
    target_passed = gap_db <= TARGET_GAP_DB
    rate_passed = actual_bpw < RATE_LIMIT_BPW
    result = {
        "format": "continuous reverse-waterfilled PLTE exact-source evaluation v1",
        "status": "passed" if target_passed and rate_passed else "target_not_met",
        "strict_ptq": True,
        "source_is_evaluator_only": True,
        "source_ledger_sha256": actual_ledger_hash,
        "ordered_source_identity_sha256": source_identity.hexdigest(),
        "source_blocks": block_count,
        "panel_values": panel_values,
        "all_source_hashes_and_ordinals_verified": True,
        "all_400_source_hashes_ordinals_and_scatter_coverage_verified": True,
        "decode_receipt_sha256": sha256_path(args.decode_receipt),
        "reconstruction_sha256": actual_reconstruction_hash,
        "encoded_sha256": encoded_hash,
        "loaded_outer_decoder_sha256": local_outer_hash,
        "normative_clean_decoder_sha256": EXPECTED_CLEAN_DECODER_SHA256,
        "normative_raw_mask_sha256": EXPECTED_RAW_MASK_SHA256,
        "encoded_bytes": encoded_bytes,
        "actual_all_in_bpw": actual_bpw,
        "source_energy": total_energy,
        "sse": total_sse,
        "relative_mse": relative_mse,
        "gaussian_reference_mse_at_actual_rate": gaussian_reference_mse,
        "gaussian_reference_gap_db": gap_db,
        "target_gap_db": TARGET_GAP_DB,
        "target_gap_le_negative_0p10_db_passed": target_passed,
        "rate_limit_bpw": RATE_LIMIT_BPW,
        "strict_rate_below_2p5_bpw_passed": rate_passed,
        "rate_interpretation": {
            "aggregate": "primary: physical self-contained bundle bits divided by all panel values",
            "role_layer_block_strata": "diagnostic: each stratum is compared at the same aggregate charged panel bpw because polar chunks mix groups across blocks",
            "mixed_chunks": "diagnostic: literal chunk container bpw plus the physical outer prelude bpw shared uniformly over panel values",
        },
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "blocks": block_rows,
        "by_role": aggregate_strata(("role",)),
        "by_layer": aggregate_strata(("layer",)),
        "by_role_and_layer": aggregate_strata(("role", "layer")),
        "mixed_chunk_diagnostics": chunk_rows,
    }
    atomic_json(args.output, result, args.overwrite)
    print(json.dumps({key: value for key, value in result.items() if key != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
