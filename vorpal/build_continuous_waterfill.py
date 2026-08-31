#!/usr/bin/env python3
"""Build an exploratory continuous reverse-waterfilled PLTE source panel.

The frozen Qwen BF16 blocks are divided into canonical groups of 2,048 values.
Each group receives a decoder-visible six-bit relative log-RMS label.  Groups
are stably ordered by their reconstructed quantized variance, collected into
full N=2^18 polar blocks, and normalized by the quantized RMS.  A single
serialized water level determines every block's scale-invariant PLTE operating
point.  This is strict PTQ preprocessing: there are no gradients, activations,
or trained parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cupy as cp
import numpy as np


BLOCK_VALUES = 1 << 18
GROUP_VALUES = 1 << 11
GROUPS_PER_BLOCK = BLOCK_VALUES // GROUP_VALUES
SIGMA_SOURCE = 3.0
BASE_D = 0.29
BASE_ETA = 0.5989929996555583
ETA_RATIO = BASE_ETA / math.sqrt((SIGMA_SOURCE**2 - BASE_D) * BASE_D / SIGMA_SOURCE**2)


def bf16_to_cupy(path: Path) -> cp.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != BLOCK_VALUES:
        raise ValueError(f"{path}: expected {BLOCK_VALUES} BF16 values, got {raw.size}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return cp.asarray(values)


def cupy_to_bf16_rne(values: cp.ndarray) -> np.ndarray:
    f32 = cp.asarray(values, dtype=cp.float32)
    bits = f32.view(cp.uint32)
    lsb = (bits >> cp.uint32(16)) & cp.uint32(1)
    rounded = bits + cp.uint32(0x7FFF) + lsb
    return cp.asnumpy((rounded >> cp.uint32(16)).astype(cp.uint16))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-variance", type=float, default=1.8252209629460492e-5)
    parser.add_argument("--min-rate", type=float, default=1.45)
    parser.add_argument("--max-rate", type=float, default=3.05)
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("results", rows.get("blocks"))
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError("expected the frozen 400-block results list")
    rows = sorted(rows, key=lambda row: str(row["id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "normalized_sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    block_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    source_gpu: list[cp.ndarray] = []
    total_energy = 0.0
    for block_ordinal, row in enumerate(rows):
        path = args.sources / f"{row['id']}.bf16.bin"
        values = bf16_to_cupy(path)
        grouped = values.reshape(GROUPS_PER_BLOCK, GROUP_VALUES).astype(cp.float64)
        group_variance = cp.mean(grouped * grouped, axis=1)
        block_variance = float(cp.mean(group_variance).get())
        if not math.isfinite(block_variance) or block_variance <= 0.0:
            raise ValueError(f"invalid variance for {row['id']}: {block_variance}")
        serialized_rms = float(np.float32(math.sqrt(block_variance)))
        gv = cp.asnumpy(group_variance)
        labels = np.clip(
            np.rint(8.0 * np.log2(gv / block_variance)), -32.0, 31.0
        ).astype(np.int8)
        qscale = serialized_rms * np.exp2(labels.astype(np.float64) / 16.0)
        qvariance = np.square(qscale)
        block_rows.append(
            {
                "ordinal": block_ordinal,
                "id": row["id"],
                "tensor": row["tensor"],
                "block_index": int(row["block_index"]),
                "role": row["role"],
                "source_path": str(path),
                "source_sha256": sha256_path(path),
                "serialized_rms_fp32": serialized_rms,
                "labels_i8": labels.astype(int).tolist(),
            }
        )
        for group_index in range(GROUPS_PER_BLOCK):
            group_rows.append(
                {
                    "block_ordinal": block_ordinal,
                    "group_index": group_index,
                    "label": int(labels[group_index]),
                    "qscale": float(qscale[group_index]),
                    "qvariance": float(qvariance[group_index]),
                    "true_variance": float(gv[group_index]),
                }
            )
        source_gpu.append(values)
        total_energy += block_variance * BLOCK_VALUES

    qv = np.asarray([float(row["qvariance"]) for row in group_rows], dtype=np.float64)
    canonical = np.arange(qv.size, dtype=np.int64)
    order = np.lexsort((canonical, qv))
    if order.size != 400 * GROUPS_PER_BLOCK:
        raise AssertionError("group census mismatch")
    if order.size % GROUPS_PER_BLOCK:
        raise AssertionError("group count is not an integer number of polar blocks")

    chunks: list[dict[str, object]] = []
    total_nominal_rate_values = 0.0
    ideal_sse = 0.0
    lambda_variance = float(args.lambda_variance)
    if lambda_variance <= 0.0:
        raise ValueError("water level must be positive")
    for chunk_index, start in enumerate(range(0, order.size, GROUPS_PER_BLOCK)):
        members = order[start : start + GROUPS_PER_BLOCK]
        mean_qvariance = float(np.mean(qv[members]))
        unconstrained_rate = 0.5 * math.log2(mean_qvariance / lambda_variance)
        nominal_rate = min(args.max_rate, max(args.min_rate, unconstrained_rate))
        relative_d = 2.0 ** (-2.0 * nominal_rate)
        test_distortion = SIGMA_SOURCE**2 * relative_d
        tilde_sigma = math.sqrt(
            (SIGMA_SOURCE**2 - test_distortion) * test_distortion / SIGMA_SOURCE**2
        )
        eta = ETA_RATIO * tilde_sigma
        normalized_parts: list[cp.ndarray] = []
        member_rows: list[dict[str, object]] = []
        for member in members:
            group = group_rows[int(member)]
            block_ordinal = int(group["block_ordinal"])
            group_index = int(group["group_index"])
            begin = group_index * GROUP_VALUES
            end = begin + GROUP_VALUES
            normalized_parts.append(
                source_gpu[block_ordinal][begin:end].astype(cp.float64)
                / float(group["qscale"])
            )
            member_rows.append(
                {
                    "canonical_group_ordinal": int(member),
                    "block_ordinal": block_ordinal,
                    "group_index": group_index,
                    "label": int(group["label"]),
                    "qscale": float(group["qscale"]),
                    "qvariance": float(group["qvariance"]),
                }
            )
            # Normalization and inverse normalization cancel in the ideal
            # conditional-Gaussian projection, so the source's true group
            # variance—not its quantized scale proxy—weights distortion.
            ideal_sse += GROUP_VALUES * float(group["true_variance"]) * relative_d
        normalized = cp.concatenate(normalized_parts)
        output_path = source_dir / f"wf-{chunk_index:03d}.bf16.bin"
        cupy_to_bf16_rne(normalized).astype("<u2", copy=False).tofile(output_path)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "normalized_source": str(output_path),
                "normalized_source_sha256": sha256_path(output_path),
                "mean_qvariance": mean_qvariance,
                "unconstrained_rate": unconstrained_rate,
                "nominal_rate": nominal_rate,
                "test_distortion": test_distortion,
                "eta": eta,
                "alphabet_size": 128 if nominal_rate >= 2.75 else 64,
                "members": member_rows,
            }
        )
        total_nominal_rate_values += nominal_rate * BLOCK_VALUES

    panel_values = len(rows) * BLOCK_VALUES
    label_bits = len(group_rows) * 6
    block_scale_bits = len(rows) * 32
    lambda_bits = 32
    outer_header_bits = 128
    side_bits = label_bits + block_scale_bits + lambda_bits + outer_header_bits
    mean_nominal_rate = total_nominal_rate_values / panel_values
    ideal_relative_mse = ideal_sse / total_energy
    all_side_rate = mean_nominal_rate + side_bits / panel_values
    ideal_gap_db = 10.0 * math.log10(ideal_relative_mse / (2.0 ** (-2.0 * all_side_rate)))
    document = {
        "format": "continuous reverse-waterfilled PLTE exploratory manifest v1",
        "strict_ptq": True,
        "training_or_retraining": False,
        "parameters": {
            "block_values": BLOCK_VALUES,
            "group_values": GROUP_VALUES,
            "groups_per_polar_block": GROUPS_PER_BLOCK,
            "sigma_source": SIGMA_SOURCE,
            "eta_over_tilde_sigma": ETA_RATIO,
            "lambda_variance": lambda_variance,
            "min_rate": args.min_rate,
            "max_rate": args.max_rate,
            "stable_order": "ascending reconstructed qvariance, then canonical group ordinal",
        },
        "census": {
            "source_blocks": len(rows),
            "groups": len(group_rows),
            "polar_chunks": len(chunks),
            "values": panel_values,
        },
        "ideal_projection": {
            "source_energy": total_energy,
            "sse": ideal_sse,
            "relative_mse": ideal_relative_mse,
            "mean_nominal_signal_bpw": mean_nominal_rate,
            "panel_side_bits": side_bits,
            "panel_side_bpw": side_bits / panel_values,
            "all_side_bpw": all_side_rate,
            "gaussian_reference_gap_db": ideal_gap_db,
        },
        "side_ledger": {
            "six_bit_group_labels": label_bits,
            "fp32_original_block_rms": block_scale_bits,
            "fp32_water_level": lambda_bits,
            "outer_magic_version_counts": outer_header_bits,
            "note": "the existing single frozen six-level reliability mask is reused because eta/tilde_sigma is invariant",
        },
        "blocks": block_rows,
        "chunks": chunks,
    }
    atomic_json(args.output_dir / "manifest.json", document)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), **document["census"], **document["ideal_projection"]}, indent=2))


if __name__ == "__main__":
    main()
