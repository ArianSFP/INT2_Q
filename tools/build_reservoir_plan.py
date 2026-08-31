#!/usr/bin/env python3
"""Freeze a deterministic tier plan from the completed 81,242-byte cap audit.

The tool recognizes only the encoder's exact base-container overflow exception.
Successful Tier-0 artifacts are retained byte-for-byte; no quality result can
change the allocation. The generated plan must be committed before retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "qwen3_stratified_v1" / "manifest.json"
DEFAULT_WORKSPACE = ROOT / "tmp" / "qwen3_stratified_v1"
TIER0_BYTES = 81_242
TIER_STEP_BYTES = 64
TIER_MAP_BITS_PER_BLOCK = 4
MAX_TIER = 15
NONROUTER_BLOCKS = 116_422
BASE_LEDGER_BITS = 75_724_918_048
CHECKPOINT_PARAMETERS = 30_532_122_624
OVERFLOW_PATTERN = re.compile(
    r"RuntimeError: base polar container (?P<bytes>\d+) exceeds enforced "
    r"81242-byte slot"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def job_paths(workspace: Path, entry_id: str) -> tuple[Path, Path, Path]:
    report = workspace / "jobs" / f"{entry_id}.json"
    container = report.with_suffix(".polar.bin")
    log = workspace / "logs" / f"{entry_id}.encode.log"
    return report, container, log


def validate_tier0(
    report_path: Path,
    container_path: Path,
    entry: dict[str, Any],
    encoder_sha256: str,
) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("implementation_sha256") != encoder_sha256:
        raise AssertionError(f"encoder SHA mismatch for {entry['id']}")
    parameters = report.get("parameters", {})
    if parameters.get("container_cap_bytes") != TIER0_BYTES:
        raise AssertionError(f"non-Tier-0 successful report for {entry['id']}")
    trials = report.get("trials", [])
    if len(trials) != 1:
        raise AssertionError(f"unexpected trial count for {entry['id']}")
    trial = trials[0]
    if trial.get("source", {}).get("block_index") != 0:
        raise AssertionError(f"unexpected local source block for {entry['id']}")
    if container_path.stat().st_size != trial.get("literal_container_bytes"):
        raise AssertionError(f"container length mismatch for {entry['id']}")
    if sha256_path(container_path) != trial.get("literal_container_sha256"):
        raise AssertionError(f"container SHA mismatch for {entry['id']}")
    if container_path.stat().st_size > TIER0_BYTES:
        raise AssertionError(f"Tier-0 container overflow for {entry['id']}")
    required_true = (
        "arithmetic_roundtrip_bits_match",
        "online_causal_arithmetic_bits_match",
        "causal_decoder_frequencies_match",
        "causal_decoder_frozen_bits_match",
        "reconstruction_indices_match",
        "tail_escape_records_roundtrip",
        "tail_escape_padding_is_zero",
        "container_header_roundtrip",
        "passes_container_cap",
    )
    if any(trial.get(field) is not True for field in required_true):
        raise AssertionError(f"Tier-0 audit flag failed for {entry['id']}")
    return {
        "id": entry["id"],
        "layer": entry["layer"],
        "role": entry["role"],
        "tensor": entry["tensor"],
        "block_index": entry["block_index"],
        "first_pass_status": "fits_tier0",
        "tier": 0,
        "container_cap_bytes": TIER0_BYTES,
        "report_sha256": sha256_path(report_path),
        "container_bytes": container_path.stat().st_size,
        "container_sha256": sha256_path(container_path),
        "first_pass_gap_db": float(trial["gap_db"]),
        "first_pass_relative_mse": float(trial["relative_mse"]),
    }


def parse_overflow(
    log_path: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(OVERFLOW_PATTERN.finditer(text))
    if len(matches) != 1:
        raise AssertionError(
            f"missing or ambiguous recognized overflow for {entry['id']}"
        )
    base_bytes = int(matches[0].group("bytes"))
    if base_bytes <= TIER0_BYTES:
        raise AssertionError(f"invalid overflow length for {entry['id']}")
    tier = math.ceil((base_bytes - TIER0_BYTES) / TIER_STEP_BYTES)
    if tier < 1 or tier > MAX_TIER:
        raise AssertionError(
            f"required tier {tier} is outside the 4-bit reservoir map for {entry['id']}"
        )
    return {
        "id": entry["id"],
        "layer": entry["layer"],
        "role": entry["role"],
        "tensor": entry["tensor"],
        "block_index": entry["block_index"],
        "first_pass_status": "recognized_base_container_overflow",
        "tier": tier,
        "container_cap_bytes": TIER0_BYTES + tier * TIER_STEP_BYTES,
        "first_pass_base_container_bytes": base_bytes,
        "overflow_bytes_above_tier0": base_bytes - TIER0_BYTES,
        "first_pass_log_sha256": sha256_path(log_path),
        "retry_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    entries = manifest.get("plte_blocks", [])
    if len(entries) != 400:
        raise AssertionError("reservoir planning requires the complete frozen panel")
    encoder_sha256 = manifest["provenance"]["encoder_sha256"]
    allocations = []
    for entry in sorted(entries, key=lambda row: str(row["id"])):
        report, container, log = job_paths(args.workspace, entry["id"])
        if report.is_file() and container.is_file():
            allocations.append(
                validate_tier0(report, container, entry, encoder_sha256)
            )
        elif log.is_file():
            allocations.append(parse_overflow(log, entry))
        else:
            raise AssertionError(f"first-pass result is missing for {entry['id']}")

    if len(allocations) != 400:
        raise AssertionError("incomplete reservoir allocation")
    tier_counts = Counter(int(row["tier"]) for row in allocations)
    sum_tiers = sum(int(row["tier"]) for row in allocations)
    overflows = [row for row in allocations if int(row["tier"]) > 0]
    valid_gaps = [float(row["first_pass_gap_db"]) for row in allocations if row["tier"] == 0]
    tier_map_bits = TIER_MAP_BITS_PER_BLOCK * NONROUTER_BLOCKS
    bit_budget_below_2p5 = CHECKPOINT_PARAMETERS * 5 // 2
    max_sum_tiers = (
        bit_budget_below_2p5 - BASE_LEDGER_BITS - tier_map_bits
    ) // (8 * TIER_STEP_BYTES)

    def global_rate(tier_sum: int) -> float:
        return (
            BASE_LEDGER_BITS
            + tier_map_bits
            + 8 * TIER_STEP_BYTES * tier_sum
        ) / CHECKPOINT_PARAMETERS

    plan = {
        "format": "PLTE Qwen3 checkpoint rate-reservoir plan v1",
        "strict_ptq": True,
        "post_hoc_engineering_amendment": True,
        "selection_manifest_sha256": sha256_path(args.manifest),
        "checkpoint": manifest["checkpoint"],
        "encoder_sha256": encoder_sha256,
        "policy": {
            "tier0_bytes": TIER0_BYTES,
            "tier_step_bytes": TIER_STEP_BYTES,
            "tier_formula": "max(0, ceil((base_container_bytes - 81242) / 64))",
            "tier_map_bits_per_nonrouter_block": TIER_MAP_BITS_PER_BLOCK,
            "tier_map_order": "canonical checkpoint nonrouter block order",
            "maximum_representable_tier": MAX_TIER,
            "retry_trigger": "only the exact recognized base-container overflow exception",
            "successful_tier0_artifacts_are_reencoded": False,
            "quality_metrics_affect_tier_selection": False,
        },
        "first_pass": {
            "attempted_blocks": 400,
            "tier0_successes": tier_counts.get(0, 0),
            "recognized_overflows": len(overflows),
            "other_failures": 0,
            "tier_counts": {str(key): value for key, value in sorted(tier_counts.items())},
            "maximum_base_container_bytes": max(
                [TIER0_BYTES]
                + [int(row["first_pass_base_container_bytes"]) for row in overflows]
            ),
            "maximum_overflow_bytes": max(
                [0] + [int(row["overflow_bytes_above_tier0"]) for row in overflows]
            ),
            "maximum_valid_tier0_gap_db": max(valid_gaps),
        },
        "checkpoint_rate_accounting": {
            "base_ledger_bits": BASE_LEDGER_BITS,
            "checkpoint_parameters": CHECKPOINT_PARAMETERS,
            "nonrouter_blocks": NONROUTER_BLOCKS,
            "tier_map_bits": tier_map_bits,
            "bits_per_tier_increment": 8 * TIER_STEP_BYTES,
            "strict_sum_tiers_limit_below_2p5": max_sum_tiers,
            "rate_with_no_overflow_tiers_bpw": global_rate(0),
            "rate_if_sample_tier_sum_were_checkpoint_total_bpw": global_rate(sum_tiers),
            "rate_if_every_nonrouter_block_were_tier1_bpw": global_rate(
                NONROUTER_BLOCKS
            ),
            "rate_if_every_nonrouter_block_were_tier10_bpw": global_rate(
                10 * NONROUTER_BLOCKS
            ),
            "below_2p5_for_every_block_at_tier10": global_rate(
                10 * NONROUTER_BLOCKS
            )
            < 2.5,
        },
        "claim_boundary": (
            "Amended after observing first-pass cap failures; broad engineering "
            "evidence on the frozen panel, not an untouched confirmatory holdout "
            "or a measured whole-checkpoint tier distribution"
        ),
        "allocations": allocations,
    }
    atomic_json(args.output, plan)
    print(json.dumps({key: value for key, value in plan.items() if key != "allocations"}, indent=2))


if __name__ == "__main__":
    main()
