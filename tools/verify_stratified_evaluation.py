#!/usr/bin/env python3
"""Verify the frozen Qwen3 stratified panel and its source-free evidence.

This verifier deliberately uses only the Python standard library.  In
``--manifest-only`` mode it rebuilds the preregistered selection from the
published safetensors headers and earlier evidence identifiers.  In full mode
it additionally verifies every finalized artifact, including the concatenated
container bundle and all recomputed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
DEFAULT_DIRECTORY = ROOT / "evaluation" / "qwen3_stratified_v1"
DEFAULT_MANIFEST = DEFAULT_DIRECTORY / "manifest.json"
DEFAULT_RESERVOIR_PLAN = DEFAULT_DIRECTORY / "reservoir_plan.json"
DEFAULT_HEADERS = PLTE / "qwen_weight_cache" / "headers"
DEFAULT_EVIDENCE = PLTE / "agent_root_polar_escape_evidence_manifest.json"
BUILDER = ROOT / "tools" / "build_stratified_manifest.py"
RUNNER = ROOT / "tools" / "run_stratified_evaluation.py"
ENCODER = PLTE / "agent_root_polar_lattice_gate.py"
DECODER = PLTE / "agent_polar_codec_audit_independent_decoder.py"
FETCHER = PLTE / "agent_root_fetch_qwen_block.py"
PROFILE = PLTE / "agent_root_polar_escape_frozen_profiles.bin"
ROUTER_REPORT = PLTE / "agent_router_adaptive_q4_t0045_all48.json"
ROUTER_CONTAINER = PLTE / "agent_router_adaptive_q4_t0045_all48.bin"

BLOCK_VALUES = 1 << 18
CONTAINER_CAP_BYTES = 81_242
TIER_STEP_BYTES = 64
TIER_MAP_BITS_PER_BLOCK = 4
MAX_TIER = 15
NONROUTER_BLOCKS = 116_422
BASE_LEDGER_BITS = 75_724_918_048
CHECKPOINT_PARAMETERS = 30_532_122_624
EXPECTED_CHECKPOINT = {
    "repo": "Qwen/Qwen3-30B-A3B",
    "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
}
LAYER_ROLES = (
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.experts.{expert}.down_proj.weight",
    "mlp.experts.{expert}.gate_proj.weight",
    "mlp.experts.{expert}.up_proj.weight",
)
GLOBAL_ROLE_COUNTS = {
    "model.embed_tokens.weight": 32,
    "lm_head.weight": 32,
}
ENCODER_TRUE_FLAGS = (
    "fp32_decoder_scale_in_mse_audit",
    "arithmetic_roundtrip_bits_match",
    "online_causal_arithmetic_bits_match",
    "causal_decoder_frequencies_match",
    "causal_decoder_frozen_bits_match",
    "reconstruction_indices_match",
    "tail_escape_records_roundtrip",
    "tail_escape_padding_is_zero",
    "container_header_roundtrip",
    "passes_container_cap",
    "passes_rate_lt_2p5",
)
EXPECTED_BASE_PARAMETERS = {
    "block_length": BLOCK_VALUES,
    "trials": 1,
    "sigma_source": 3.0,
    "test_channel_distortion": 0.29,
    "eta": 0.5989929996555583,
    "alphabet_size": 64,
    "decision": "map",
    "tilde_sigma": 0.5297693418418581,
    "capacity_schedule": [
        0.0006403541494273135,
        0.2226280511277603,
        0.906837113158238,
        0.9999736826737476,
        1.0,
        1.0,
    ],
    "seed": 20260831,
}
EXPECTED_CLAIM_BOUNDARY = (
    "Measured on the frozen 400-block panel with a post-hoc deterministic "
    "rate-reservoir amendment, plus complete router and rank-one exception "
    "censuses; not an untouched confirmatory holdout, whole-checkpoint "
    "distortion measurement, or worst-case guarantee"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class VerificationError(AssertionError):
    """A concise, user-facing evidence verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read valid JSON from {path}: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise VerificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(value, indent=2, allow_nan=False) + "\n"
    return sha256_bytes(rendered.encode("utf-8"))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(not isinstance(value, bool), f"{label} is Boolean, not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not numeric") from error
    require(math.isfinite(number), f"{label} is non-finite")
    if positive:
        require(number > 0.0, f"{label} must be positive")
    return number


def close(actual: Any, expected: Any, label: str, *, tolerance: float = 2e-12) -> None:
    a = finite_number(actual, label)
    e = finite_number(expected, f"expected {label}")
    require(
        math.isclose(a, e, rel_tol=tolerance, abs_tol=tolerance),
        f"{label} mismatch: {a!r} != {e!r}",
    )


def deep_close(actual: Any, expected: Any, label: str = "value") -> None:
    """Compare a JSON-like object, allowing only tiny finite-float drift."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label} is not an object")
        require(
            set(actual) == set(expected),
            f"{label} keys mismatch: missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}",
        )
        for key in expected:
            deep_close(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        require(isinstance(actual, list), f"{label} is not an array")
        require(len(actual) == len(expected), f"{label} length mismatch")
        for index, (a_value, e_value) in enumerate(zip(actual, expected)):
            deep_close(a_value, e_value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected and type(actual) is type(expected), f"{label} mismatch")
        return
    if isinstance(expected, float) or isinstance(actual, float):
        close(actual, expected, label)
        return
    require(actual == expected and type(actual) is type(expected), f"{label} mismatch")


def require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} is not an object")
    require(
        set(value) == expected,
        f"{label} keys mismatch: missing={sorted(expected - set(value))}, "
        f"extra={sorted(set(value) - expected)}",
    )
    return value


def import_builder() -> ModuleType:
    require(BUILDER.is_file(), f"missing manifest builder: {BUILDER}")
    name = "_plte_frozen_manifest_builder"
    spec = importlib.util.spec_from_file_location(name, BUILDER)
    require(spec is not None and spec.loader is not None, "cannot load manifest builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_range(row: dict[str, Any], expected_bytes: int, label: str) -> None:
    byte_range = row.get("absolute_byte_range_in_shard")
    require(
        isinstance(byte_range, list)
        and len(byte_range) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in byte_range),
        f"{label} has an invalid byte range",
    )
    require(byte_range[0] >= 0 and byte_range[1] >= byte_range[0], f"{label} range is invalid")
    require(byte_range[1] - byte_range[0] + 1 == expected_bytes, f"{label} range length mismatch")


def validate_manifest(
    path: Path, headers: Path, evidence: Path
) -> tuple[dict[str, Any], str]:
    manifest = load_json(path)
    require(manifest.get("format") == "PLTE Qwen3 stratified evaluation manifest v1", "manifest format mismatch")
    require(manifest.get("checkpoint") == EXPECTED_CHECKPOINT, "manifest checkpoint mismatch")
    require(manifest.get("strict_ptq") is True, "manifest is not declared strict PTQ")
    require(manifest.get("selection_reads_weight_payloads") is False, "selection is not weight blind")

    builder = import_builder()
    try:
        catalog, header_digest = builder.load_headers(headers)
        rebuilt = builder.build_manifest(catalog, header_digest, evidence)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        raise VerificationError(f"manifest rebuild failed: {error}") from error
    require(manifest == rebuilt, "committed manifest is not exactly reproducible from headers and prior identifiers")

    rows = manifest.get("plte_blocks")
    require(isinstance(rows, list) and len(rows) == 400, "manifest must contain 400 PLTE rows")
    ids = [row.get("id") for row in rows]
    identities = [(row.get("tensor"), row.get("block_index")) for row in rows]
    require(len(set(ids)) == 400 and all(isinstance(item, str) and item for item in ids), "PLTE IDs are not unique")
    require(len(set(identities)) == 400, "PLTE tensor/block identities are not unique")

    layer_cells: dict[tuple[int, str], int] = {}
    global_counts = {role: 0 for role in GLOBAL_ROLE_COUNTS}
    for row in rows:
        label = f"PLTE row {row['id']}"
        require(row.get("dtype") == "BF16", f"{label} is not BF16")
        require(row.get("source_values") == BLOCK_VALUES, f"{label} value count mismatch")
        require(row.get("source_bytes") == 2 * BLOCK_VALUES, f"{label} byte count mismatch")
        validate_range(row, 2 * BLOCK_VALUES, label)
        layer = row.get("layer")
        role = row.get("role")
        if layer is None:
            require(role in global_counts, f"{label} has an unexpected global role")
            global_counts[role] += 1
        else:
            require(isinstance(layer, int) and not isinstance(layer, bool) and 0 <= layer < 48, f"{label} layer is invalid")
            require(role in LAYER_ROLES, f"{label} layer role is invalid")
            key = (layer, role)
            layer_cells[key] = layer_cells.get(key, 0) + 1
    expected_cells = {(layer, role): 1 for layer in range(48) for role in LAYER_ROLES}
    require(layer_cells == expected_cells, "manifest does not contain the complete 48-by-7 layer-role grid")
    require(global_counts == GLOBAL_ROLE_COUNTS, "manifest global-role counts mismatch")

    routers = manifest.get("router_blocks")
    require(isinstance(routers, list) and len(routers) == 48, "router census is incomplete")
    require({row.get("layer") for row in routers} == set(range(48)), "router layers are incomplete")
    for row in routers:
        label = f"router row {row.get('id')}"
        require(row.get("role") == "mlp.gate.weight", f"{label} role mismatch")
        require(row.get("block_index") == 0 and row.get("source_values") == BLOCK_VALUES, f"{label} block mismatch")
        validate_range(row, 2 * BLOCK_VALUES, label)

    rank1 = manifest.get("rank1_tensors")
    require(isinstance(rank1, list) and len(rank1) == 193, "rank-one census is incomplete")
    require(len({row.get("id") for row in rank1}) == 193, "rank-one IDs are not unique")
    require(sum(int(row.get("values", -1)) for row in rank1) == 210_944, "rank-one value census mismatch")
    for row in rank1:
        label = f"rank-one row {row.get('id')}"
        values = row.get("values")
        require(isinstance(values, int) and not isinstance(values, bool) and values > 0, f"{label} values invalid")
        require(row.get("shape") == [values], f"{label} is not rank one")
        require(row.get("dtype") == "BF16" and row.get("bytes") == 2 * values, f"{label} BF16 size mismatch")
        require(row.get("codec") == "lossless raw BF16 exception", f"{label} codec mismatch")
        validate_range(row, 2 * values, label)

    selection = manifest.get("selection", {})
    require(selection.get("new_plte_blocks") == 400, "selection count mismatch")
    require(selection.get("layer_role_blocks") == 336, "layer-role selection count mismatch")
    require(selection.get("embedding_blocks") == 32 and selection.get("lm_head_blocks") == 32, "global selection count mismatch")
    require(selection.get("complete_router_blocks_reused") == 48, "router selection count mismatch")
    require(selection.get("complete_rank1_tensors") == 193, "rank-one selection count mismatch")
    require(selection.get("published_plte_blocks_excluded") == 47, "prior evidence exclusion count mismatch")

    provenance = manifest.get("provenance", {})
    pinned_paths = {
        "published_evidence_manifest_sha256": evidence,
        "encoder_sha256": ENCODER,
        "independent_decoder_sha256": DECODER,
        "fetcher_sha256": FETCHER,
        "frozen_profile_sha256": PROFILE,
    }
    for key, pinned_path in pinned_paths.items():
        require(provenance.get(key) == sha256_path(pinned_path), f"manifest {key} does not match repository bytes")

    manifest_sha = sha256_path(path)
    print(
        f"manifest {manifest_sha}: reproducible 400-block panel, "
        "48x7 layer-role grid, 48 routers, 193 rank-one tensors"
    )
    return manifest, manifest_sha


def validate_reservoir_plan(
    path: Path, manifest: dict[str, Any], manifest_sha: str
) -> tuple[dict[str, Any], str]:
    plan = require_keys(
        load_json(path),
        {
            "format", "strict_ptq", "post_hoc_engineering_amendment",
            "selection_manifest_sha256", "checkpoint", "encoder_sha256",
            "policy", "first_pass", "checkpoint_rate_accounting",
            "claim_boundary", "allocations",
        },
        "reservoir_plan.json",
    )
    require(plan["format"] == "PLTE Qwen3 checkpoint rate-reservoir plan v1", "reservoir plan format mismatch")
    require(plan["strict_ptq"] is True, "reservoir plan is not strict PTQ")
    require(plan["post_hoc_engineering_amendment"] is True, "reservoir amendment is not disclosed")
    require(plan["selection_manifest_sha256"] == manifest_sha, "reservoir plan targets another manifest")
    require(plan["checkpoint"] == manifest["checkpoint"], "reservoir plan checkpoint mismatch")
    require(plan["encoder_sha256"] == manifest["provenance"]["encoder_sha256"] == sha256_path(ENCODER), "reservoir encoder hash mismatch")
    expected_policy = {
        "tier0_bytes": CONTAINER_CAP_BYTES,
        "tier_step_bytes": TIER_STEP_BYTES,
        "tier_formula": "max(0, ceil((base_container_bytes - 81242) / 64))",
        "tier_map_bits_per_nonrouter_block": TIER_MAP_BITS_PER_BLOCK,
        "tier_map_order": "canonical checkpoint nonrouter block order",
        "maximum_representable_tier": MAX_TIER,
        "retry_trigger": "only the exact recognized base-container overflow exception",
        "successful_tier0_artifacts_are_reencoded": False,
        "quality_metrics_affect_tier_selection": False,
    }
    deep_close(plan["policy"], expected_policy, "reservoir plan policy")

    allocations = plan["allocations"]
    require(isinstance(allocations, list) and len(allocations) == 400, "reservoir plan must allocate 400 rows")
    require(allocations == sorted(allocations, key=lambda row: str(row.get("id"))), "reservoir allocations are not ID-sorted")
    manifest_by_id = {row["id"]: row for row in manifest["plte_blocks"]}
    require({row.get("id") for row in allocations} == set(manifest_by_id), "reservoir allocation IDs mismatch")
    tier_counts: dict[int, int] = {}
    tier0_gaps: list[float] = []
    overflows: list[dict[str, Any]] = []
    common_keys = {"id", "layer", "role", "tensor", "block_index", "first_pass_status", "tier", "container_cap_bytes"}
    tier0_keys = common_keys | {
        "report_sha256", "container_bytes", "container_sha256",
        "first_pass_gap_db", "first_pass_relative_mse",
    }
    overflow_keys = common_keys | {
        "first_pass_base_container_bytes", "overflow_bytes_above_tier0",
        "first_pass_log_sha256", "retry_required",
    }
    for allocation in allocations:
        entry = manifest_by_id[allocation["id"]]
        tier = allocation.get("tier")
        require(isinstance(tier, int) and not isinstance(tier, bool) and 0 <= tier <= MAX_TIER, f"invalid reservoir tier for {allocation['id']}")
        require(allocation.get("container_cap_bytes") == CONTAINER_CAP_BYTES + tier * TIER_STEP_BYTES, f"reservoir cap formula mismatch for {allocation['id']}")
        for key in ("layer", "role", "tensor", "block_index"):
            require(allocation.get(key) == entry[key], f"reservoir identity mismatch for {allocation['id']}: {key}")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if tier == 0:
            require_keys(allocation, tier0_keys, f"Tier-0 allocation {allocation['id']}")
            require(allocation["first_pass_status"] == "fits_tier0", f"Tier-0 status mismatch for {allocation['id']}")
            require(is_sha256(allocation["report_sha256"]) and is_sha256(allocation["container_sha256"]), f"Tier-0 hashes malformed for {allocation['id']}")
            require(isinstance(allocation["container_bytes"], int) and 0 < allocation["container_bytes"] <= CONTAINER_CAP_BYTES, f"Tier-0 size invalid for {allocation['id']}")
            tier0_gaps.append(finite_number(allocation["first_pass_gap_db"], f"Tier-0 gap {allocation['id']}"))
            finite_number(allocation["first_pass_relative_mse"], f"Tier-0 MSE {allocation['id']}", positive=True)
        else:
            require_keys(allocation, overflow_keys, f"overflow allocation {allocation['id']}")
            require(allocation["first_pass_status"] == "recognized_base_container_overflow" and allocation["retry_required"] is True, f"overflow status mismatch for {allocation['id']}")
            base_bytes = allocation["first_pass_base_container_bytes"]
            require(isinstance(base_bytes, int) and base_bytes > CONTAINER_CAP_BYTES, f"overflow base length invalid for {allocation['id']}")
            require(allocation["overflow_bytes_above_tier0"] == base_bytes - CONTAINER_CAP_BYTES, f"overflow delta mismatch for {allocation['id']}")
            require(tier == math.ceil((base_bytes - CONTAINER_CAP_BYTES) / TIER_STEP_BYTES), f"overflow tier mismatch for {allocation['id']}")
            require(is_sha256(allocation["first_pass_log_sha256"]), f"overflow log hash malformed for {allocation['id']}")
            overflows.append(allocation)

    first_pass_expected = {
        "attempted_blocks": 400,
        "tier0_successes": tier_counts.get(0, 0),
        "recognized_overflows": len(overflows),
        "other_failures": 0,
        "tier_counts": {str(key): value for key, value in sorted(tier_counts.items())},
        "maximum_base_container_bytes": max([CONTAINER_CAP_BYTES] + [int(row["first_pass_base_container_bytes"]) for row in overflows]),
        "maximum_overflow_bytes": max([0] + [int(row["overflow_bytes_above_tier0"]) for row in overflows]),
        "maximum_valid_tier0_gap_db": max(tier0_gaps),
    }
    deep_close(plan["first_pass"], first_pass_expected, "reservoir first pass")

    tier_map_bits = TIER_MAP_BITS_PER_BLOCK * NONROUTER_BLOCKS
    bit_budget_below_2p5 = CHECKPOINT_PARAMETERS * 5 // 2
    max_sum_tiers = (bit_budget_below_2p5 - BASE_LEDGER_BITS - tier_map_bits) // (8 * TIER_STEP_BYTES)
    sum_tiers = sum(tier * count for tier, count in tier_counts.items())

    def global_rate(tier_sum: int) -> float:
        return (BASE_LEDGER_BITS + tier_map_bits + 8 * TIER_STEP_BYTES * tier_sum) / CHECKPOINT_PARAMETERS

    accounting_expected = {
        "base_ledger_bits": BASE_LEDGER_BITS,
        "checkpoint_parameters": CHECKPOINT_PARAMETERS,
        "nonrouter_blocks": NONROUTER_BLOCKS,
        "tier_map_bits": tier_map_bits,
        "bits_per_tier_increment": 8 * TIER_STEP_BYTES,
        "strict_sum_tiers_limit_below_2p5": max_sum_tiers,
        "rate_with_no_overflow_tiers_bpw": global_rate(0),
        "rate_if_sample_tier_sum_were_checkpoint_total_bpw": global_rate(sum_tiers),
        "rate_if_every_nonrouter_block_were_tier1_bpw": global_rate(NONROUTER_BLOCKS),
        "rate_if_every_nonrouter_block_were_tier10_bpw": global_rate(10 * NONROUTER_BLOCKS),
        "below_2p5_for_every_block_at_tier10": global_rate(10 * NONROUTER_BLOCKS) < 2.5,
    }
    deep_close(plan["checkpoint_rate_accounting"], accounting_expected, "reservoir checkpoint accounting")
    expected_claim = (
        "Amended after observing first-pass cap failures; broad engineering "
        "evidence on the frozen panel, not an untouched confirmatory holdout "
        "or a measured whole-checkpoint tier distribution"
    )
    require(plan["claim_boundary"] == expected_claim, "reservoir plan claim boundary is overstated")
    plan_sha = sha256_path(path)
    print(
        f"reservoir {plan_sha}: {tier_counts.get(0, 0)} Tier-0 and "
        f"{len(overflows)} overflow-tier allocations, post-hoc amendment disclosed"
    )
    return plan, plan_sha


def expected_original_tier0_outcome(
    directory: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
    reservoir_plan: dict[str, Any],
    reservoir_plan_sha: str,
) -> dict[str, Any]:
    allocation_by_id = {
        row["id"]: row for row in reservoir_plan["allocations"]
    }
    failures = []
    failure_log_directory = directory / "original_tier0_failure_logs"
    expected_log_names = {
        f"{row['id']}.txt"
        for row in reservoir_plan["allocations"]
        if int(row["tier"]) > 0
    }
    require(failure_log_directory.is_dir(), "original Tier-0 failure-log directory is missing")
    actual_log_names = {
        path.name for path in failure_log_directory.iterdir() if path.is_file()
    }
    require(actual_log_names == expected_log_names, "original Tier-0 failure-log set is stale or incomplete")
    for entry in sorted(manifest["plte_blocks"], key=lambda row: str(row["id"])):
        allocation = allocation_by_id[entry["id"]]
        if int(allocation["tier"]) == 0:
            continue
        relative_log = f"original_tier0_failure_logs/{entry['id']}.txt"
        log_path = directory / relative_log
        log_sha = sha256_path(log_path)
        require(log_sha == allocation["first_pass_log_sha256"], f"published first-pass log hash mismatch for {entry['id']}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(
            r"RuntimeError: base polar container (\d+) exceeds enforced "
            r"81242-byte slot",
            text,
        )
        require(len(matches) == 1 and int(matches[0]) == allocation["first_pass_base_container_bytes"], f"published first-pass log does not prove the frozen overflow for {entry['id']}")
        failures.append(
            {
                "id": entry["id"],
                "tensor": entry["tensor"],
                "block_index": entry["block_index"],
                "layer": entry["layer"],
                "role": entry["role"],
                "failure": "recognized_base_container_overflow",
                "tier0_cap_bytes": reservoir_plan["policy"]["tier0_bytes"],
                "base_container_bytes": allocation[
                    "first_pass_base_container_bytes"
                ],
                "overflow_bytes": allocation["overflow_bytes_above_tier0"],
                "first_pass_log_sha256": allocation["first_pass_log_sha256"],
                "published_log": relative_log,
                "published_log_sha256": log_sha,
            }
        )
    first_pass = reservoir_plan["first_pass"]
    return {
        "format": "PLTE Qwen3 original Tier-0 outcome v1",
        "selection_manifest_sha256": manifest_sha,
        "reservoir_plan_sha256": reservoir_plan_sha,
        "endpoint": "every selected block fits the original 81,242-byte cap",
        "passes": len(failures) == 0,
        "attempted_blocks": 400,
        "tier0_successes": first_pass["tier0_successes"],
        "recognized_cap_failures": len(failures),
        "other_failures": first_pass["other_failures"],
        "maximum_base_container_bytes": first_pass[
            "maximum_base_container_bytes"
        ],
        "maximum_overflow_bytes": first_pass["maximum_overflow_bytes"],
        "maximum_valid_tier0_gap_db": first_pass[
            "maximum_valid_tier0_gap_db"
        ],
        "failures": failures,
        "claim_boundary": (
            "This is the immutable outcome of the original fixed-cap endpoint. "
            "Later reservoir retries do not reclassify these failures."
        ),
    }


def validate_source_hashes(
    document: Any, manifest: dict[str, Any], manifest_sha: str
) -> dict[str, dict[str, Any]]:
    document = require_keys(
        document,
        {"format", "selection_manifest_sha256", "checkpoint", "records", "raw_source_bytes_are_publishable"},
        "source_hashes.json",
    )
    require(document["format"] == "PLTE Qwen3 fetched source hashes v1", "source hash format mismatch")
    require(document["selection_manifest_sha256"] == manifest_sha, "source hash manifest mismatch")
    require(document["checkpoint"] == manifest["checkpoint"], "source hash checkpoint mismatch")
    require(document["raw_source_bytes_are_publishable"] is False, "raw source publication boundary changed")
    rows = document["records"]
    require(isinstance(rows, list) and len(rows) == 593, "source hash manifest must contain 593 records")
    require(rows == sorted(rows, key=lambda row: str(row.get("id"))), "source hash records are not ID-sorted")
    require(len({row.get("id") for row in rows}) == 593, "source hash IDs are not unique")

    entries: dict[str, tuple[dict[str, Any], str]] = {}
    for row in manifest["plte_blocks"]:
        entries[row["id"]] = (row, "plte_block")
    for row in manifest["rank1_tensors"]:
        entries[row["id"]] = (row, "rank1_exact_bf16")
    require(set(entries) == {row.get("id") for row in rows}, "source hash IDs do not match manifest")

    output = {}
    expected_keys = {
        "id", "kind", "repo", "revision", "tensor", "block_index", "shard",
        "absolute_byte_range_in_shard", "bytes", "sha256",
    }
    for row in rows:
        row = require_keys(row, expected_keys, f"source record {row.get('id')}")
        entry, kind = entries[row["id"]]
        expected_bytes = int(entry.get("source_bytes", entry.get("bytes")))
        require(row["kind"] == kind, f"source kind mismatch for {row['id']}")
        require(row["repo"] == manifest["checkpoint"]["repo"] and row["revision"] == manifest["checkpoint"]["revision"], f"source checkpoint mismatch for {row['id']}")
        require(row["tensor"] == entry["tensor"] and row["block_index"] == entry.get("block_index"), f"source identity mismatch for {row['id']}")
        require(row["shard"] == entry["shard"] and row["absolute_byte_range_in_shard"] == entry["absolute_byte_range_in_shard"], f"source range mismatch for {row['id']}")
        require(row["bytes"] == expected_bytes and is_sha256(row["sha256"]), f"source size/hash invalid for {row['id']}")
        output[row["id"]] = row
    return output


def validate_encoder_report(
    report: Any,
    result: dict[str, Any],
    entry: dict[str, Any],
    source_sha: str,
    encoder_sha: str,
    charged_slot_bytes: int,
) -> dict[str, float | int | str]:
    label = f"encoder report {entry['id']}"
    require(isinstance(report, dict), f"{label} is not an object")
    require(report.get("strict_ptq") is True, f"{label} is not strict PTQ")
    require(report.get("source_training_or_retraining") is False, f"{label} reports training")
    require(report.get("implementation_sha256") == encoder_sha, f"{label} encoder hash mismatch")
    expected_parameters = {
        **EXPECTED_BASE_PARAMETERS,
        "container_cap_bytes": charged_slot_bytes,
    }


    deep_close(report.get("parameters"), expected_parameters, f"{label}.parameters")
    trials = report.get("trials")
    require(isinstance(trials, list) and len(trials) == 1, f"{label} must contain one trial")
    trial = trials[0]
    require(isinstance(trial, dict), f"{label} trial is invalid")
    source = trial.get("source")
    require(isinstance(source, dict), f"{label} source is invalid")
    require(source.get("kind") == "frozen_bf16_weight_block", f"{label} source kind mismatch")
    require(source.get("block_index") == 0 and source.get("values") == BLOCK_VALUES, f"{label} source block mismatch")
    require(source.get("block_bf16_sha256") == source_sha, f"{label} source hash mismatch")
    rms = finite_number(source.get("block_rms_fp64"), f"{label} source RMS", positive=True)
    finite_number(source.get("decoder_scale_fp32"), f"{label} decoder scale", positive=True)

    for flag in ENCODER_TRUE_FLAGS:
        require(trial.get(flag) is True, f"{label} integrity flag is not true: {flag}")
    require(isinstance(trial.get("passes_gap_lt_0p10db"), bool), f"{label} quality flag is not Boolean")
    require(trial.get("container_cap_bytes") == charged_slot_bytes, f"{label} cap mismatch")
    container_bytes = result["container_bytes"]
    require(isinstance(container_bytes, int) and 0 < container_bytes <= charged_slot_bytes, f"{label} container size invalid")
    require(trial.get("literal_container_bytes") == container_bytes, f"{label} container length mismatch")
    require(trial.get("literal_container_sha256") == result["container_sha256"], f"{label} container hash mismatch")
    base_bytes = trial.get("base_literal_container_bytes")
    require(isinstance(base_bytes, int) and 0 < base_bytes <= container_bytes, f"{label} base container exceeds final container")
    require(trial.get("framing_bits") == 64, f"{label} framing mismatch")
    require(trial.get("total_screen_bits") == 8 * container_bytes, f"{label} screen bit count mismatch")

    absolute_mse = finite_number(trial.get("literal_decoded_absolute_mse"), f"{label} absolute MSE", positive=True)
    relative_mse = finite_number(trial.get("relative_mse"), f"{label} relative MSE", positive=True)
    close(trial.get("literal_decoded_relative_mse"), relative_mse, f"{label} literal relative MSE")
    energy = rms * rms * BLOCK_VALUES
    sse = absolute_mse * BLOCK_VALUES
    close(relative_mse, sse / energy, f"{label} relative-MSE formula")
    rate = 8 * container_bytes / BLOCK_VALUES
    close(trial.get("screen_bpw"), rate, f"{label} rate")
    gaussian = 2.0 ** (-2.0 * rate)
    gap = 10.0 * math.log10(relative_mse / gaussian)
    close(trial.get("gaussian_limit_mse_at_screen_rate"), gaussian, f"{label} Gaussian reference")
    close(trial.get("threshold_mse_0p10db"), gaussian * 10.0 ** 0.01, f"{label} threshold")
    close(trial.get("gap_db"), gap, f"{label} gap")
    require(trial["passes_gap_lt_0p10db"] is (gap < 0.10), f"{label} quality flag mismatch")

    aggregate = report.get("aggregate")
    require(isinstance(aggregate, dict), f"{label} aggregate is invalid")
    close(aggregate.get("mean_relative_mse"), relative_mse, f"{label} aggregate MSE")
    close(aggregate.get("mean_screen_bpw"), rate, f"{label} aggregate rate")
    close(aggregate.get("gaussian_limit_mse"), gaussian, f"{label} aggregate Gaussian reference")
    close(aggregate.get("threshold_mse_0p10db"), gaussian * 10.0 ** 0.01, f"{label} aggregate threshold")
    close(aggregate.get("gap_db"), gap, f"{label} aggregate gap")
    require(aggregate.get("passes_rate_lt_2p5") is (rate < 2.5), f"{label} aggregate rate flag mismatch")
    require(aggregate.get("passes_gap_lt_0p10db") is (gap < 0.10), f"{label} aggregate quality flag mismatch")
    return {
        "id": entry["id"],
        "layer": entry["layer"],
        "role": entry["role"],
        "source_energy": energy,
        "sse": sse,
        "relative_mse": relative_mse,
        "container_bytes": container_bytes,
        "actual_bpw": rate,
        "actual_gap_db": gap,
        "reservoir_tier": result["reservoir_tier"],
        "charged_slot_bytes": charged_slot_bytes,
        "charged_slot_bits_including_4bit_map": (
            8 * charged_slot_bytes + TIER_MAP_BITS_PER_BLOCK
        ),
        "charged_slot_bpw_including_4bit_map": (
            (8 * charged_slot_bytes + TIER_MAP_BITS_PER_BLOCK) / BLOCK_VALUES
        ),
        "charged_tier_slot_gap_db": 10.0 * math.log10(
            relative_mse
            / (
                2.0
                ** (
                    -2.0
                    * (
                        (8 * charged_slot_bytes + TIER_MAP_BITS_PER_BLOCK)
                        / BLOCK_VALUES
                    )
                )
            )
        ),
    }


def infer_plte_prefix_length(slot: bytes, label: str) -> tuple[int, int, int]:
    require(len(slot) >= 8, f"{label} is shorter than the PLTE header")
    header_word = struct.unpack_from("<I", slot, 0)[0]
    logical_bits = header_word & ((1 << 20) - 1)
    escape_count = header_word >> 20
    require(logical_bits > 0, f"{label} has a zero arithmetic length")
    arithmetic_bytes = (logical_bits + 7) // 8
    escape_bits = 34 * escape_count
    escape_bytes = (escape_bits + 7) // 8
    literal_bytes = 8 + arithmetic_bytes + escape_bytes
    require(literal_bytes <= len(slot), f"{label} header length exceeds its slot")
    if logical_bits % 8:
        padding_mask = (1 << (8 - logical_bits % 8)) - 1
        require(
            slot[8 + arithmetic_bytes - 1] & padding_mask == 0,
            f"{label} arithmetic padding bits are non-zero",
        )
    if escape_bits % 8:
        padding_mask = (1 << (8 - escape_bits % 8)) - 1
        require(
            slot[literal_bytes - 1] & padding_mask == 0,
            f"{label} tail padding bits are non-zero",
        )
    return literal_bytes, logical_bits, escape_count


def audit_packed_artifacts(
    bundle: bytes,
    tier_map: bytes,
    slots: bytes,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(tier_map) == (len(rows) + 1) // 2, "tier-map byte length mismatch")
    bundle_offset = 0
    slot_offset = 0
    for index, row in enumerate(rows):
        tier_byte = tier_map[index // 2]
        decoded_tier = (
            tier_byte & 0x0F if index % 2 == 0 else (tier_byte >> 4) & 0x0F
        )
        tier = int(row["reservoir_tier"])
        require(decoded_tier == tier, f"tier-map extraction mismatch for {row['id']}")
        charged_bytes = CONTAINER_CAP_BYTES + TIER_STEP_BYTES * tier
        require(row["charged_slot_bytes"] == charged_bytes, f"charged-slot formula mismatch for {row['id']}")
        require(row["container_offset"] == bundle_offset, f"bundle offset mismatch for {row['id']}")
        require(row["tiered_slot_offset"] == slot_offset, f"slot offset mismatch for {row['id']}")
        slot = slots[slot_offset : slot_offset + charged_bytes]
        require(len(slot) == charged_bytes, f"short tiered slot for {row['id']}")
        literal_bytes, logical_bits, escape_count = infer_plte_prefix_length(
            slot, f"tiered slot {row['id']}"
        )
        require(literal_bytes == row["container_bytes"], f"header-derived length mismatch for {row['id']}")
        trial = row["report"]["trials"][0]
        require(logical_bits == trial.get("arithmetic_logical_bits"), f"packed arithmetic length mismatch for {row['id']}")
        require(escape_count == trial.get("tail_escape_count"), f"packed escape count mismatch for {row['id']}")
        prefix = slot[:literal_bytes]
        segment = bundle[bundle_offset : bundle_offset + literal_bytes]
        require(prefix == segment and sha256_bytes(prefix) == row["container_sha256"], f"packed prefix/hash mismatch for {row['id']}")
        require(not any(slot[literal_bytes:]), f"non-zero slot padding for {row['id']}")
        require(row["zero_padding_bytes"] == charged_bytes - literal_bytes, f"recorded padding mismatch for {row['id']}")
        bundle_offset += literal_bytes
        slot_offset += charged_bytes
    require(bundle_offset == len(bundle), "container bundle has unreferenced trailing bytes")
    require(slot_offset == len(slots), "tiered-slot image has unreferenced trailing bytes")
    return {
        "format": "PLTE Qwen3 tiered-slot readback audit v1",
        "blocks": len(rows),
        "order": "results sorted by id; low nibble first",
        "tier_map_bytes": len(tier_map),
        "tier_map_sha256": sha256_bytes(tier_map),
        "container_bundle_bytes": len(bundle),
        "container_bundle_sha256": sha256_bytes(bundle),
        "tiered_slots_bytes": len(slots),
        "tiered_slots_sha256": sha256_bytes(slots),
        "all_header_derived_prefixes_exact": True,
        "all_prefixes_match_independently_decoded_container_hashes": True,
        "all_arithmetic_and_tail_padding_bits_zero": True,
        "all_slot_padding_bytes_zero": True,
        "bundle_and_slot_eof_exact": True,
    }


def validate_results(
    directory: Path,
    document: Any,
    manifest: dict[str, Any],
    manifest_sha: str,
    source_records: dict[str, dict[str, Any]],
    reservoir_plan: dict[str, Any],
    original_outcome_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    document = require_keys(
        document,
        {
            "format", "selection_manifest_sha256", "container_bundle",
            "container_bundle_bytes", "container_bundle_sha256", "tier_map",
            "tier_map_bytes", "tier_map_sha256", "tiered_slots",
            "tiered_slots_bytes", "tiered_slots_sha256",
            "packed_artifact_readback",
            "original_tier0_outcome", "original_tier0_outcome_sha256",
            "results",
        },
        "results.json",
    )
    require(document["format"] == "PLTE Qwen3 stratified encoder results v1", "results format mismatch")
    require(document["selection_manifest_sha256"] == manifest_sha, "results manifest mismatch")
    require(document["container_bundle"] == "containers.polar.bin", "unexpected container bundle name")
    bundle = directory / document["container_bundle"]
    require(bundle.is_file() and bundle.parent.resolve() == directory.resolve(), "container bundle is missing or outside artifact directory")
    require(document["container_bundle_bytes"] == bundle.stat().st_size, "bundle size mismatch")
    require(document["container_bundle_sha256"] == sha256_path(bundle), "bundle SHA-256 mismatch")
    payload = bundle.read_bytes()
    require(document["tier_map"] == "tier_map.bin", "unexpected tier-map name")
    tier_map_path = directory / document["tier_map"]
    require(tier_map_path.is_file() and tier_map_path.parent.resolve() == directory.resolve(), "tier map is missing or outside artifact directory")
    tier_map = tier_map_path.read_bytes()
    require(document["tier_map_bytes"] == len(tier_map) == 200, "tier-map size mismatch")
    require(document["tier_map_sha256"] == sha256_bytes(tier_map), "tier-map hash mismatch")
    require(document["tiered_slots"] == "tiered_slots.bin", "unexpected tiered-slot name")
    slots_path = directory / document["tiered_slots"]
    require(slots_path.is_file() and slots_path.parent.resolve() == directory.resolve(), "tiered slots are missing or outside artifact directory")
    slots = slots_path.read_bytes()
    require(document["tiered_slots_bytes"] == len(slots), "tiered-slot size mismatch")
    require(document["tiered_slots_sha256"] == sha256_bytes(slots), "tiered-slot hash mismatch")
    require(document["original_tier0_outcome"] == "original_tier0_outcome.json", "unexpected original Tier-0 outcome name")
    require(document["original_tier0_outcome_sha256"] == original_outcome_sha, "original Tier-0 outcome hash mismatch")

    rows = document["results"]
    require(isinstance(rows, list) and len(rows) == 400, "results must contain 400 rows")
    require(rows == sorted(rows, key=lambda row: str(row.get("id"))), "result rows are not ID-sorted")
    manifest_by_id = {row["id"]: row for row in manifest["plte_blocks"]}
    require({row.get("id") for row in rows} == set(manifest_by_id), "result IDs do not match manifest")
    expected_keys = {
        "id", "tensor", "block_index", "layer", "role", "source_sha256",
        "report_sha256", "container_offset", "container_bytes", "container_sha256",
        "reservoir_tier", "charged_slot_bytes",
        "charged_slot_bits_including_4bit_map",
        "charged_slot_bpw_including_4bit_map", "tiered_slot_offset",
        "zero_padding_bytes", "report",
    }
    allocation_by_id = {row["id"]: row for row in reservoir_plan["allocations"]}
    encoder_sha = manifest["provenance"]["encoder_sha256"]
    require(encoder_sha == sha256_path(ENCODER), "checked-out encoder hash mismatch")
    offset = 0
    slot_offset = 0
    summaries = []
    by_id = {}
    for row in rows:
        row = require_keys(row, expected_keys, f"result {row.get('id')}")
        entry = manifest_by_id[row["id"]]
        require(row["tensor"] == entry["tensor"] and row["block_index"] == entry["block_index"], f"result identity mismatch for {row['id']}")
        require(row["layer"] == entry["layer"] and row["role"] == entry["role"], f"result stratum mismatch for {row['id']}")
        require(row["source_sha256"] == source_records[row["id"]]["sha256"], f"result source hash mismatch for {row['id']}")
        allocation = allocation_by_id[row["id"]]
        require(row["reservoir_tier"] == allocation["tier"], f"result tier mismatch for {row['id']}")
        require(row["charged_slot_bytes"] == allocation["container_cap_bytes"], f"result charged slot mismatch for {row['id']}")
        require(row["tiered_slot_offset"] == slot_offset, f"tiered-slot offset is non-contiguous at {row['id']}")
        require(row["container_offset"] == offset, f"bundle offset is non-contiguous at {row['id']}")
        size = row["container_bytes"]
        charged = row["charged_slot_bytes"]
        charged_bits = 8 * charged + TIER_MAP_BITS_PER_BLOCK
        require(row["charged_slot_bits_including_4bit_map"] == charged_bits, f"all-in charged bits mismatch for {row['id']}")
        close(row["charged_slot_bpw_including_4bit_map"], charged_bits / BLOCK_VALUES, f"all-in charged rate {row['id']}")
        require(isinstance(size, int) and 0 < size <= charged, f"container size invalid for {row['id']}")
        require(row["zero_padding_bytes"] == charged - size, f"tier padding length mismatch for {row['id']}")
        segment = payload[offset : offset + size]
        require(len(segment) == size and sha256_bytes(segment) == row["container_sha256"], f"bundle segment hash mismatch for {row['id']}")
        slot = slots[slot_offset : slot_offset + charged]
        require(len(slot) == charged and slot[:size] == segment, f"tiered slot prefix mismatch for {row['id']}")
        require(not any(slot[size:]), f"tiered slot padding is non-zero for {row['id']}")
        require(is_sha256(row["report_sha256"]), f"report hash malformed for {row['id']}")
        require(canonical_json_sha256(row["report"]) == row["report_sha256"], f"embedded report hash mismatch for {row['id']}")
        if row["reservoir_tier"] == 0:
            require(row["report_sha256"] == allocation["report_sha256"], f"Tier-0 report changed for {row['id']}")
            require(size == allocation["container_bytes"] and row["container_sha256"] == allocation["container_sha256"], f"Tier-0 container changed for {row['id']}")
        else:
            require(row["report"]["trials"][0].get("base_literal_container_bytes") == allocation["first_pass_base_container_bytes"], f"overflow base length did not reproduce for {row['id']}")
        summaries.append(
            validate_encoder_report(
                row["report"], row, entry, row["source_sha256"], encoder_sha, charged
            )
        )
        if row["reservoir_tier"] == 0:
            close(row["report"]["trials"][0]["relative_mse"], allocation["first_pass_relative_mse"], f"Tier-0 frozen MSE {row['id']}")
            close(row["report"]["trials"][0]["gap_db"], allocation["first_pass_gap_db"], f"Tier-0 frozen gap {row['id']}")
        by_id[row["id"]] = row
        offset += size
        slot_offset += charged
    require(offset == len(payload), "container bundle has unreferenced trailing bytes")
    require(slot_offset == len(slots), "tiered-slot image has unreferenced trailing bytes")
    expected_tiers = [int(row["reservoir_tier"]) for row in rows]
    expected_map = bytes(
        expected_tiers[index]
        | ((expected_tiers[index + 1] if index + 1 < len(expected_tiers) else 0) << 4)
        for index in range(0, len(expected_tiers), 2)
    )
    require(tier_map == expected_map, "tier-map nibbles do not match result allocation")
    packing_readback = audit_packed_artifacts(payload, tier_map, slots, rows)
    deep_close(
        document["packed_artifact_readback"],
        packing_readback,
        "results.json.packed_artifact_readback",
    )
    return summaries, by_id, packing_readback


def validate_independent_decodes(
    document: Any,
    manifest: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> None:
    document = require_keys(document, {"format", "decoder_sha256", "audits"}, "independent_decodes.json")
    decoder_sha = manifest["provenance"]["independent_decoder_sha256"]
    require(document["format"] == "PLTE Qwen3 stratified independent decode audit v1", "independent audit format mismatch")
    require(document["decoder_sha256"] == decoder_sha == sha256_path(DECODER), "independent decoder hash mismatch")
    rows = document["audits"]
    require(isinstance(rows, list) and len(rows) == 400, "independent audit must contain 400 rows")
    require(rows == sorted(rows, key=lambda row: str(row.get("id"))), "independent audits are not ID-sorted")
    require({row.get("id") for row in rows} == set(result_by_id), "independent audit IDs mismatch")
    for row in rows:
        row = require_keys(row, {"id", "audit_sha256", "audit", "receipt"}, f"independent audit {row.get('id')}")
        result = result_by_id[row["id"]]
        audit = row["audit"]
        require(is_sha256(row["audit_sha256"]) and canonical_json_sha256(audit) == row["audit_sha256"], f"audit hash mismatch for {row['id']}")
        require(audit.get("status") == "decoded in a clean implementation without encoder probabilities", f"decoder status mismatch for {row['id']}")
        require(audit.get("container_sha256") == result["container_sha256"] and audit.get("container_bytes") == result["container_bytes"], f"decoder container mismatch for {row['id']}")
        require(audit.get("source_block_bf16_sha256") == source_records[row["id"]]["sha256"], f"decoder source mismatch for {row['id']}")
        for flag in (
            "tail_escape_padding_zero",
            "decoded_reconstruction_matches_encoder_metric_at_1e_12",
            "decoded_indices_match_encoder_metric_at_1e_12",
        ):
            require(audit.get(flag) is True, f"decoder flag is not true for {row['id']}: {flag}")
        compatibility = audit.get("conditional_slot_budget_compatibility", {})
        require(compatibility.get("fits_conditional_fixed_slot_budget") is True, f"decoder slot failure for {row['id']}")
        require(compatibility.get("fixed_slot_cap_bytes") == result["charged_slot_bytes"], f"decoder cap mismatch for {row['id']}")
        require(compatibility.get("actual_container_bytes") == result["container_bytes"], f"decoder byte count mismatch for {row['id']}")
        require(compatibility.get("zero_padding_bytes_needed_by_hypothetical_packer") == result["zero_padding_bytes"], f"decoder padding count mismatch for {row['id']}")
        require(compatibility.get("realized_checkpoint_packer_exercised") is False, f"decoder overstates checkpoint packing for {row['id']}")
        trial = result["report"]["trials"][0]
        decoded_relative = finite_number(audit.get("decoded_relative_mse_with_serialized_scale"), f"decoded relative MSE {row['id']}", positive=True)
        close(decoded_relative, trial["relative_mse"], f"decoded/encoder relative MSE {row['id']}", tolerance=1e-12)
        close(audit.get("encoder_metadata_relative_mse"), trial["relative_mse"], f"decoder metadata MSE {row['id']}", tolerance=1e-12)
        close(audit.get("decoded_absolute_mse_with_serialized_scale"), trial["literal_decoded_absolute_mse"], f"decoded/encoder absolute MSE {row['id']}", tolerance=1e-12)
        delta = finite_number(audit.get("serialized_scale_mse_delta_from_encoder_metric"), f"decoder MSE delta {row['id']}")
        require(abs(delta) <= 1e-12, f"decoder MSE delta exceeds 1e-12 for {row['id']}")
        require(audit.get("logical_payload_bits") == trial.get("arithmetic_logical_bits"), f"decoder logical length mismatch for {row['id']}")

        receipt = row["receipt"]
        require(isinstance(receipt, dict), f"decoder receipt is not an object for {row['id']}")
        expected_receipt = {
            "id": row["id"],
            "decoder_sha256": decoder_sha,
            "profile_sha256": manifest["provenance"]["frozen_profile_sha256"],
            "report_sha256": result["report_sha256"],
            "container_sha256": result["container_sha256"],
            "source_sha256": source_records[row["id"]]["sha256"],
            "audit_sha256": row["audit_sha256"],
        }
        for key, value in expected_receipt.items():
            require(receipt.get(key) == value, f"decoder receipt {key} mismatch for {row['id']}")
        require(is_sha256(receipt.get("log_sha256")), f"decoder receipt log hash malformed for {row['id']}")
        if "relative_mse" in receipt:
            close(receipt["relative_mse"], trial["relative_mse"], f"decoder receipt MSE {row['id']}", tolerance=1e-12)
        if "wall_seconds" in receipt:
            finite_number(receipt["wall_seconds"], f"decoder receipt duration {row['id']}", positive=True)
        argv = receipt.get("canonical_argv")
        require(isinstance(argv, list) and len(argv) == 14 and all(isinstance(value, str) for value in argv), f"decoder receipt argv malformed for {row['id']}")
        require(bool(argv[0]) and Path(argv[1]).name == DECODER.name, f"decoder receipt executable mismatch for {row['id']}")
        expected_tail = [
            "--container",
            f"tmp/qwen3_stratified_v1/jobs/{row['id']}.polar.bin",
            "--container-layout",
            "plte-u20-tail-fp32",
            "--metadata",
            f"tmp/qwen3_stratified_v1/jobs/{row['id']}.json",
            "--raw-mask",
            "plte/agent_root_polar_escape_frozen_profiles.bin",
            "--source-bf16",
            f"tmp/qwen3_stratified_v1/sources/{row['id']}.bf16.bin",
            "--output",
        ]
        require(argv[2:13] == expected_tail, f"decoder receipt argv inputs mismatch for {row['id']}")
        require(Path(argv[13]).name == f".{row['id']}.partial.json", f"decoder receipt output mismatch for {row['id']}")


def validate_rank1(
    document: Any, manifest: dict[str, Any], source_records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected_top_keys = {
        "format", "codec", "tensor_count", "values", "source_energy_fp64",
        "sse", "relative_mse", "all_literal_decodes_exact", "tensors",
    }
    document = require_keys(document, expected_top_keys, "rank1_exact_audit.json")
    require(document["format"] == "PLTE Qwen3 rank-one exact-BF16 census v1", "rank-one audit format mismatch")
    require(document["codec"] == "lossless raw BF16 exception", "rank-one codec mismatch")
    require(document["tensor_count"] == 193 and document["values"] == 210_944, "rank-one census totals mismatch")
    require(document["sse"] == 0.0 and document["relative_mse"] == 0.0, "rank-one exception is not lossless")
    require(document["all_literal_decodes_exact"] is True, "rank-one exact-decode flag is false")
    rows = document["tensors"]
    require(isinstance(rows, list) and len(rows) == 193, "rank-one audit row count mismatch")
    manifest_by_id = {row["id"]: row for row in manifest["rank1_tensors"]}
    require([row.get("id") for row in rows] == [row["id"] for row in manifest["rank1_tensors"]], "rank-one audit ordering/IDs mismatch")
    expected_row_keys = {
        "id", "layer", "role", "tensor", "values", "bytes", "source_sha256",
        "decoded_sha256", "source_energy_fp64", "literal_decode_exact", "sse",
    }
    energy = 0.0
    for row in rows:
        row = require_keys(row, expected_row_keys, f"rank-one audit {row.get('id')}")
        entry = manifest_by_id[row["id"]]
        for key in ("layer", "role", "tensor", "values", "bytes"):
            require(row[key] == entry[key], f"rank-one metadata mismatch for {row['id']}: {key}")
        require(row["source_sha256"] == source_records[row["id"]]["sha256"], f"rank-one source hash mismatch for {row['id']}")
        require(row["decoded_sha256"] == row["source_sha256"] and is_sha256(row["source_sha256"]), f"rank-one decode hash mismatch for {row['id']}")
        require(row["literal_decode_exact"] is True and row["sse"] == 0.0, f"rank-one decode is not exact for {row['id']}")
        energy += finite_number(row["source_energy_fp64"], f"rank-one energy {row['id']}", positive=True)
    close(document["source_energy_fp64"], energy, "rank-one aggregate energy")
    return {key: value for key, value in document.items() if key != "tensors"}


def validate_router_census(manifest: dict[str, Any]) -> dict[str, Any]:
    report = load_json(ROUTER_REPORT)
    require(report.get("strict_ptq") is True and report.get("model_training_or_retraining") is False, "router evidence is not strict PTQ")
    require(report.get("repo") == manifest["checkpoint"]["repo"] and report.get("revision") == manifest["checkpoint"]["revision"], "router checkpoint mismatch")
    routers = report.get("routers")
    require(report.get("router_count") == 48 and isinstance(routers, list) and len(routers) == 48, "router census count mismatch")
    require(report.get("router_values") == 48 * BLOCK_VALUES, "router value count mismatch")
    require([row.get("layer") for row in routers] == list(range(48)), "router layer census mismatch")
    manifest_by_layer = {row["layer"]: row for row in manifest["router_blocks"]}
    energy = 0.0
    sse = 0.0
    record_bytes = 0
    for row in routers:
        layer = row["layer"]
        require(row.get("tensor") == manifest_by_layer[layer]["tensor"], f"router tensor mismatch at layer {layer}")
        require(row.get("input_bytes") == 2 * BLOCK_VALUES and is_sha256(row.get("input_sha256")), f"router source provenance invalid at layer {layer}")
        require(row.get("selected_tag") == 4 and row.get("inverse_decode_exact") is True, f"router Q4 decode failure at layer {layer}")
        attempts = row.get("attempts")
        require(isinstance(attempts, list) and {attempt.get("q") for attempt in attempts} == {2, 3, 4}, f"router attempts incomplete at layer {layer}")
        require(all(attempt.get("labels_roundtrip") is True for attempt in attempts), f"router label round trip failed at layer {layer}")
        selected = next(attempt for attempt in attempts if attempt["q"] == 4)
        close(row.get("selected_sse"), selected.get("sse"), f"router selected SSE layer {layer}")
        close(row.get("selected_relative_mse"), selected.get("relative_mse"), f"router selected MSE layer {layer}")
        energy += finite_number(row.get("energy"), f"router energy layer {layer}", positive=True)
        sse += finite_number(row.get("selected_sse"), f"router SSE layer {layer}", positive=True)
        require(isinstance(row.get("selected_record_bytes"), int) and row["selected_record_bytes"] > 0, f"router record size invalid at layer {layer}")
        record_bytes += row["selected_record_bytes"]
    require(report.get("selected_tag_counts") == {"2": 0, "3": 0, "4": 48, "16": 0}, "router selection is not all Q4")
    aggregate = report.get("aggregate", {})
    close(aggregate.get("source_energy"), energy, "router aggregate energy")
    close(aggregate.get("sse"), sse, "router aggregate SSE")
    close(aggregate.get("relative_mse"), sse / energy, "router aggregate relative MSE")
    require(aggregate.get("container_bytes") == ROUTER_CONTAINER.stat().st_size, "router container size mismatch")
    require(aggregate.get("container_sha256") == sha256_path(ROUTER_CONTAINER), "router container hash mismatch")
    require(report.get("format", {}).get("global_bytes") + record_bytes == aggregate["container_bytes"], "router record framing mismatch")
    independent = report.get("independent_literal_decode", {})
    require(independent.get("exact_file_length") is True and independent.get("bytes_consumed") == aggregate["container_bytes"], "router independent decoder did not consume exact file")
    close(independent.get("source_energy"), energy, "router independent energy")
    close(independent.get("sse"), sse, "router independent SSE")
    close(independent.get("relative_mse"), sse / energy, "router independent relative MSE")
    require(independent.get("aggregate_energy_match") is True and independent.get("aggregate_sse_match") is True, "router independent aggregate mismatch")
    return {
        "codec": "literal Q4 router exception",
        "router_count": 48,
        "layers": list(range(48)),
        "values": report["router_values"],
        "source_energy": aggregate["source_energy"],
        "sse": aggregate["sse"],
        "relative_mse": aggregate["relative_mse"],
        "container_bytes": aggregate["container_bytes"],
        "container_sha256": aggregate["container_sha256"],
        "all_inverse_decodes_exact": True,
        "independent_literal_decode": independent,
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot take percentile of an empty set")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(rows), "cannot summarize an empty row set")
    energy = sum(float(row["source_energy"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    distortion = sse / energy
    actual_bits = sum(int(row["container_bytes"]) * 8 for row in rows)
    actual_rate = actual_bits / (BLOCK_VALUES * len(rows))
    actual_gap = 10.0 * math.log10(distortion / (2.0 ** (-2.0 * actual_rate)))
    charged_slot_bits = sum(
        int(row["charged_slot_bytes"]) * 8 + TIER_MAP_BITS_PER_BLOCK
        for row in rows
    )
    charged_slot_rate = charged_slot_bits / (BLOCK_VALUES * len(rows))
    charged_slot_gap = 10.0 * math.log10(
        distortion / (2.0 ** (-2.0 * charged_slot_rate))
    )
    charged_gaps = [float(row["charged_tier_slot_gap_db"]) for row in rows]
    actual_gaps = [float(row["actual_gap_db"]) for row in rows]
    return {
        "blocks": len(rows),
        "source_energy": energy,
        "sse": sse,
        "energy_weighted_relative_mse": distortion,
        "actual_container_bits": actual_bits,
        "mean_actual_bpw": actual_rate,
        "aggregate_gap_at_mean_actual_rate_db": actual_gap,
        "charged_tier_slot_bits_including_4bit_map": charged_slot_bits,
        "mean_charged_tier_slot_bpw_including_4bit_map": charged_slot_rate,
        "aggregate_gap_at_charged_tier_slot_rate_db": charged_slot_gap,
        "pointwise_actual_gap_db": {
            "min": min(actual_gaps),
            "p50": percentile(actual_gaps, 0.50),
            "p95": percentile(actual_gaps, 0.95),
            "p99": percentile(actual_gaps, 0.99),
            "max": max(actual_gaps),
        },
        "pointwise_charged_tier_slot_gap_db": {
            "min": min(charged_gaps),
            "p50": percentile(charged_gaps, 0.50),
            "p95": percentile(charged_gaps, 0.95),
            "p99": percentile(charged_gaps, 0.99),
            "max": max(charged_gaps),
        },
        "charged_tier_slot_failures_ge_0p10db": sum(
            gap >= 0.10 for gap in charged_gaps
        ),
    }


def validate_summary(
    document: Any,
    manifest: dict[str, Any],
    manifest_sha: str,
    directory: Path,
    summary_rows: list[dict[str, Any]],
    rank1: dict[str, Any],
    routers: dict[str, Any],
    reservoir_plan: dict[str, Any],
    reservoir_plan_sha: str,
    original_outcome: dict[str, Any],
    original_outcome_sha: str,
    packing_readback: dict[str, Any],
) -> None:
    by_role = {
        role: summarize_rows([row for row in summary_rows if row["role"] == role])
        for role in sorted({str(row["role"]) for row in summary_rows})
    }
    by_layer = {
        str(layer): summarize_rows([row for row in summary_rows if row["layer"] == layer])
        for layer in range(48)
    }
    failures = [
        row for row in summary_rows if row["charged_tier_slot_gap_db"] >= 0.10
    ]
    bundle = directory / "containers.polar.bin"
    tier_map = directory / "tier_map.bin"
    tiered_slots = directory / "tiered_slots.bin"
    tiers = [int(row["reservoir_tier"]) for row in summary_rows]
    expected = {
        "format": "PLTE Qwen3 stratified evaluation summary v1",
        "checkpoint": manifest["checkpoint"],
        "strict_ptq": True,
        "selection_manifest_sha256": manifest_sha,
        "selection_was_held_out_from_previous_evidence": True,
        "post_hoc_engineering_amendment": True,
        "reservoir_plan_sha256": reservoir_plan_sha,
        "coverage": {
            "new_plte_blocks": 400,
            "layer_role_cells": 336,
            "layers": 48,
            "plte_layer_roles": 7,
            "embedding_blocks": 32,
            "lm_head_blocks": 32,
            "router_census_blocks": 48,
            "rank1_census_tensors": 193,
            "prior_plus_new_unique_plte_blocks": 447,
        },
        "integrity": {
            "runner_sha256": sha256_path(RUNNER),
            "encoder_sha256": manifest["provenance"]["encoder_sha256"],
            "independent_decoder_sha256": manifest["provenance"]["independent_decoder_sha256"],
            "frozen_profile_sha256": manifest["provenance"]["frozen_profile_sha256"],
            "all_encoder_internal_decodes_passed": True,
            "independent_clean_decodes": 400,
            "all_independent_clean_decodes_passed": True,
            "all_decode_receipts_bound": True,
            "container_bundle_bytes": bundle.stat().st_size,
            "container_bundle_sha256": sha256_path(bundle),
            "tier_map_bytes": tier_map.stat().st_size,
            "tier_map_sha256": sha256_path(tier_map),
            "tiered_slots_bytes": tiered_slots.stat().st_size,
            "tiered_slots_sha256": sha256_path(tiered_slots),
            "all_tier_padding_zero": True,
            "packed_artifact_readback": packing_readback,
        },
        "plte_panel": summarize_rows(summary_rows),
        "quality_endpoint": {
            "definition": (
                "every new block has all-in charged reservoir-tier slot gap "
                "including its four-bit map charge < 0.10 dB"
            ),
            "passes": len(failures) == 0,
            "failures": failures,
        },
        "original_tier0_endpoint": {
            **original_outcome,
            "artifact": "original_tier0_outcome.json",
            "artifact_sha256": original_outcome_sha,
        },
        "by_role": by_role,
        "by_layer": by_layer,
        "routers": routers,
        "reservoir": {
            "format": reservoir_plan["format"],
            "plan_sha256": reservoir_plan_sha,
            "tier_counts": {
                str(tier): tiers.count(tier) for tier in sorted(set(tiers))
            },
            "maximum_tier": max(tiers),
            "panel_tier_map_order": "results sorted by id; low nibble first",
            "panel_sum_tiers": sum(tiers),
            "checkpoint_rate_accounting": reservoir_plan[
                "checkpoint_rate_accounting"
            ],
        },
        "rank1": rank1,
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }
    deep_close(document, expected, "summary.json")
    require(document["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "claim boundary is overstated")


def validate_artifacts(
    directory: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
    reservoir_plan: dict[str, Any],
    reservoir_plan_sha: str,
) -> None:
    filenames = (
        "source_hashes.json",
        "results.json",
        "independent_decodes.json",
        "rank1_exact_audit.json",
        "summary.json",
        "containers.polar.bin",
        "tier_map.bin",
        "tiered_slots.bin",
        "original_tier0_outcome.json",
    )
    missing = [name for name in filenames if not (directory / name).is_file()]
    require(not missing, f"final artifacts are incomplete; missing: {', '.join(missing)}")
    forbidden = [
        path for path in directory.rglob("*")
        if path.is_file() and (path.name.endswith(".bf16.bin") or path.suffix == ".safetensors")
    ]
    if forbidden:
        raise VerificationError(
            f"raw weight payload found in publication directory: {forbidden[0]}"
        )

    source_records = validate_source_hashes(
        load_json(directory / "source_hashes.json"), manifest, manifest_sha
    )
    original_outcome = expected_original_tier0_outcome(
        directory, manifest, manifest_sha, reservoir_plan, reservoir_plan_sha
    )
    original_outcome_path = directory / "original_tier0_outcome.json"
    deep_close(
        load_json(original_outcome_path),
        original_outcome,
        "original_tier0_outcome.json",
    )
    original_outcome_sha = sha256_path(original_outcome_path)
    summary_rows, result_by_id, packing_readback = validate_results(
        directory,
        load_json(directory / "results.json"),
        manifest,
        manifest_sha,
        source_records,
        reservoir_plan,
        original_outcome_sha,
    )
    validate_independent_decodes(
        load_json(directory / "independent_decodes.json"),
        manifest,
        source_records,
        result_by_id,
    )
    rank1 = validate_rank1(
        load_json(directory / "rank1_exact_audit.json"), manifest, source_records
    )
    routers = validate_router_census(manifest)
    validate_summary(
        load_json(directory / "summary.json"),
        manifest,
        manifest_sha,
        directory,
        summary_rows,
        rank1,
        routers,
        reservoir_plan,
        reservoir_plan_sha,
        original_outcome,
        original_outcome_sha,
        packing_readback,
    )
    print(
        "artifacts: 400 bundle segments and encoder reports, 400 independent "
        "decodes, a 400-entry reservoir map and padded slot image, 193 exact "
        "rank-one tensors, and 48 routers verified"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reservoir-plan", type=Path, default=DEFAULT_RESERVOIR_PLAN)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--headers", type=Path, default=DEFAULT_HEADERS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="verify selection reproducibility and coverage without requiring result artifacts",
    )
    args = parser.parse_args()
    try:
        manifest, manifest_sha = validate_manifest(
            args.manifest.resolve(), args.headers.resolve(), args.evidence.resolve()
        )
        reservoir_plan, reservoir_plan_sha = validate_reservoir_plan(
            args.reservoir_plan.resolve(), manifest, manifest_sha
        )
        if not args.manifest_only:
            validate_artifacts(
                args.artifact_dir.resolve(),
                manifest,
                manifest_sha,
                reservoir_plan,
                reservoir_plan_sha,
            )
    except VerificationError as error:
        raise SystemExit(f"verification failed: {error}") from None
    scope = (
        "manifest and frozen reservoir plan"
        if args.manifest_only
        else "manifest, reservoir plan, and source-free artifacts"
    )
    print(f"status: all stratified evaluation checks passed ({scope})")


if __name__ == "__main__":
    main()
