#!/usr/bin/env python3
"""Verify the self-contained, non-weight PLTE evidence in this repository.

This verifier intentionally uses only the Python standard library. It checks
published bytes and audit metadata; it does not claim to remeasure MSE without
the original Qwen BF16 source blocks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
ENCODER = "agent_root_polar_lattice_gate.py"
DECODER = "agent_polar_codec_audit_independent_decoder.py"
LEDGER_CODE = "agent_root_polar_escape_full_model_ledger.py"
ROUTER_CODE = "agent_router_adaptive_q234.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_true(row: dict, names: list[str], context: str) -> None:
    for name in names:
        assert row.get(name) is True, f"{context}: {name} is not literal true"


def verify_frozen_profile() -> tuple[int, str]:
    profile_path = PLTE / "agent_root_polar_escape_frozen_profiles.bin"
    manifest = read_json(profile_path.with_suffix(".manifest.json"))
    payload = profile_path.read_bytes()
    assert len(payload) == 196_608
    assert manifest["block_length"] == 262_144
    assert manifest["levels"] == 6
    assert manifest["bits"] == 1_572_864
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    level_bytes = 262_144 // 8
    for level0, row in enumerate(manifest["per_level"]):
        level = payload[level0 * level_bytes : (level0 + 1) * level_bytes]
        frozen = sum(byte.bit_count() for byte in level)
        assert row == {
            "level": level0 + 1,
            "frozen": frozen,
            "open": 262_144 - frozen,
            "sha256": hashlib.sha256(level).hexdigest(),
        }
    return len(payload), manifest["sha256"]


def verify_polar_evidence() -> tuple[dict[str, dict], int, int]:
    manifest_path = PLTE / "agent_root_polar_escape_evidence_manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["format"] == "PLTE exact polar evidence manifest v1"
    assert manifest["checkpoint"] == {
        "repo": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    }
    assert manifest["encoder_sha256"] == sha256(PLTE / ENCODER)
    assert manifest["frozen_profile_sha256"] == sha256(
        PLTE / "agent_root_polar_escape_frozen_profiles.bin"
    )

    reports: dict[str, dict] = {}
    canonical_blocks: set[tuple[str, int]] = set()
    final_count = 0
    normative_count = 0
    audit_names = [
        "arithmetic_roundtrip_bits_match",
        "online_causal_arithmetic_bits_match",
        "causal_decoder_frequencies_match",
        "causal_decoder_frozen_bits_match",
        "reconstruction_indices_match",
        "tail_escape_records_roundtrip",
        "tail_escape_padding_is_zero",
        "container_header_roundtrip",
        "fp32_decoder_scale_in_mse_audit",
        "passes_container_cap",
        "passes_rate_lt_2p5",
        "passes_gap_lt_0p10db",
    ]
    for entry in manifest["reports"]:
        report_path = PLTE / entry["report"]
        container_path = PLTE / entry["container"]
        assert report_path.name not in reports, f"duplicate report: {report_path.name}"
        assert sha256(report_path) == entry["report_sha256"]
        assert container_path.stat().st_size == entry["container_bytes"]
        assert sha256(container_path) == entry["container_sha256"]
        assert entry["implementation_sha256"] == manifest["encoder_sha256"]

        report = read_json(report_path)
        assert report["strict_ptq"] is True
        assert report["source_training_or_retraining"] is False
        assert report["implementation_sha256"] == manifest["encoder_sha256"]
        assert len(report["trials"]) == 1
        trial = report["trials"][0]
        require_true(trial, audit_names, report_path.name)
        assert trial["literal_container_bytes"] == entry["container_bytes"]
        assert trial["literal_container_sha256"] == entry["container_sha256"]
        assert trial["source"]["block_bf16_sha256"] == entry["source_block_sha256"]
        assert trial["literal_container_bytes"] <= 81_242
        assert trial["gap_db"] < 0.10

        canonical_blocks.add(
            (entry["canonical_tensor"], int(entry["canonical_block_index"]))
        )
        final_count += entry["kind"] == "final"
        normative_count += entry["kind"] == "normative"
        reports[report_path.name] = {
            "entry": entry,
            "trial": trial,
        }

    assert len(reports) == 49
    assert len(canonical_blocks) == 47
    assert final_count == 47
    assert normative_count == 2
    return reports, len(canonical_blocks), max(
        row["trial"]["literal_container_bytes"] for row in reports.values()
    )


def verify_standalone(reports: dict[str, dict]) -> int:
    paths = sorted(PLTE.glob("agent_root_polar_escape_*_standalone_decode.json"))
    assert len(paths) == 8
    for path in paths:
        audit = read_json(path)
        require_true(
            audit,
            [
                "tail_escape_padding_zero",
                "decoded_reconstruction_matches_encoder_metric_at_1e_12",
                "decoded_indices_match_encoder_metric_at_1e_12",
            ],
            path.name,
        )
        compatibility = audit["conditional_slot_budget_compatibility"]
        assert compatibility["fits_conditional_fixed_slot_budget"] is True
        assert compatibility["realized_checkpoint_packer_exercised"] is False
        report_name = path.name.replace("_standalone_decode.json", ".json")
        row = reports[report_name]
        assert audit["container_sha256"] == row["entry"]["container_sha256"]
        assert audit["container_bytes"] == row["entry"]["container_bytes"]
        assert audit["source_block_bf16_sha256"] == row["entry"]["source_block_sha256"]
        assert (
            abs(
                audit["decoded_relative_mse_with_serialized_scale"]
                - row["trial"]["relative_mse"]
            )
            <= 1e-12
        )
    return len(paths)


def verify_router() -> int:
    audit = read_json(PLTE / "agent_router_adaptive_q4_all48_audit.json")
    router = audit["literal_router_codec"]
    artifacts = audit["artifacts"]
    assert audit["strict_ptq"] is True
    assert audit["model_training_or_retraining"] is False
    assert router["tag_counts"] == {"2": 0, "3": 0, "4": 48, "16": 0}
    assert router["container_bytes"] == 6_488_688
    assert router["container_bits"] == 51_909_504
    assert router["inverse_decode"] == {
        "bytes_consumed": 6_488_688,
        "exact_file_length": True,
        "source_energy_match": True,
        "sse_match": True,
        "per_record_crc32": True,
    }
    assert sha256(PLTE / artifacts["container_local"]) == artifacts["container_sha256"]
    assert sha256(PLTE / artifacts["result_local"]) == artifacts["result_sha256"]
    assert sha256(PLTE / artifacts["encoder_local"]) == artifacts["encoder_sha256"]
    return router["container_bytes"]


def verify_ledger() -> dict:
    path = PLTE / "agent_root_polar_escape_full_model_ledger.json"
    ledger = read_json(path)
    implementations = ledger["implementation_artifacts"]
    expected = {
        ENCODER: PLTE / ENCODER,
        DECODER: PLTE / DECODER,
        ROUTER_CODE: PLTE / ROUTER_CODE,
        LEDGER_CODE: PLTE / LEDGER_CODE,
    }
    for name, source in expected.items():
        assert implementations[name]["bytes"] == source.stat().st_size
        assert implementations[name]["sha256"] == sha256(source)
    evidence = ledger["polar_evidence"]["manifest"]
    evidence_path = PLTE / evidence["path"]
    assert evidence["bytes"] == evidence_path.stat().st_size
    assert evidence["sha256"] == sha256(evidence_path)

    budget = ledger["conditional_fixed_slot_rate_budget"]
    projection = ledger["sample_based_distortion_projection"]
    assert budget["total_bits"] == 75_724_918_048
    assert abs(budget["global_bpw"] - 2.4801720791097526) < 1e-15
    assert budget["budget_fits_below_2p5_if_all_blocks_are_encodable"] is True
    assert abs(projection["mixed_projected_relative_mse"] - 0.03273955118126599) < 1e-15
    assert abs(projection["mixed_projected_gap_db"] - 0.08285101917462104) < 1e-15
    assert projection["passes_0p10db_projection"] is True
    assert projection["all_six_projection_containers_clean_decoded"] is True
    assert ledger["polar_evidence"]["observed_unique_source_blocks"] == 47
    assert ledger["polar_evidence"]["checkpoint_nonrouter_blocks_unencoded"] == 116_375
    assert ledger["standalone_decoder_evidence"]["all_exact_metric_matches"] is True
    return {
        "conditional_bpw": budget["global_bpw"],
        "projected_relative_mse": projection["mixed_projected_relative_mse"],
        "projected_gap_db": projection["mixed_projected_gap_db"],
    }


def main() -> None:
    profile_bytes, profile_sha = verify_frozen_profile()
    reports, unique_blocks, max_bytes = verify_polar_evidence()
    standalone = verify_standalone(reports)
    router_bytes = verify_router()
    metrics = verify_ledger()
    print(
        json.dumps(
            {
                "status": "all published evidence checks passed",
                "polar_reports": len(reports),
                "unique_qwen_blocks": unique_blocks,
                "standalone_clean_decodes": standalone,
                "max_observed_polar_container_bytes": max_bytes,
                "router_container_bytes": router_bytes,
                "frozen_profile_bytes": profile_bytes,
                "frozen_profile_sha256": profile_sha,
                **metrics,
                "source_weight_bytes_required_for_this_integrity_check": False,
                "whole_checkpoint_claim": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
