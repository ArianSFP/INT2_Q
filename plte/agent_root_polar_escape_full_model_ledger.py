#!/usr/bin/env python3
"""Build the auditable checkpoint ledger for the polar-tail PTQ candidate.

This script deliberately separates exact byte accounting from the sampled
distortion projection.  It never labels the latter a full-checkpoint result.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MODEL_REPO = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BLOCK_LENGTH = 1 << 18
POLAR_SLOT_BYTES = 81_242
GLOBAL_HEADER_BITS = 4_096
PER_TENSOR_HEADER_BITS = 64
FROZEN_PROFILE_BITS = 6 * BLOCK_LENGTH
TARGET_BPW = 2.5
TARGET_GAP_DB = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_bf16_block(path: Path, block_index: int) -> tuple[str, float, float]:
    block_bytes = BLOCK_LENGTH * 2
    with path.open("rb") as handle:
        handle.seek(block_index * block_bytes)
        payload = handle.read(block_bytes)
    if len(payload) != block_bytes:
        raise AssertionError(
            f"short BF16 source block {path} index {block_index}: "
            f"{len(payload)} != {block_bytes} bytes"
        )
    raw = np.frombuffer(payload, dtype="<u2")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    energy = float(np.square(values).sum(dtype=np.float64))
    rms = math.sqrt(energy / BLOCK_LENGTH)
    return hashlib.sha256(payload).hexdigest(), energy, rms


def canonical_source_identity(source_path: Path, source_block_index: int) -> tuple[str, int]:
    manifest_path = source_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": "BF16",
            "block_values": BLOCK_LENGTH,
            "bytes": BLOCK_LENGTH * 2,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise AssertionError(
                    f"range manifest mismatch for {source_path}: {key}={manifest.get(key)!r}"
                )
        manifest_output = Path(manifest["output"]).as_posix()
        source_output = source_path.as_posix()
        if not (
            manifest_output == source_output
            or manifest_output.endswith("/" + source_output)
        ):
            raise AssertionError(f"range manifest output mismatch for {source_path}")
        if source_block_index != 0:
            raise AssertionError("a one-block range artifact must be decoded at local block zero")
        if manifest["sha256"] != sha256_file(source_path):
            raise AssertionError(f"range source file hash mismatch for {source_path}")
        return str(manifest["tensor"]), int(manifest["block_index"])

    prefix = "qwen_weight_cache/tensors/"
    source_posix = source_path.as_posix()
    suffix = ".bf16.bin"
    if not source_posix.startswith(prefix) or not source_posix.endswith(suffix):
        raise AssertionError(f"source lacks a canonical range manifest or tensor path: {source_path}")
    return source_posix[len(prefix) : -len(suffix)], source_block_index


def classify_rank2(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if ".mlp.experts." in name:
        return "moe_expert"
    if name.endswith(".mlp.gate.weight"):
        return "moe_router"
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if name.endswith(f".self_attn.{role}.weight"):
            return f"attention_{role[0]}"
    return "other_rank2"


def load_inventory(header_dir: Path) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    for path in sorted(header_dir.glob("*.header.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        header = document.get("header", document)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise AssertionError(f"duplicate tensor {name}")
            tensors[name] = metadata

    category_rows: dict[str, dict[str, int]] = {}
    rank1_parameters = 0
    rank1_tensors = 0
    rank2_parameters = 0
    rank2_tensors = 0
    rank2_padding = 0
    dtypes: set[str] = set()
    for name, metadata in tensors.items():
        shape = [int(value) for value in metadata["shape"]]
        parameters = math.prod(shape)
        dtypes.add(str(metadata["dtype"]))
        if len(shape) == 1:
            rank1_tensors += 1
            rank1_parameters += parameters
        elif len(shape) == 2:
            rank2_tensors += 1
            rank2_parameters += parameters
            padding = (-parameters) % BLOCK_LENGTH
            rank2_padding += padding
            category = classify_rank2(name)
            row = category_rows.setdefault(category, {"tensors": 0, "parameters": 0, "blocks": 0})
            row["tensors"] += 1
            row["parameters"] += parameters
            row["blocks"] += (parameters + padding) // BLOCK_LENGTH
        else:
            raise AssertionError((name, shape))

    total_parameters = rank1_parameters + rank2_parameters
    router = category_rows["moe_router"]
    nonrouter_parameters = rank2_parameters - router["parameters"]
    nonrouter_blocks = sum(
        row["blocks"] for category, row in category_rows.items() if category != "moe_router"
    )
    result = {
        "tensor_count": len(tensors),
        "dtypes": sorted(dtypes),
        "total_parameters": total_parameters,
        "rank1_tensors": rank1_tensors,
        "rank1_parameters": rank1_parameters,
        "rank2_tensors": rank2_tensors,
        "rank2_parameters": rank2_parameters,
        "rank2_padding_parameters": rank2_padding,
        "router_tensors": router["tensors"],
        "router_parameters": router["parameters"],
        "nonrouter_rank2_parameters": nonrouter_parameters,
        "nonrouter_rank2_blocks": nonrouter_blocks,
        "categories": dict(sorted(category_rows.items())),
    }
    expected = {
        "tensor_count": 18_867,
        "total_parameters": 30_532_122_624,
        "rank1_tensors": 193,
        "rank1_parameters": 210_944,
        "rank2_tensors": 18_674,
        "rank2_parameters": 30_531_911_680,
        "rank2_padding_parameters": 0,
        "router_tensors": 48,
        "router_parameters": 12_582_912,
        "nonrouter_rank2_blocks": 116_422,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise AssertionError((key, result[key], value))
    return result


def discover_polar_reports(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(name) for name in glob.glob(pattern))
    paths = [path for path in sorted(set(paths)) if not path.name.endswith("_standalone_decode.json")]
    if not paths:
        raise AssertionError("no polar evidence reports discovered")
    return paths


def write_evidence_manifest(paths: list[Path], output: Path) -> None:
    entries: list[dict[str, Any]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if len(document.get("trials", [])) != 1:
            raise AssertionError(f"expected exactly one trial in {path}")
        trial = document["trials"][0]
        source = trial["source"]
        source_path = Path(source["path"])
        source_block_index = int(source["block_index"])
        source_sha256, _, _ = audit_bf16_block(source_path, source_block_index)
        canonical_tensor, canonical_block_index = canonical_source_identity(
            source_path, source_block_index
        )
        container_path = path.with_suffix(".polar.bin")
        entries.append(
            {
                "kind": "normative" if "_normative_" in path.name else "final",
                "report": path.as_posix(),
                "report_sha256": sha256_file(path),
                "container": container_path.as_posix(),
                "container_bytes": container_path.stat().st_size,
                "container_sha256": sha256_file(container_path),
                "source": source_path.as_posix(),
                "source_local_block_index": source_block_index,
                "source_block_sha256": source_sha256,
                "canonical_tensor": canonical_tensor,
                "canonical_block_index": canonical_block_index,
                "implementation_sha256": document["implementation_sha256"],
            }
        )
    normative_names = {Path(entry["report"]).name for entry in entries if entry["kind"] == "normative"}
    expected_normative_names = {
        "agent_root_polar_escape_normative_primary.json",
        "agent_root_polar_escape_normative_embed_b0.json",
    }
    if normative_names != expected_normative_names:
        raise AssertionError(
            f"normative evidence set mismatch: {normative_names} != {expected_normative_names}"
        )
    payload = {
        "format": "PLTE exact polar evidence manifest v1",
        "checkpoint": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "encoder_sha256": sha256_file(Path("agent_root_polar_lattice_gate.py")),
        "frozen_profile_sha256": sha256_file(
            Path("agent_root_polar_escape_frozen_profiles.bin")
        ),
        "reports": entries,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_evidence_manifest(path: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "PLTE exact polar evidence manifest v1":
        raise AssertionError("unexpected polar evidence manifest format")
    if manifest.get("checkpoint") != {"repo": MODEL_REPO, "revision": MODEL_REVISION}:
        raise AssertionError("polar evidence manifest checkpoint mismatch")
    if manifest.get("encoder_sha256") != sha256_file(Path("agent_root_polar_lattice_gate.py")):
        raise AssertionError("polar evidence manifest encoder mismatch")
    if manifest.get("frozen_profile_sha256") != sha256_file(
        Path("agent_root_polar_escape_frozen_profiles.bin")
    ):
        raise AssertionError("polar evidence manifest frozen profile mismatch")
    entries = manifest.get("reports")
    if not isinstance(entries, list) or not entries:
        raise AssertionError("empty polar evidence manifest")
    report_paths: list[Path] = []
    seen: set[str] = set()
    for entry in entries:
        report_path = Path(entry["report"])
        key = report_path.as_posix()
        if key in seen:
            raise AssertionError(f"duplicate evidence report {key}")
        seen.add(key)
        container_path = Path(entry["container"])
        source_path = Path(entry["source"])
        if sha256_file(report_path) != entry["report_sha256"]:
            raise AssertionError(f"evidence report drift: {report_path}")
        if (
            container_path.stat().st_size != int(entry["container_bytes"])
            or sha256_file(container_path) != entry["container_sha256"]
        ):
            raise AssertionError(f"evidence container drift: {container_path}")
        source_sha256, _, _ = audit_bf16_block(
            source_path, int(entry["source_local_block_index"])
        )
        if source_sha256 != entry["source_block_sha256"]:
            raise AssertionError(f"evidence source drift: {source_path}")
        canonical = canonical_source_identity(
            source_path, int(entry["source_local_block_index"])
        )
        if canonical != (entry["canonical_tensor"], int(entry["canonical_block_index"])):
            raise AssertionError(f"canonical source identity drift: {source_path}")
        report_paths.append(report_path)
    return report_paths, manifest


def load_polar_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["parameters"]["block_length"] != BLOCK_LENGTH:
            raise AssertionError(f"unexpected block length in {path}")
        if len(document["trials"]) != 1:
            raise AssertionError(f"expected exactly one trial in {path}")
        expected_parameters = {
            "block_length": BLOCK_LENGTH,
            "trials": 1,
            "sigma_source": 3.0,
            "test_channel_distortion": 0.29,
            "eta": 0.5989929996555583,
            "alphabet_size": 64,
            "decision": "map",
            "seed": 20260831,
            "container_cap_bytes": POLAR_SLOT_BYTES,
        }
        for name, expected in expected_parameters.items():
            if document["parameters"].get(name) != expected:
                raise AssertionError(f"codec parameter mismatch in {path}: {name}")
        if document.get("strict_ptq") is not True:
            raise AssertionError(f"strict-PTQ flag is not true in {path}")
        if document.get("source_training_or_retraining") is not False:
            raise AssertionError(f"training/retraining flag is not false in {path}")
        for trial in document["trials"]:
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
            ]
            audits = {name: trial[name] is True for name in audit_names}
            source = trial["source"]
            if source.get("kind") != "frozen_bf16_weight_block":
                raise AssertionError(f"non-frozen source in {path}")
            if int(source.get("values", -1)) != BLOCK_LENGTH:
                raise AssertionError(f"source block length mismatch in {path}")
            source_path = Path(source["path"])
            source_block_index = int(source["block_index"])
            source_block_sha256 = str(source["block_bf16_sha256"])
            local_source_sha256, energy, rms = audit_bf16_block(
                source_path, source_block_index
            )
            if local_source_sha256 != source_block_sha256:
                raise AssertionError(
                    f"source block hash mismatch for {path}: "
                    f"{local_source_sha256} != {source_block_sha256}"
                )
            if not math.isclose(
                rms, float(source["block_rms_fp64"]), rel_tol=1e-12, abs_tol=0.0
            ):
                raise AssertionError(f"source RMS mismatch for {path}")
            canonical_tensor, canonical_block_index = canonical_source_identity(
                source_path, source_block_index
            )
            container_path = path.with_suffix(".polar.bin")
            if not container_path.exists():
                raise AssertionError(f"missing literal container for {path}: {container_path}")
            container_sha256 = sha256_file(container_path)
            if container_sha256 != trial["literal_container_sha256"]:
                raise AssertionError(f"container hash mismatch for {path}")
            if container_path.stat().st_size != int(trial["literal_container_bytes"]):
                raise AssertionError(f"container length mismatch for {path}")
            if trial["passes_container_cap"] is not True:
                raise AssertionError(f"container cap audit failed for {path}")
            if trial["passes_rate_lt_2p5"] is not True:
                raise AssertionError(f"per-block rate audit failed for {path}")
            rate_bpw = container_path.stat().st_size * 8 / BLOCK_LENGTH
            if not math.isclose(rate_bpw, float(trial["screen_bpw"]), abs_tol=1e-15):
                raise AssertionError(f"rate recomputation mismatch for {path}")
            distortion = float(trial["relative_mse"])
            gaussian = 2.0 ** (-2.0 * rate_bpw)
            gap_db = 10.0 * math.log10(distortion / gaussian)
            if not math.isclose(gap_db, float(trial["gap_db"]), abs_tol=1e-12):
                raise AssertionError(f"gap recomputation mismatch for {path}")
            rows.append(
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "implementation_sha256": document["implementation_sha256"],
                    "source_path": source["path"],
                    "source_block_index": source_block_index,
                    "canonical_tensor": canonical_tensor,
                    "canonical_block_index": canonical_block_index,
                    "source_block_sha256": source_block_sha256,
                    "source_block_sha256_matches_local": True,
                    "container_file": container_path.name,
                    "container_sha256": container_sha256,
                    "container_bytes": int(trial["literal_container_bytes"]),
                    "rate_bpw": rate_bpw,
                    "relative_mse": distortion,
                    "gap_db": gap_db,
                    "source_energy": energy,
                    "sse": energy * distortion,
                    "tail_escape_count": int(trial["tail_escape_count"]),
                    "passes_container_cap": True,
                    "frozen_seed_regeneration_audited": True,
                    "all_decoder_audits": all(audits.values()),
                    "audits": audits,
                }
            )
    if not rows:
        raise AssertionError("no N=2^18 polar evidence found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-dir", type=Path, default=Path("qwen_weight_cache/headers"))
    parser.add_argument(
        "--router-audit", type=Path, default=Path("agent_router_adaptive_q4_all48_audit.json")
    )
    parser.add_argument(
        "--polar-pattern",
        action="append",
        default=None,
        help="discovery glob used only with --refresh-evidence-manifest",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("agent_root_polar_escape_evidence_manifest.json"),
    )
    parser.add_argument("--refresh-evidence-manifest", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("agent_root_polar_escape_full_model_ledger.json")
    )
    args = parser.parse_args()

    inventory = load_inventory(args.header_dir)
    if args.refresh_evidence_manifest:
        patterns = args.polar_pattern or [
            "agent_root_polar_escape_final_*.json",
            "agent_root_polar_escape_normative_*.json",
        ]
        write_evidence_manifest(
            discover_polar_reports(patterns), args.evidence_manifest
        )
    elif args.polar_pattern:
        raise ValueError("--polar-pattern requires --refresh-evidence-manifest")
    evidence_paths, evidence_manifest = load_evidence_manifest(
        args.evidence_manifest
    )
    polar_rows = load_polar_rows(evidence_paths)
    encoder_sha256 = sha256_file(Path("agent_root_polar_lattice_gate.py"))
    if not all(row["implementation_sha256"] == encoder_sha256 for row in polar_rows):
        raise AssertionError("an observed polar result was not produced by the current encoder")
    if not all(row["all_decoder_audits"] for row in polar_rows):
        raise AssertionError("a polar decoder audit failed")
    if not all(row["passes_container_cap"] for row in polar_rows):
        raise AssertionError("an observed tail-escape container exceeded its cap")
    observed_max = max(row["container_bytes"] for row in polar_rows)
    if observed_max > POLAR_SLOT_BYTES:
        raise AssertionError((observed_max, POLAR_SLOT_BYTES))

    router_document = json.loads(args.router_audit.read_text(encoding="utf-8"))
    expected_checkpoint = {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "parameters": inventory["total_parameters"],
        "router_tensors": 48,
        "router_shape_each": [128, 2048],
        "router_parameters_each": BLOCK_LENGTH,
        "router_parameters_total": inventory["router_parameters"],
        "nonrouter_rank2_blocks": inventory["nonrouter_rank2_blocks"],
        "rank2_padding_parameters": 0,
    }
    for name, expected in expected_checkpoint.items():
        if router_document["checkpoint"].get(name) != expected:
            raise AssertionError(f"router checkpoint mismatch: {name}")
    router = router_document["literal_router_codec"]
    if router.get("tag_counts") != {"2": 0, "3": 0, "4": 48, "16": 0}:
        raise AssertionError("router tag counts are not the pinned all-Q4 allocation")
    if (
        int(router["container_bytes"]) != 6_488_688
        or int(router["container_bits"]) != 6_488_688 * 8
        or int(router["record_bytes_each"]) != 135_180
    ):
        raise AssertionError("router byte geometry mismatch")
    expected_inverse = {
        "bytes_consumed": int(router["container_bytes"]),
        "exact_file_length": True,
        "source_energy_match": True,
        "sse_match": True,
        "per_record_crc32": True,
    }
    if router.get("inverse_decode") != expected_inverse:
        raise AssertionError("router inverse decode audit failed")
    if not math.isclose(
        float(router["sse"]) / float(router["source_energy"]),
        float(router["relative_mse"]),
        abs_tol=1e-15,
    ):
        raise AssertionError("router MSE arithmetic mismatch")
    router_artifacts = router_document["artifacts"]
    router_container_path = Path(router_artifacts["container_local"])
    router_result_path = Path(router_artifacts["result_local"])
    router_encoder_path = Path(router_artifacts["encoder_local"])
    if (
        router_container_path.stat().st_size != int(router_artifacts["container_bytes"])
        or router_container_path.stat().st_size != int(router["container_bytes"])
        or sha256_file(router_container_path) != router_artifacts["container_sha256"]
        or router_result_path.stat().st_size != int(router_artifacts["result_bytes"])
        or sha256_file(router_result_path) != router_artifacts["result_sha256"]
        or sha256_file(router_encoder_path) != router_artifacts["encoder_sha256"]
    ):
        raise AssertionError("router implementation/result/container provenance mismatch")

    frozen_profile_path = Path("agent_root_polar_escape_frozen_profiles.bin")
    frozen_manifest_path = frozen_profile_path.with_suffix(".manifest.json")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_payload = frozen_profile_path.read_bytes()
    expected_frozen_parameters = {
        "sigma_source": 3.0,
        "test_channel_distortion": 0.29,
        "eta": 0.5989929996555583,
        "tilde_sigma": 0.5297693418418581,
        "capacity_schedule": [
            0.0006403541494273135,
            0.2226280511277603,
            0.906837113158238,
            0.9999736826737476,
            1.0,
            1.0,
        ],
    }
    if (
        frozen_profile_path.stat().st_size * 8 != FROZEN_PROFILE_BITS
        or frozen_manifest.get("block_length") != BLOCK_LENGTH
        or frozen_manifest.get("levels") != 6
        or frozen_manifest.get("bytes") != len(frozen_payload)
        or frozen_manifest.get("bits") != FROZEN_PROFILE_BITS
        or frozen_manifest.get("parameters") != expected_frozen_parameters
        or hashlib.sha256(frozen_payload).hexdigest() != frozen_manifest.get("sha256")
        or len(frozen_manifest.get("per_level", [])) != 6
    ):
        raise AssertionError("serialized frozen profile does not match the charged ledger bits")
    level_bytes = BLOCK_LENGTH // 8
    for level0, row in enumerate(frozen_manifest["per_level"]):
        payload = frozen_payload[level0 * level_bytes : (level0 + 1) * level_bytes]
        frozen_count = sum(byte.bit_count() for byte in payload)
        if row != {
            "level": level0 + 1,
            "frozen": frozen_count,
            "open": BLOCK_LENGTH - frozen_count,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }:
            raise AssertionError(f"serialized frozen profile level {level0 + 1} mismatch")

    nonrouter_bits = inventory["nonrouter_rank2_blocks"] * POLAR_SLOT_BYTES * 8
    router_bits = int(router["container_bits"])
    rank1_bits = inventory["rank1_parameters"] * 16
    tensor_header_bits = inventory["tensor_count"] * PER_TENSOR_HEADER_BITS
    total_bits = (
        nonrouter_bits
        + router_bits
        + rank1_bits
        + tensor_header_bits
        + FROZEN_PROFILE_BITS
        + GLOBAL_HEADER_BITS
    )
    total_parameters = inventory["total_parameters"]
    rate = total_bits / total_parameters
    gaussian = 2.0 ** (-2.0 * rate)
    threshold = 10.0 ** (TARGET_GAP_DB / 10.0) * gaussian

    # The six adjacent down-projection confirmations are the controlled proxy.
    projection_tensor = "model.layers.27.mlp.experts.57.down_proj.weight"
    confirmation = [
        row
        for row in polar_rows
        if row["canonical_tensor"] == projection_tensor
        and row["canonical_block_index"] in range(6)
        and row["file"].startswith("agent_root_polar_escape_final_l27e57down_b")
    ]
    if (
        len(confirmation) != 6
        or {row["canonical_block_index"] for row in confirmation} != set(range(6))
    ):
        raise AssertionError(f"expected six final down-projection rows, got {len(confirmation)}")
    confirmation_energy = sum(row["source_energy"] for row in confirmation)
    confirmation_sse = sum(row["sse"] for row in confirmation)
    mean_energy = confirmation_energy / len(confirmation)
    mean_sse = confirmation_sse / len(confirmation)
    projected_nonrouter_energy = mean_energy * inventory["nonrouter_rank2_blocks"]
    projected_nonrouter_sse = mean_sse * inventory["nonrouter_rank2_blocks"]
    mixed_energy = projected_nonrouter_energy + float(router["source_energy"])
    mixed_sse = projected_nonrouter_sse + float(router["sse"])
    projected_mse = mixed_sse / mixed_energy
    projected_gap = 10.0 * math.log10(projected_mse / gaussian)

    implementation_paths = [
        Path("agent_root_polar_lattice_gate.py"),
        Path("agent_polar_codec_audit_independent_decoder.py"),
        Path("agent_router_adaptive_q234.py"),
        Path(__file__),
    ]
    missing_implementations = [path for path in implementation_paths if not path.exists()]
    if missing_implementations:
        raise AssertionError(f"missing implementation artifacts: {missing_implementations}")
    implementations = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in implementation_paths
    }
    decoder_path = Path("agent_polar_codec_audit_independent_decoder.py")
    decoder_tree = ast.parse(decoder_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(decoder_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(decoder_tree)
        if isinstance(node, ast.ImportFrom)
    }
    if "agent_root_polar_lattice_gate" in imported_modules:
        raise AssertionError("independent decoder imports the encoder module")
    standalone_paths = sorted(Path(".").glob("agent_root_polar_escape_*_standalone_decode.json"))
    expected_standalone_names = {
        "agent_root_polar_escape_normative_primary_standalone_decode.json",
        "agent_root_polar_escape_normative_embed_b0_standalone_decode.json",
        *{
            f"agent_root_polar_escape_final_l27e57down_b{block}_standalone_decode.json"
            for block in range(6)
        },
    }
    if {path.name for path in standalone_paths} != expected_standalone_names:
        raise AssertionError("standalone decoder evidence set is incomplete or contains extras")
    row_by_report = {row["file"]: row for row in polar_rows}
    standalone_rows = []
    for path in standalone_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            audit.get("decoded_reconstruction_matches_encoder_metric_at_1e_12") is not True
            or audit.get("decoded_indices_match_encoder_metric_at_1e_12") is not True
            or audit.get("tail_escape_padding_zero") is not True
            or audit.get("conditional_slot_budget_compatibility", {}).get(
                "fits_conditional_fixed_slot_budget"
            ) is not True
            or audit.get("conditional_slot_budget_compatibility", {}).get(
                "realized_checkpoint_packer_exercised"
            ) is not False
        ):
            raise AssertionError(f"standalone decoder failed for {path}")
        report_name = path.name.replace("_standalone_decode.json", ".json")
        evidence_row = row_by_report.get(report_name)
        if evidence_row is None:
            raise AssertionError(f"standalone report lacks pinned encoder evidence: {path}")
        if (
            audit["container_sha256"] != evidence_row["container_sha256"]
            or int(audit["container_bytes"]) != evidence_row["container_bytes"]
            or audit["source_block_bf16_sha256"] != evidence_row["source_block_sha256"]
            or not math.isclose(
                float(audit["decoded_relative_mse_with_serialized_scale"]),
                evidence_row["relative_mse"],
                abs_tol=1e-12,
            )
        ):
            raise AssertionError(f"standalone evidence provenance mismatch for {path}")
        standalone_rows.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "container_sha256": audit["container_sha256"],
                "container_bytes": audit["container_bytes"],
                "tail_escape_count": audit["tail_escape_count"],
                "decoded_relative_mse": audit[
                    "decoded_relative_mse_with_serialized_scale"
                ],
                "exact_metric_match": audit[
                    "decoded_reconstruction_matches_encoder_metric_at_1e_12"
                ],
            }
        )

    result = {
        "architecture": "fixed-slot-budget-neutral sparse-tail entropy-coded polar lattice plus literal all-Q4 routers",
        "strict_ptq": True,
        "training_retraining_qat_or_calibration_optimization": False,
        "target": {"rate_bpw_lt": TARGET_BPW, "gaussian_gap_db_lt": TARGET_GAP_DB},
        "checkpoint": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            **inventory,
        },
        "literal_codec": {
            "polar_block_length": BLOCK_LENGTH,
            "polar_slot_bytes": POLAR_SLOT_BYTES,
            "polar_header": "u32: low20 arithmetic bits, high12 sparse-tail count; then FP32 scale",
            "tail_record": "18-bit absolute index plus exact 16-bit BF16 word",
            "tail_selection": "stable largest decoded squared errors, limited to unused bytes in the slot",
            "frozen_profile_artifact": {
                "path": frozen_profile_path.name,
                "manifest": frozen_manifest_path.name,
                "bits": frozen_manifest["bits"],
                "sha256": frozen_manifest["sha256"],
            },
            "router_format": router_document["literal_router_codec"],
        },
        "conditional_fixed_slot_rate_budget": {
            "scope": "arithmetically exact budget if every nonrouter block is encodable within the enforced 81242-byte slot; not a realized packed checkpoint",
            "nonrouter_polar_bits": nonrouter_bits,
            "router_container_bits": router_bits,
            "rank1_lossless_bf16_bits": rank1_bits,
            "tensor_headers_64b_each": tensor_header_bits,
            "six_raw_frozen_masks_bits": FROZEN_PROFILE_BITS,
            "global_format_header_bits": GLOBAL_HEADER_BITS,
            "total_bits": total_bits,
            "total_bytes": total_bits // 8,
            "global_bpw": rate,
            "headroom_to_2p5_bpw": TARGET_BPW - rate,
            "headroom_bits": int(TARGET_BPW * total_parameters - total_bits),
            "gaussian_limit_relative_mse": gaussian,
            "threshold_relative_mse_at_0p10db": threshold,
            "budget_fits_below_2p5_if_all_blocks_are_encodable": rate < TARGET_BPW,
        },
        "sample_based_distortion_projection": {
            "status": "projection, not a whole-checkpoint MSE measurement",
            "controlled_nonrouter_blocks": len(confirmation),
            "controlled_nonrouter_energy": confirmation_energy,
            "controlled_nonrouter_sse": confirmation_sse,
            "controlled_energy_weighted_relative_mse": confirmation_sse / confirmation_energy,
            "router_scope": "all 48 routers measured exactly",
            "mixed_projected_relative_mse": projected_mse,
            "mixed_projected_gap_db": projected_gap,
            "passes_0p10db_projection": projected_gap < TARGET_GAP_DB,
            "all_six_projection_containers_clean_decoded": sum(
                "l27e57down" in row["file"] for row in standalone_rows
            ) == 6,
        },
        "polar_evidence": {
            "manifest": {
                "path": args.evidence_manifest.name,
                "bytes": args.evidence_manifest.stat().st_size,
                "sha256": sha256_file(args.evidence_manifest),
                "encoder_sha256": evidence_manifest["encoder_sha256"],
                "frozen_profile_sha256": evidence_manifest["frozen_profile_sha256"],
            },
            "rows": polar_rows,
            "observed_rows": len(polar_rows),
            "observed_unique_source_blocks": len(
                {(row["canonical_tensor"], row["canonical_block_index"]) for row in polar_rows}
            ),
            "checkpoint_nonrouter_block_coverage_fraction": len(
                {(row["canonical_tensor"], row["canonical_block_index"]) for row in polar_rows}
            ) / inventory["nonrouter_rank2_blocks"],
            "checkpoint_nonrouter_blocks_unencoded": inventory["nonrouter_rank2_blocks"]
            - len(
                {(row["canonical_tensor"], row["canonical_block_index"]) for row in polar_rows}
            ),
            "observed_container_max_bytes": observed_max,
            "observed_gap_max_db": max(row["gap_db"] for row in polar_rows),
            "all_available_decoder_audits": all(
                row["all_decoder_audits"] for row in polar_rows
            ),
            "frozen_seed_regeneration_audited_rows": sum(
                row["frozen_seed_regeneration_audited"] for row in polar_rows
            ),
            "frozen_seed_regeneration_missing_rows": sum(
                not row["frozen_seed_regeneration_audited"] for row in polar_rows
            ),
            "all_observed_containers_within_slot": all(
                row["container_bytes"] <= POLAR_SLOT_BYTES for row in polar_rows
            ),
        },
        "implementation_artifacts": implementations,
        "standalone_decoder_evidence": {
            "scope": "two normative exemplars plus all six controlled-projection blocks; not a concatenated whole-checkpoint stream",
            "imports_encoder_module": False,
            "inputs": "polar bytes, JSON profile metadata, serialized six-mask artifact; frozen BF16 source only for MSE audit",
            "rows": standalone_rows,
            "all_exact_metric_matches": all(row["exact_metric_match"] for row in standalone_rows),
        },
        "claim_limits": [
            "The rate number is a conditional fixed-slot budget, not a realized checkpoint filesize; no concatenated packer currently emits the slot padding, tensor headers, or global header.",
            "A production encoder must reject or route any base arithmetic stream already larger than the slot; no overflow route is implemented.",
            "The mixed distortion and gap extrapolate six controlled nonrouter blocks; only the 48-router aggregate is checkpoint-complete.",
            "A definitive checkpoint MSE requires encoding every nonrouter block and accumulating original-energy-weighted SSE.",
            "Portable decoding requires pinned floating-point/CuPy behavior or replacement of probability generation by normative fixed-point lookup tables.",
            "Every manifest-pinned final stream and both normative exemplars were regenerated by the pinned current encoder and exercise explicit frozen-seed regeneration, source-block hashing, and literal-container hashing.",
            "A clean-process decoder consumes serialized masks and reproduces both normative exemplars and all six projection blocks exactly, but no concatenated whole-checkpoint packer/decoder has been exercised.",
            "The construction is a literal two-set MAP realization inspired by the authors' simulation, not a claim to reproduce their fixed-length F/I/S Construction-D code.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
