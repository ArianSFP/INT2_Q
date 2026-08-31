#!/usr/bin/env python3
"""Retry only recognized PLTE cap overflows at their frozen reservoir tier."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
ENCODER = PLTE / "agent_root_polar_lattice_gate.py"
DEFAULT_MANIFEST = ROOT / "evaluation" / "qwen3_stratified_v1" / "manifest.json"
DEFAULT_PLAN = ROOT / "evaluation" / "qwen3_stratified_v1" / "reservoir_plan.json"
DEFAULT_WORKSPACE = ROOT / "tmp" / "qwen3_stratified_v1"
BLOCK_VALUES = 1 << 18
_PRINT_LOCK = threading.Lock()


def progress(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


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


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {path}") from error


def paths(workspace: Path, entry_id: str, tier: int) -> tuple[Path, Path, Path]:
    report = workspace / "jobs" / f"{entry_id}.json"
    container = report.with_suffix(".polar.bin")
    log = workspace / "logs" / f"{entry_id}.tier{tier}.encode.log"
    return report, container, log


def validate_result(
    report_path: Path,
    container_path: Path,
    allocation: dict[str, Any],
    source_sha256: str,
    encoder_sha256: str,
) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("implementation_sha256") != encoder_sha256:
        raise AssertionError("encoder SHA mismatch")
    if report.get("strict_ptq") is not True or report.get("source_training_or_retraining") is not False:
        raise AssertionError("strict PTQ declaration mismatch")
    parameters = report.get("parameters", {})
    expected = {
        "block_length": BLOCK_VALUES,
        "trials": 1,
        "sigma_source": 3.0,
        "test_channel_distortion": 0.29,
        "eta": 0.5989929996555583,
        "alphabet_size": 64,
        "decision": "map",
        "seed": 20260831,
        "container_cap_bytes": int(allocation["container_cap_bytes"]),
    }
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise AssertionError("retry parameters differ from frozen tier plan")
    trials = report.get("trials", [])
    if len(trials) != 1:
        raise AssertionError("retry did not emit exactly one trial")
    trial = trials[0]
    if trial.get("source", {}).get("block_bf16_sha256") != source_sha256:
        raise AssertionError("retry source SHA mismatch")
    if trial.get("base_literal_container_bytes") != allocation["first_pass_base_container_bytes"]:
        raise AssertionError("retry did not reproduce first-pass base length")
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
        "passes_rate_lt_2p5",
    )
    if any(trial.get(field) is not True for field in required_true):
        raise AssertionError("retry audit flag is false")
    container_bytes = container_path.stat().st_size
    if container_bytes > int(allocation["container_cap_bytes"]):
        raise AssertionError("retry container exceeds assigned tier")
    container_sha256 = sha256_path(container_path)
    if (
        trial.get("literal_container_bytes") != container_bytes
        or trial.get("literal_container_sha256") != container_sha256
    ):
        raise AssertionError("retry container metadata mismatch")
    if not all(
        math.isfinite(float(trial[key]))
        for key in ("relative_mse", "literal_decoded_absolute_mse", "gap_db")
    ):
        raise AssertionError("retry emitted non-finite metric")
    charged_rate = 8 * int(allocation["container_cap_bytes"]) / BLOCK_VALUES
    charged_gap = 10.0 * math.log10(
        float(trial["relative_mse"]) / (2.0 ** (-2.0 * charged_rate))
    )
    return {
        "id": allocation["id"],
        "tier": allocation["tier"],
        "container_cap_bytes": allocation["container_cap_bytes"],
        "reproduced_base_container_bytes": trial["base_literal_container_bytes"],
        "tail_escape_count": trial["tail_escape_count"],
        "container_bytes": container_bytes,
        "container_sha256": container_sha256,
        "report_sha256": sha256_path(report_path),
        "relative_mse": trial["relative_mse"],
        "literal_rate_gap_db": trial["gap_db"],
        "charged_tier_slot_gap_db": charged_gap,
        "passes_charged_tier_gap_lt_0p10db": charged_gap < 0.10,
    }


def repair_one(
    workspace: Path,
    allocation: dict[str, Any],
    source_sha256: str,
    encoder_sha256: str,
    python: Path,
    polar_repo: Path,
) -> dict[str, Any]:
    report, container, log = paths(workspace, allocation["id"], allocation["tier"])
    if report.is_file() and container.is_file():
        return validate_result(
            report, container, allocation, source_sha256, encoder_sha256
        )
    partial_report = report.with_name(f".{report.stem}.tier{allocation['tier']}.partial.json")
    partial_container = partial_report.with_suffix(".polar.bin")
    partial_report.unlink(missing_ok=True)
    partial_container.unlink(missing_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    source = workspace / "sources" / f"{allocation['id']}.bf16.bin"
    command = [
        str(python),
        str(ENCODER),
        "--polar-repo",
        str(polar_repo),
        "--block-length",
        str(BLOCK_VALUES),
        "--trials",
        "1",
        "--sigma-source",
        "3.0",
        "--test-distortion",
        "0.29",
        "--eta",
        "0.5989929996555583",
        "--alphabet-size",
        "64",
        "--decision",
        "map",
        "--seed",
        "20260831",
        "--input-bf16",
        relative_to_root(source),
        "--input-block-start",
        "0",
        "--container-cap-bytes",
        str(allocation["container_cap_bytes"]),
        "--emit-container-hex",
        "--output",
        str(partial_report),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"retry encoder exited {result.returncode}: {tail}")
    validated = validate_result(
        partial_report,
        partial_container,
        allocation,
        source_sha256,
        encoder_sha256,
    )
    os.replace(partial_report, report)
    os.replace(partial_container, container)
    validated["wall_seconds"] = time.perf_counter() - started
    validated["retry_log_sha256"] = sha256_path(log)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    plan = load_json(args.plan)
    if plan.get("format") != "PLTE Qwen3 checkpoint rate-reservoir plan v1":
        raise AssertionError("unexpected reservoir plan")
    if plan.get("selection_manifest_sha256") != sha256_path(args.manifest):
        raise AssertionError("reservoir plan targets a different selection manifest")
    if plan.get("encoder_sha256") != sha256_path(ENCODER):
        raise AssertionError("reservoir plan encoder SHA mismatch")
    allocations = [row for row in plan["allocations"] if int(row["tier"]) > 0]
    if not allocations:
        raise AssertionError("reservoir plan has no retry allocations")
    source_manifest = load_json(args.workspace / "source_manifest.json")
    source_records = {row["id"]: row for row in source_manifest["records"]}

    statuses = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                repair_one,
                args.workspace,
                allocation,
                source_records[allocation["id"]]["sha256"],
                manifest["provenance"]["encoder_sha256"],
                args.python,
                args.polar_repo,
            ): allocation["id"]
            for allocation in allocations
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            entry_id = futures[future]
            try:
                statuses.append(future.result())
            except BaseException as error:
                failures.append(
                    {"id": entry_id, "error": f"{type(error).__name__}: {error}"}
                )
            progress(
                f"reservoir retry: {index}/{len(allocations)}, "
                f"{len(failures)} failures"
            )
    output = {
        "format": "PLTE Qwen3 reservoir retry status v1",
        "reservoir_plan_sha256": sha256_path(args.plan),
        "attempted": len(allocations),
        "completed": len(statuses),
        "failures": failures,
        "all_charged_tier_gaps_lt_0p10db": bool(statuses)
        and all(row["passes_charged_tier_gap_lt_0p10db"] for row in statuses),
        "statuses": sorted(statuses, key=lambda row: str(row["id"])),
    }
    atomic_json(args.workspace / "reservoir_retry_status.json", output)
    print(json.dumps({key: value for key, value in output.items() if key != "statuses"}, indent=2))
    if failures:
        raise RuntimeError(f"reservoir retries failed for {len(failures)} entries")


if __name__ == "__main__":
    main()
