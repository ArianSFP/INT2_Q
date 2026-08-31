#!/usr/bin/env python3
"""All-400 adaptive candidate runner, isolated V3 audit copy.

The trigger is evaluated over every one of the 400 validated base gaps. A64
triggers receive an A128 upgrade plus sparse tails. A128 triggers receive
sparse tails only; their optional ``upgrade`` field is JSON null. No base
alphabet is silently omitted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


PINNED_RUNNER_CORE_SHA256 = "348a1d14799a2e13e6dcd5e054704ac7c0c8ed712a4075dd5b55eddd55a13c07"
EXPECTED_CHUNKS = 400


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_core() -> Path:
    here = Path(__file__).resolve().parent.parent
    candidates = (
        here / "adaptive_candidate_audit" / "run_adaptive_candidates.py",
        here / "int2_adaptive_candidate_audit" / "run_adaptive_candidates.py",
    )
    for candidate in candidates:
        if candidate.is_file() and sha256_path(candidate) == PINNED_RUNNER_CORE_SHA256:
            return candidate
    raise FileNotFoundError(
        "pinned hardened runner core not found; expected SHA-256 "
        f"{PINNED_RUNNER_CORE_SHA256} in one of {candidates}"
    )


CORE_PATH = find_core()
SPEC = importlib.util.spec_from_file_location("adaptive_runner_core_v2", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def trigger_all_validated(validated: list[dict], threshold: float) -> list[dict]:
    """Return exactly all canonical base rows with gap strictly above threshold."""

    if len(validated) != EXPECTED_CHUNKS:
        raise AssertionError("trigger universe must contain exactly 400 validated rows")
    if not math.isfinite(threshold):
        raise ValueError("trigger threshold must be finite")
    result = []
    expected_indices = []
    for canonical_index, row in enumerate(validated):
        index = int(row["chunk"]["chunk_index"])
        if index != canonical_index:
            raise AssertionError("validated trigger universe is not canonical")
        alphabet = int(row["chunk"]["alphabet_size"])
        if alphabet not in (64, 128):
            raise ValueError(f"unsupported base alphabet {alphabet} at chunk {index}")
        gap = float(row["trial"]["gap_db"])
        if not math.isfinite(gap):
            raise ValueError(f"non-finite base gap at chunk {index}")
        if gap > threshold:
            expected_indices.append(index)
            result.append(row)
    actual_indices = [int(row["chunk"]["chunk_index"]) for row in result]
    if actual_indices != expected_indices:
        raise AssertionError("trigger list differs from exact all-400 predicate")
    return result


def run_tail_candidates(args, manifest: dict, base_row: dict) -> dict:
    """Emit or resume tails for either an A64 or A128 base row."""

    chunk = base_row["chunk"]
    index = int(chunk["chunk_index"])
    base_report = base_row["report_path"]
    base_container = base_row["container_path"]
    tail_dir = args.output_dir / "tails" / f"wf-{index:03d}"
    log_dir = args.output_dir / "logs"
    tail_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    requested_ks = tuple(sorted(set(args.tail_ks)))
    tail_report = tail_dir / f"wf-{index:03d}-tail-prefixes.json"
    if not CORE.valid_tails(
        tail_report,
        base_container,
        requested_ks,
        args.tail_ranking,
        args,
        manifest,
        chunk,
        base_report,
    ):
        command = [
            str(args.python),
            str(args.repacker),
            "--decoder",
            str(args.decoder),
            "--raw-mask",
            str(args.raw_mask),
            "--manifest",
            str(args.manifest),
            "--chunk-index",
            str(index),
            "--base-metadata",
            str(base_report),
            "--base-container",
            str(base_container),
            "--repo",
            str(args.repo),
            "--output-dir",
            str(tail_dir),
            "--ranking",
            args.tail_ranking,
            "--ks",
            *(str(value) for value in requested_ks),
        ]
        CORE.run_logged(command, log_dir / f"wf-{index:03d}-tails.log", args.repo)
    if not CORE.valid_tails(
        tail_report,
        base_container,
        requested_ks,
        args.tail_ranking,
        args,
        manifest,
        chunk,
        base_report,
    ):
        raise AssertionError(f"invalid tail candidates for chunk {index}")
    return CORE.load_json(tail_report)


def normalize_v2_a64_row(row: dict) -> dict:
    """Convert the hardened V2 A64 result to the explicit V3 option schema."""

    upgrade = row["a128"]
    if upgrade.get("independent_decode_passed") is not True:
        raise AssertionError("A64 upgrade lacks a literal clean-decode pass")
    return {
        "chunk_index": int(row["chunk_index"]),
        "trigger_gap_db": float(row["trigger_gap_db"]),
        "base_alphabet_size": 64,
        "available_option_kinds": ["base", "alphabet-upgrade", "tail"],
        "base": {"alphabet_size": 64, **row["base"]},
        "upgrade": {
            **upgrade,
            "kind": "alphabet-upgrade",
            "from_alphabet_size": 64,
            "to_alphabet_size": 128,
            "independent_decode_passed": True,
        },
        "tails": row["tails"],
    }


def generate_one_v3(args, manifest: dict, base_row: dict) -> dict:
    chunk = base_row["chunk"]
    index = int(chunk["chunk_index"])
    alphabet = int(chunk["alphabet_size"])
    if alphabet == 64:
        # The pinned hardened core supplies atomic A128 generation, tail
        # generation, clean A128 scoring, and all resume bindings.
        return normalize_v2_a64_row(CORE.generate_one(args, manifest, base_row))
    if alphabet != 128:
        raise ValueError(f"unsupported base alphabet {alphabet} at chunk {index}")

    # A128 is already the base. Generate tails only and expose null upgrade so
    # selectors cannot accidentally duplicate the base as an upgrade.
    tails = run_tail_candidates(args, manifest, base_row)
    if tails.get("base_decoded_with_clean_decoder") is not True:
        raise AssertionError("A128 tail report lacks a literal clean base decode pass")
    base_report = base_row["report_path"]
    base_container = base_row["container_path"]
    return {
        "chunk_index": index,
        "trigger_gap_db": float(base_row["trial"]["gap_db"]),
        "base_alphabet_size": 128,
        "available_option_kinds": ["base", "tail"],
        "base": {
            "alphabet_size": 128,
            "report": str(base_report),
            "container": str(base_container),
            "container_bytes": base_container.stat().st_size,
            "container_sha256": sha256_path(base_container),
            "raw_source_energy": float(tails["raw_source_energy"]),
            "raw_sse": float(tails["base_raw_sse"]),
            "independent_clean_decode_passed": True,
        },
        "upgrade": None,
        "tails": tails["rows"],
    }


def validate_v3_row_schema(row: dict, expected_base_alphabet: int) -> None:
    if int(row["base_alphabet_size"]) != expected_base_alphabet:
        raise AssertionError("V3 row base alphabet mismatch")
    if int(row["base"]["alphabet_size"]) != expected_base_alphabet:
        raise AssertionError("V3 base record alphabet mismatch")
    if expected_base_alphabet == 64:
        upgrade = row["upgrade"]
        if not (
            row["available_option_kinds"] == ["base", "alphabet-upgrade", "tail"]
            and upgrade is not None
            and upgrade["kind"] == "alphabet-upgrade"
            and int(upgrade["from_alphabet_size"]) == 64
            and int(upgrade["to_alphabet_size"]) == 128
            and upgrade["independent_decode_passed"] is True
        ):
            raise AssertionError("A64 V3 upgrade schema invalid")
    elif expected_base_alphabet == 128:
        if not (
            row["available_option_kinds"] == ["base", "tail"]
            and row["upgrade"] is None
            and row["base"]["independent_clean_decode_passed"] is True
        ):
            raise AssertionError("A128 V3 tail-only schema invalid")
    else:
        raise ValueError(expected_base_alphabet)
    if not isinstance(row["tails"], list) or not row["tails"]:
        raise AssertionError("V3 row must contain a nonempty tail option list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--repacker", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--trigger-gap-db", type=float, default=0.10)
    parser.add_argument(
        "--tail-ks",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 30, 60, 120, 240, 480],
    )
    parser.add_argument(
        "--tail-ranking", choices=("raw-gain", "normalized"), default="raw-gain"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.trigger_gap_db):
        raise ValueError("trigger threshold must be finite")
    args.tail_ks = sorted(set(int(k) for k in args.tail_ks))
    if not args.tail_ks or any(k <= 0 or k >= (1 << 12) for k in args.tail_ks):
        raise ValueError(f"invalid tail prefix lengths: {args.tail_ks}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    for path in (
        args.manifest,
        args.encoder,
        args.repacker,
        args.scorer,
        args.decoder,
        args.raw_mask,
        args.python,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.base_dir.is_dir() or not args.repo.is_dir() or not args.polar_repo.is_dir():
        raise NotADirectoryError("base, repository, or polar-reference directory missing")

    manifest = CORE.load_json(args.manifest)
    # Preserve the V2 safety gate: no adaptive output exists until the complete
    # hash-bound 400-row base receipt and every physical pair validate.
    validated = CORE.validate_complete_base_run(args, manifest)
    triggered = trigger_all_validated(validated, args.trigger_gap_db)
    trigger_indices = [int(row["chunk"]["chunk_index"]) for row in triggered]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(generate_one_v3, args, manifest, row): int(
                row["chunk"]["chunk_index"]
            )
            for row in triggered
        }
        for count, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            index = future_map[future]
            try:
                produced = future.result()
                validate_v3_row_schema(
                    produced, int(validated[index]["chunk"]["alphabet_size"])
                )
                rows.append(produced)
                CORE.progress(f"[{count}/{len(triggered)}] chunk {index:03d} complete")
            except BaseException as error:
                failures.append({"chunk_index": index, "error": repr(error)})
                CORE.progress(
                    f"[{count}/{len(triggered)}] chunk {index:03d} FAILED {error!r}"
                )
    rows.sort(key=lambda row: int(row["chunk_index"]))
    if not failures and [int(row["chunk_index"]) for row in rows] != trigger_indices:
        raise AssertionError("candidate row set differs from exact all-400 trigger set")

    base_receipt = args.base_receipt or (args.base_dir / "run.receipt.json")
    a64_count = sum(int(row["chunk"]["alphabet_size"]) == 64 for row in triggered)
    a128_count = sum(int(row["chunk"]["alphabet_size"]) == 128 for row in triggered)
    receipt = {
        "format": "continuous PLTE all-base adaptive candidate receipt v3",
        "status": "complete" if not failures and len(rows) == len(triggered) else "failed",
        "strict_ptq": True,
        "training_or_retraining": False,
        "implementation_sha256": sha256_path(Path(__file__)),
        "pinned_runner_core_sha256": PINNED_RUNNER_CORE_SHA256,
        "pinned_repacker_core_sha256": (
            "122685abed7626320cf1f0e51b9578674ec763ce6982ff0594ed0c25ff0e0ebc"
        ),
        "manifest_sha256": sha256_path(args.manifest),
        "base_receipt_sha256": sha256_path(base_receipt),
        "base_receipt_status": "complete",
        "encoder_sha256": sha256_path(args.encoder),
        "repacker_sha256": sha256_path(args.repacker),
        "scorer_sha256": sha256_path(args.scorer),
        "decoder_sha256": sha256_path(args.decoder),
        "raw_mask_sha256": sha256_path(args.raw_mask),
        "base_reports_scanned": len(validated),
        "scanned_chunk_indices": list(range(EXPECTED_CHUNKS)),
        "trigger_gap_db_strictly_greater_than": args.trigger_gap_db,
        "trigger_predicate_universe": "all 400 canonical validated base gaps",
        "triggered_chunk_indices": trigger_indices,
        "triggered_base_alphabet_counts": {"64": a64_count, "128": a128_count},
        "row_schema": {
            "base_alphabet_size": "required: 64 or 128",
            "base": "required and explicitly carries alphabet_size",
            "upgrade": "A64: required A128 object; A128: null",
            "tails": "required prefixes against the base alphabet",
        },
        "tail_prefixes": args.tail_ks,
        "tail_ranking": args.tail_ranking,
        "rows": rows,
        "failures": failures,
    }
    output = args.output_dir / "candidate.receipt.json"
    CORE.atomic_write_json(output, receipt)
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "rows"},
            indent=2,
            allow_nan=False,
        )
    )
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
