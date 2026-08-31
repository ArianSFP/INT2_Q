#!/usr/bin/env python3
"""Resume-safe adaptive PLTE candidate generator for all completed base chunks.

The runner never modifies ``base_dir``. It scans all 400 base reports and
triggers on measured normalized gap. A64 bases receive A128 and sparse-tail
candidates; A128 bases receive tail-only candidates, with tails preserving the
base alphabet. Tail candidates use exact original-coordinate SSE-gain ranking
and retain the base arithmetic payload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path


PRINT_LOCK = threading.Lock()


def progress(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log: Path, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def base_artifacts(base_dir: Path, index: int) -> tuple[Path, Path]:
    stem = base_dir / f"wf-{index:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".polar.bin")


def validate_base(report_path: Path, container_path: Path, chunk: dict) -> dict:
    report = load_json(report_path)
    parameters = report["parameters"]
    trial = report["trials"][0]
    if not (
        int(parameters["block_length"]) == 1 << 18
        and int(parameters["alphabet_size"]) == int(chunk["alphabet_size"])
        and int(parameters["container_cap_bytes"]) == 0
        and float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"])
        and float(parameters["eta"]) == float(chunk["eta"])
        and trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"]
        and int(trial["literal_container_bytes"]) == container_path.stat().st_size
        and trial["literal_container_sha256"] == sha256_path(container_path)
    ):
        raise AssertionError(f"invalid base artifacts for chunk {chunk['chunk_index']}")
    return report


def valid_a128(report_path: Path, container_path: Path, chunk: dict) -> bool:
    try:
        report = load_json(report_path)
        parameters = report["parameters"]
        trial = report["trials"][0]
        return (
            int(parameters["alphabet_size"]) == 128
            and int(parameters["container_cap_bytes"]) == 0
            and float(parameters["test_channel_distortion"]) == float(chunk["test_distortion"])
            and float(parameters["eta"]) == float(chunk["eta"])
            and trial["source"]["block_bf16_sha256"] == chunk["normalized_source_sha256"]
            and int(trial["literal_container_bytes"]) == container_path.stat().st_size
            and trial["literal_container_sha256"] == sha256_path(container_path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_decode(path: Path, container_path: Path, index: int) -> bool:
    try:
        row = load_json(path)
        return (
            row["status"] == "passed"
            and int(row["chunk_index"]) == index
            and row["container_sha256"] == sha256_path(container_path)
            and float(row["raw_source_energy"]) > 0.0
            and float(row["raw_sse"]) >= 0.0
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_tails(
    report_path: Path,
    base_container: Path,
    requested_ks: tuple[int, ...],
    ranking: str,
) -> bool:
    try:
        report = load_json(report_path)
        expected_ranking = (
            "descending original-coordinate SSE gain, stable ordinal ties"
            if ranking == "raw-gain"
            else "descending normalized squared residual, stable ordinal ties"
        )
        rows = report["rows"]
        return (
            report["base_container_sha256"] == sha256_path(base_container)
            and report["stable_ranking"] == expected_ranking
            and tuple(int(row["escape_count"]) for row in rows) == requested_ks
            and all(
                Path(row["container_path"]).is_file()
                and Path(row["container_path"]).stat().st_size == int(row["container_bytes"])
                and sha256_path(Path(row["container_path"])) == row["container_sha256"]
                and row["payload_unchanged"] is True
                for row in rows
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def generate_one(args, manifest: dict, chunk: dict) -> dict:
    index = int(chunk["chunk_index"])
    base_report, base_container = base_artifacts(args.base_dir, index)
    base = validate_base(base_report, base_container, chunk)
    base_alphabet = int(chunk["alphabet_size"])
    if base_alphabet not in (64, 128):
        raise AssertionError(f"unsupported base alphabet {base_alphabet} for chunk {index}")
    a128_dir = args.output_dir / "a128"
    tail_dir = args.output_dir / "tails" / f"wf-{index:03d}"
    log_dir = args.output_dir / "logs"
    for directory in (a128_dir, tail_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    a128_report = a128_dir / f"wf-{index:03d}-a128.json"
    a128_container = a128_dir / f"wf-{index:03d}-a128.polar.bin"
    if base_alphabet == 64:
        if not valid_a128(a128_report, a128_container, chunk):
            command = [
                str(args.python),
                str(args.encoder),
                "--polar-repo",
                str(args.polar_repo),
                "--input-bf16",
                str(chunk["normalized_source"]),
                "--test-distortion",
                repr(float(chunk["test_distortion"])),
                "--eta",
                repr(float(chunk["eta"])),
                "--alphabet-size",
                "128",
                "--container-cap-bytes",
                "0",
                "--emit-container-hex",
                "--output",
                str(a128_report),
            ]
            run_logged(command, log_dir / f"wf-{index:03d}-a128.log", args.repo)
        if not valid_a128(a128_report, a128_container, chunk):
            raise AssertionError(f"invalid A128 candidate for chunk {index}")

    tail_report = tail_dir / f"wf-{index:03d}-tail-prefixes.json"
    requested_ks = tuple(sorted(set(args.tail_ks)))
    if not valid_tails(tail_report, base_container, requested_ks, args.tail_ranking):
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
        run_logged(command, log_dir / f"wf-{index:03d}-tails.log", args.repo)
    if not valid_tails(tail_report, base_container, requested_ks, args.tail_ranking):
        raise AssertionError(f"invalid tail candidates for chunk {index}")

    a128_decode = a128_dir / f"wf-{index:03d}-a128.decode.json"
    if base_alphabet == 64:
        if not valid_decode(a128_decode, a128_container, index):
            command = [
                str(args.python),
                str(args.scorer),
                "--decoder",
                str(args.decoder),
                "--raw-mask",
                str(args.raw_mask),
                "--manifest",
                str(args.manifest),
                "--chunk-index",
                str(index),
                "--metadata",
                str(a128_report),
                "--container",
                str(a128_container),
                "--repo",
                str(args.repo),
                "--output",
                str(a128_decode),
            ]
            run_logged(command, log_dir / f"wf-{index:03d}-a128-decode.log", args.repo)
        if not valid_decode(a128_decode, a128_container, index):
            raise AssertionError(f"invalid A128 raw decode for chunk {index}")

    tails = load_json(tail_report)
    result = {
        "chunk_index": index,
        "trigger_gap_db": float(base["trials"][0]["gap_db"]),
        "base": {
            "alphabet_size": base_alphabet,
            "report": str(base_report),
            "container": str(base_container),
            "container_bytes": base_container.stat().st_size,
            "container_sha256": sha256_path(base_container),
            "raw_source_energy": float(tails["raw_source_energy"]),
            "raw_sse": float(tails["base_raw_sse"]),
        },
        "tails": tails["rows"],
    }
    if base_alphabet == 64:
        decoded = load_json(a128_decode)
        result["a128"] = {
            "report": str(a128_report),
            "container": str(a128_container),
            "decode": str(a128_decode),
            "container_bytes": a128_container.stat().st_size,
            "container_sha256": sha256_path(a128_container),
            "raw_source_energy": float(decoded["raw_source_energy"]),
            "raw_sse": float(decoded["raw_sse"]),
        }
    else:
        result["a128"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
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
    parser.add_argument("--tail-ranking", choices=("raw-gain", "normalized"), default="raw-gain")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    triggered: list[dict] = []
    scanned = []
    for chunk in manifest["chunks"]:
        index = int(chunk["chunk_index"])
        report_path, container_path = base_artifacts(args.base_dir, index)
        if not report_path.is_file() or not container_path.is_file():
            raise FileNotFoundError(f"missing completed base artifacts for chunk {index}")
        report = validate_base(report_path, container_path, chunk)
        gap = float(report["trials"][0]["gap_db"])
        scanned.append(index)
        if gap > args.trigger_gap_db:
            triggered.append(chunk)
    rows = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(generate_one, args, manifest, chunk): int(chunk["chunk_index"])
            for chunk in triggered
        }
        for count, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            index = future_map[future]
            try:
                rows.append(future.result())
                progress(f"[{count}/{len(triggered)}] chunk {index:03d} complete")
            except BaseException as error:
                failures.append({"chunk_index": index, "error": repr(error)})
                progress(f"[{count}/{len(triggered)}] chunk {index:03d} FAILED {error!r}")
    rows.sort(key=lambda row: int(row["chunk_index"]))
    receipt = {
        "format": "continuous PLTE adaptive candidate receipt v2",
        "status": "complete" if not failures and len(rows) == len(triggered) else "failed",
        "strict_ptq": True,
        "manifest_sha256": sha256_path(args.manifest),
        "base_reports_scanned": len(scanned),
        "base_alphabet_census": {
            "64": sum(int(chunk["alphabet_size"]) == 64 for chunk in manifest["chunks"]),
            "128": sum(int(chunk["alphabet_size"]) == 128 for chunk in manifest["chunks"]),
        },
        "candidate_policy": {
            "base_A64": "base + A128 + arbitrary validated tail prefixes",
            "base_A128": "base + arbitrary validated tail prefixes; no A256",
            "tails_preserve_base_alphabet": True,
        },
        "scanned_chunk_indices": scanned,
        "trigger_gap_db_strictly_greater_than": args.trigger_gap_db,
        "triggered_chunk_indices": [int(chunk["chunk_index"]) for chunk in triggered],
        "tail_prefixes": sorted(set(args.tail_ks)),
        "tail_ranking": args.tail_ranking,
        "rows": rows,
        "failures": failures,
    }
    output = args.output_dir / "candidate.receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
