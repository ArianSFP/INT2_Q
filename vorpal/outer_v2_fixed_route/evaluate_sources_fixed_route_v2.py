#!/usr/bin/env python3
"""Exact-source evaluator wrapper for isolated fixed-route decoder receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import fixed_route_codec as route_codec
import outer_decode_fixed_route_v2 as fixed_decoder


EXPECTED_DECODE_FORMAT = (
    "continuous reverse-waterfilled PLTE independent outer decode "
    "fixed-route experiment v2"
)
PINNED_V1_EVALUATOR_SHA256 = (
    "1fa3ba98529860d2e900b89d188f5451bf7b2b63becfb7b89469cf07f9b75f52"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def argument_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing {flag}") from error


def load_v1_evaluator():
    outer_path = Path(route_codec.load_v1_outer().__file__).resolve()
    path = outer_path.with_name("evaluate_sources.py")
    actual_hash = sha256_path(path)
    if actual_hash != PINNED_V1_EVALUATOR_SHA256:
        raise ValueError(
            f"unaudited v1 evaluator {path}: {actual_hash} != "
            f"{PINNED_V1_EVALUATOR_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("_wfoutr_v1_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class DecoderProxy:
    __file__ = fixed_decoder.__file__
    NORMATIVE_BLOCK_LENGTH = fixed_decoder.NORMATIVE_BLOCK_LENGTH
    read_bundle_prelude = staticmethod(fixed_decoder.read_bundle_prelude)
    read_all_frames = staticmethod(fixed_decoder.read_all_frames)


def evaluation_dependency_bindings(evaluator) -> dict[str, str]:
    decoder_dependencies = route_codec.dependency_bindings(
        fixed_decoder.BASE,
        "fixed_route_decoder_wrapper_sha256",
        Path(fixed_decoder.__file__),
    )
    evaluator_hash = sha256_path(Path(evaluator.__file__))
    if evaluator_hash != PINNED_V1_EVALUATOR_SHA256:
        raise ValueError("delegated v1 evaluator changed after audited import")
    return {
        "fixed_route_evaluator_wrapper_sha256": sha256_path(Path(__file__)),
        "delegated_v1_evaluator_sha256": evaluator_hash,
        "pinned_v1_evaluator_sha256": PINNED_V1_EVALUATOR_SHA256,
        **decoder_dependencies,
    }


def replace_argument(argv: list[str], flag: str, value: str) -> list[str]:
    result = list(argv)
    index = result.index(flag)
    result[index + 1] = value
    return result


def main() -> None:
    original_receipt_path = argument_path("--decode-receipt")
    output_path = argument_path("--output")
    original = json.loads(original_receipt_path.read_text(encoding="utf-8"))
    if original.get("format") != EXPECTED_DECODE_FORMAT:
        raise ValueError("decode receipt is not fixed-route experiment v2")
    if (
        original.get("experimental_not_v1") is not True
        or original.get("outer_bundle", {}).get("side_codec_id")
        != route_codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400
        or original.get("outer_bundle", {}).get(
            "reconstructed_literal_side_hash_verified"
        )
        is not True
    ):
        raise ValueError("decode receipt lacks fixed-route audit evidence")

    decoder_dependencies = route_codec.dependency_bindings(
        fixed_decoder.BASE,
        "fixed_route_decoder_wrapper_sha256",
        Path(fixed_decoder.__file__),
    )
    route_codec.validate_dependency_bindings(
        original.get("dependency_bindings"), decoder_dependencies
    )
    decoder_assets = original.get("decoder_assets", {})
    if any(
        decoder_assets.get(key) != value
        for key, value in decoder_dependencies.items()
    ) or (
        decoder_assets.get("outer_decoder_sha256")
        != decoder_dependencies["audited_v1_outer_decoder_sha256"]
    ):
        raise ValueError("decode receipt dependency assets do not match loaded files")

    translated = json.loads(json.dumps(original))
    translated["format"] = "continuous reverse-waterfilled PLTE independent outer decode v1"
    # The delegated evaluator checks its decoder module against this temporary
    # adapter.  The immutable original above remains bound separately to the
    # audited v1 decoder, fixed wrapper, and fixed-route codec.
    translated["decoder_assets"]["outer_decoder_sha256"] = sha256_path(
        Path(fixed_decoder.__file__)
    )
    with tempfile.TemporaryDirectory() as directory:
        translated_path = Path(directory) / "translated.decode.receipt.json"
        translated_path.write_text(json.dumps(translated), encoding="utf-8")
        evaluator = load_v1_evaluator()
        evaluator_dependencies = evaluation_dependency_bindings(evaluator)
        evaluator.dec = DecoderProxy
        old_argv = sys.argv
        try:
            sys.argv = replace_argument(
                old_argv, "--decode-receipt", str(translated_path)
            )
            evaluator.main()
        finally:
            sys.argv = old_argv

    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["format"] = (
        "continuous reverse-waterfilled PLTE exact-source evaluation "
        "fixed-route experiment v2"
    )
    result["experimental_not_v1"] = True
    result["decode_receipt_sha256"] = sha256_path(original_receipt_path)
    result["loaded_outer_decoder_sha256"] = sha256_path(
        Path(fixed_decoder.__file__)
    )
    result["evaluation_dependency_bindings"] = evaluator_dependencies
    result["fixed_route_codec"] = {
        "side_codec_id": route_codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400,
        "route_bits": route_codec.ROUTE_BITS,
        "literal_side_hash_verified_before_source_scoring": True,
    }
    temporary = Path(str(output_path) + ".fixed-route.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": result["status"],
                "actual_all_in_bpw": result["actual_all_in_bpw"],
                "gaussian_reference_gap_db": result[
                    "gaussian_reference_gap_db"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
