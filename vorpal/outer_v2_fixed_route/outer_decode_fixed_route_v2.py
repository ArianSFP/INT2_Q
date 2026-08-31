#!/usr/bin/env python3
"""Independent decoder entry for the isolated fixed-route side codec v2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import fixed_route_codec as route_codec


BASE = route_codec.load_v1_outer()
NORMATIVE_BLOCK_LENGTH = BASE.NORMATIVE_BLOCK_LENGTH


def read_bundle_prelude(handle):
    return route_codec.read_bundle_prelude_v2(BASE, handle)


def read_all_frames(handle, count):
    return BASE.read_all_frames(handle, count)


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
        raise ValueError(f"{flag} is required by fixed-route wrapper") from error


def atomic_rewrite(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".fixed-route.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    bundle_path = argument_path("--bundle")
    receipt_path = argument_path("--receipt")
    # Reuse the audited chunk decoder/scatter path but replace only its bundle
    # prelude reader.  Keep BASE.__file__ untouched so the original receipt
    # truthfully binds the delegated audited v1 decoder; add this wrapper and
    # codec as separate dependencies below.
    dependency_bindings = route_codec.dependency_bindings(
        BASE,
        "fixed_route_decoder_wrapper_sha256",
        Path(__file__),
    )
    BASE.read_bundle_prelude = read_bundle_prelude
    BASE.main()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed":
        raise AssertionError("base decode did not pass")
    if (
        receipt.get("decoder_assets", {}).get("outer_decoder_sha256")
        != dependency_bindings["audited_v1_outer_decoder_sha256"]
    ):
        raise AssertionError("delegated v1 decoder receipt hash mismatch")
    with bundle_path.open("rb") as handle:
        prelude = read_bundle_prelude(handle)
    payload = prelude.compressed_side
    canonical_xz = payload[: -route_codec.ROUTE_BYTES]
    route = payload[-route_codec.ROUTE_BYTES :]
    canonical = BASE.decompress_lzma_xz_exact(canonical_xz, len(prelude.side.blob))
    reconstructed = route_codec.reconstruct_literal_side(BASE, canonical, route)
    if reconstructed != prelude.side.blob:
        raise AssertionError("post-decode route reconstruction mismatch")

    receipt["format"] = (
        "continuous reverse-waterfilled PLTE independent outer decode "
        "fixed-route experiment v2"
    )
    receipt["experimental_not_v1"] = True
    receipt["encoded_stream"]["layout"] = (
        "WFOUTR01 168-byte header, XZ(canonical all-A64 WFPLTE01) + exact "
        "route400, BZ2 six-mask, raw self-delimiting PLTE containers"
    )
    receipt["outer_bundle"].update(
        {
            "side_codec": "codec 3: XZ(canonical all-A64 side) + route400",
            "side_codec_id": route_codec.SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400,
            "canonical_side_raw_bytes": len(canonical),
            "canonical_side_raw_sha256": hashlib.sha256(canonical).hexdigest(),
            "canonical_side_xz_bytes": len(canonical_xz),
            "canonical_side_xz_sha256": hashlib.sha256(canonical_xz).hexdigest(),
            "route_bits": route_codec.ROUTE_BITS,
            "route_bytes": len(route),
            "route_sha256": hashlib.sha256(route).hexdigest(),
            "reconstructed_literal_side_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "reconstructed_literal_side_hash_verified": True,
            "canonical_profile_offsets_verified": True,
            "alphabet_domain": [64, 128],
            "decompressor_exact_eof": True,
        }
    )
    receipt["decoder_assets"].update(dependency_bindings)
    receipt["dependency_bindings"] = dependency_bindings
    atomic_rewrite(receipt_path, receipt)
    print(
        json.dumps(
            {
                "fixed_route_receipt": str(receipt_path),
                "status": receipt["status"],
                "physical_prelude_bytes": receipt["encoded_stream"][
                    "physical_prelude_bytes"
                ],
                "actual_all_in_bpw": receipt["encoded_stream"]["actual_all_in_bpw"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
