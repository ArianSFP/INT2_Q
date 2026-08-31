#!/usr/bin/env python3
"""V3 audit launcher for the pinned hardened tail-prefix implementation.

The V2 implementation already supports clean decoding and deterministic tails
for both A64 and A128 bases, arbitrary positive prefix lengths through the
12-bit header limit, physical reparsing, and raw-gain identities.  This small
launcher pins its exact bytes while giving V3 a distinct implementation hash.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


PINNED_CORE_SHA256 = "122685abed7626320cf1f0e51b9578674ec763ce6982ff0594ed0c25ff0e0ebc"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_core() -> Path:
    here = Path(__file__).resolve().parent.parent
    candidates = (
        here / "adaptive_candidate_audit" / "repack_tail_prefixes.py",
        here / "int2_adaptive_candidate_audit" / "repack_tail_prefixes.py",
    )
    for candidate in candidates:
        if candidate.is_file() and sha256_path(candidate) == PINNED_CORE_SHA256:
            return candidate
    raise FileNotFoundError(
        "pinned hardened repacker core not found; expected SHA-256 "
        f"{PINNED_CORE_SHA256} in one of {candidates}"
    )


CORE_PATH = find_core()
SPEC = importlib.util.spec_from_file_location("adaptive_tail_core_v2", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)

# Export the framing helpers so the V3 tests exercise the exact pinned core.
N = CORE.N
LOGICAL_LENGTH_BITS = CORE.LOGICAL_LENGTH_BITS
ESCAPE_RECORD_BITS = CORE.ESCAPE_RECORD_BITS
MAX_ESCAPE_RECORDS = CORE.MAX_ESCAPE_RECORDS
parse_container_bytes = CORE.parse_container_bytes


def pack_escape_records(positions, values) -> bytes:
    """Apply the physical 12-bit count limit before delegating to the core."""

    if len(positions) > MAX_ESCAPE_RECORDS:
        raise ValueError(
            f"{len(positions)} escapes exceed the {MAX_ESCAPE_RECORDS}-record header limit"
        )
    return CORE.pack_escape_records(positions, values)


def main() -> None:
    # The core writes an implementation hash through its module-global
    # ``__file__``. Bind that receipt field to this V3 launcher. This launcher
    # has already cryptographically pinned the executed core above.
    CORE.__file__ = str(Path(__file__).resolve())
    CORE.main()


if __name__ == "__main__":
    main()
