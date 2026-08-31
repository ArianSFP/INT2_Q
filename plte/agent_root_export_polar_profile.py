#!/usr/bin/env python3
"""Export the six frozen masks charged by the PLTE global rate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from agent_root_polar_lattice_gate import periodic_binary_capacity, reliability_freeze_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    parser.add_argument(
        "--output", type=Path, default=Path("agent_root_polar_escape_frozen_profiles.bin")
    )
    args = parser.parse_args()

    block_length = 1 << 18
    sigma_source = 3.0
    distortion = 0.29
    eta = 0.5989929996555583
    levels = 6
    sigma_recon = math.sqrt(sigma_source**2 - distortion)
    tilde_sigma = sigma_recon * math.sqrt(distortion) / sigma_source
    capacities = [
        periodic_binary_capacity(tilde_sigma / eta / (1 << level0))
        for level0 in range(levels)
    ]
    flags = reliability_freeze_flags(args.polar_repo, block_length, capacities)

    packed_levels = [np.packbits(flag.astype(np.uint8), bitorder="big").tobytes() for flag in flags]
    payload = b"".join(packed_levels)
    expected_bytes = levels * block_length // 8
    if len(payload) != expected_bytes:
        raise AssertionError((len(payload), expected_bytes))
    args.output.write_bytes(payload)

    manifest = {
        "format": "six concatenated N-bit masks, level 1 through 6, MSB-first within each byte",
        "semantics": "one means frozen; zero means transmitted/causally entropy-coded",
        "block_length": block_length,
        "levels": levels,
        "bytes": len(payload),
        "bits": 8 * len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameters": {
            "sigma_source": sigma_source,
            "test_channel_distortion": distortion,
            "eta": eta,
            "tilde_sigma": tilde_sigma,
            "capacity_schedule": capacities,
        },
        "per_level": [
            {
                "level": index + 1,
                "frozen": int(flag.sum()),
                "open": int((flag == 0).sum()),
                "sha256": hashlib.sha256(packed).hexdigest(),
            }
            for index, (flag, packed) in enumerate(zip(flags, packed_levels, strict=True))
        ],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
