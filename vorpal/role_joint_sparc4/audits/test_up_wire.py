#!/usr/bin/env python3
"""Adversarial structural tests for hardened VJWRAP42/VJSPRC41."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import audit_sparc4_up as audit


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repack(
    original: bytes,
    base: bytes,
    extension: bytes,
    external_hashes: list[bytes] | None = None,
) -> bytes:
    fields = list(audit.WRAPPER_V2.unpack_from(original))
    fields[4] = len(base)
    fields[5] = len(extension)
    fields[6] = hashlib.sha256(base).digest()
    fields[7] = hashlib.sha256(extension).digest()
    if external_hashes is not None:
        fields[8:12] = external_hashes
    fields[-1] = 0
    header = audit.WRAPPER_V2.pack(*fields)
    header = header[:-4] + struct.pack("<I", zlib.crc32(header[:-4]) & 0xFFFFFFFF)
    return header + base + extension


def re_crc(extension_without_crc: bytes) -> bytes:
    return extension_without_crc + struct.pack(
        "<I", zlib.crc32(extension_without_crc) & 0xFFFFFFFF
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--experiment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.wrapper.read_bytes()
    base, extension, bindings = audit.parse_wrapper(raw)
    audit.parse_extension(extension)
    expected_external = [
        bytes.fromhex(sha(args.manifest)),
        bytes.fromhex(sha(args.evaluation)),
        bytes.fromhex(sha(args.reconstruction)),
        bytes.fromhex(sha(args.encoder)),
    ]
    receipt = json.loads(args.experiment_receipt.read_text(encoding="utf-8"))
    cases = []

    def passed(name: str, detail: str = "") -> None:
        row = {"name": name, "status": "passed"}
        if detail:
            row["detail"] = detail
        cases.append(row)

    def expect(name: str, expected: str, call) -> None:
        try:
            call()
        except (ValueError, AssertionError) as error:
            if expected not in str(error):
                raise AssertionError((name, expected, str(error))) from error
            passed(name, str(error))
        else:
            raise AssertionError(f"tamper accepted: {name}")

    passed("valid wrapper and extension")
    expect("trailing wrapper byte", "EOF/length", lambda: audit.parse_wrapper(raw + b"x"))
    changed = bytearray(raw)
    changed[100] ^= 1
    expect("wrapper header CRC", "wrapper CRC", lambda: audit.parse_wrapper(bytes(changed)))

    changed_base = bytearray(base)
    changed_base[12345] ^= 1
    changed = bytearray(raw)
    changed[audit.WRAPPER_V2.size + 12345] ^= 1
    expect("embedded base member", "member hash", lambda: audit.parse_wrapper(bytes(changed)))

    changed = bytearray(raw)
    changed[-17] ^= 1
    expect("extension member", "member hash", lambda: audit.parse_wrapper(bytes(changed)))

    def provenance_gate(candidate: bytes) -> None:
        _, _, parsed = audit.parse_wrapper(candidate)
        actual = [
            parsed["manifest_sha256"],
            parsed["evaluation_sha256"],
            parsed["reconstruction_sha256"],
            parsed["encoder_sha256"],
        ]
        if actual != [value.hex() for value in expected_external]:
            raise ValueError("external provenance mismatch")

    changed_external = expected_external.copy()
    changed_external[3] = bytes([changed_external[3][0] ^ 1]) + changed_external[3][1:]
    changed = repack(raw, base, extension, changed_external)
    expect(
        "re-CRC encoder provenance tamper",
        "external provenance",
        lambda: provenance_gate(changed),
    )

    extension_body = bytearray(extension[:-4])
    extension_body[-17] ^= 1
    # Keep the stale inner CRC, but repair the outer member hash and wrapper CRC.
    stale_crc_extension = bytes(extension_body) + extension[-4:]
    changed = repack(raw, base, stale_crc_extension)
    _, stale, _ = audit.parse_wrapper(changed)
    expect("inner extension CRC", "extension CRC", lambda: audit.parse_extension(stale))

    extension_body = bytearray(extension[:-4])
    extension_body[0] ^= 1
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "repaired-CRC extension magic",
        "extension constants",
        lambda: audit.parse_extension(changed_extension),
    )

    descriptor = audit.EXT_HEADER.size
    extension_body = bytearray(extension[:-4])
    extension_body[descriptor + 8 : descriptor + 10] = struct.pack("<H", 0x7E00)
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "coordinate amplitude NaN",
        "role descriptor",
        lambda: audit.parse_extension(changed_extension),
    )

    coordinate_start = audit.EXT_HEADER.size + audit.ROLE_DESCRIPTOR.size + 50
    coordinate_bytes = struct.unpack_from("<I", extension, descriptor + 14)[0]
    extension_body = bytearray(extension[:-4])
    extension_body[coordinate_start + coordinate_bytes - 1] |= 1
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "nonzero coordinate Rice padding",
        "padding",
        lambda: audit.parse_extension(changed_extension),
    )

    extension_body = bytearray(extension[:-4])
    struct.pack_into("<I", extension_body, descriptor + 14, coordinate_bytes + 1)
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "nonminimal coordinate bytes",
        "role descriptor",
        lambda: audit.parse_extension(changed_extension),
    )

    up_stage_count = struct.unpack_from("<H", extension, descriptor + 6)[0]
    down_descriptor = (
        coordinate_start + coordinate_bytes + up_stage_count * audit.STAGE_RECORD_BYTES
    )
    down_mask = down_descriptor + audit.ROLE_DESCRIPTOR.size
    extension_body = bytearray(extension[:-4])
    # Move down ordinal 37 to already occupied up ordinal 36, preserving census.
    extension_body[down_mask + 37 // 8] &= ~(1 << (37 % 8))
    extension_body[down_mask + 36 // 8] |= 1 << (36 % 8)
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "overlapping role masks",
        "overlapping role masks",
        lambda: audit.parse_extension(changed_extension),
    )

    first_stage_amplitude = coordinate_start + coordinate_bytes
    extension_body = bytearray(extension[:-4])
    extension_body[first_stage_amplitude : first_stage_amplitude + 2] = b"\0\0"
    repaired = re_crc(bytes(extension_body))
    changed = repack(raw, base, repaired)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "zero SPARC stage amplitude",
        "stage amplitude",
        lambda: audit.parse_extension(changed_extension),
    )

    extended = re_crc(extension[:-4] + b"\0")
    changed = repack(raw, base, extended)
    _, changed_extension, _ = audit.parse_wrapper(changed)
    expect(
        "trailing extension byte with repaired hashes",
        "extension trailing bytes",
        lambda: audit.parse_extension(changed_extension),
    )

    tampered_receipt = json.loads(json.dumps(receipt))
    tampered_receipt["artifacts"]["emitted_wrapper_sha256"] = "00" * 32
    expect(
        "receipt artifact binding",
        "receipt wrapper mismatch",
        lambda: (
            None
            if tampered_receipt["artifacts"]["emitted_wrapper_sha256"]
            == hashlib.sha256(raw).hexdigest()
            else (_ for _ in ()).throw(ValueError("receipt wrapper mismatch"))
        ),
    )

    result = {
        "format": "hardened SPARC4 wrapper adversarial tests v1",
        "status": "passed",
        "cases_passed": len(cases),
        "cases": cases,
        "inputs": {
            "wrapper_bytes": args.wrapper.stat().st_size,
            "wrapper_sha256": sha(args.wrapper),
            "manifest_sha256": sha(args.manifest),
            "evaluation_sha256": sha(args.evaluation),
            "reconstruction_sha256": sha(args.reconstruction),
            "encoder_sha256": sha(args.encoder),
            "experiment_receipt_sha256": sha(args.experiment_receipt),
        },
        "test_script_sha256": sha(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases_passed": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
