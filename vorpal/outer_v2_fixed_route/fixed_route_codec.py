#!/usr/bin/env python3
"""Shared wire helpers for the isolated fixed-route WFOUTR side experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import BinaryIO


SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400 = 3
ROUTE_BITS = 400
ROUTE_BYTES = ROUTE_BITS // 8
PROFILE_ALPHABET_OFFSET = 16
PINNED_V1_OUTER_DECODER_SHA256 = (
    "15417800e16598b1fefe68b96796b5812b8294c0e53fc58a3092db3f6286b8fa"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for payload in iter(lambda: handle.read(1 << 20), b""):
            digest.update(payload)
    return digest.hexdigest()


def validate_dependency_bindings(
    bindings: object, expected: dict[str, str]
) -> None:
    """Require an exact dependency map with canonical literal hash values."""

    if not isinstance(bindings, dict) or bindings != expected:
        raise ValueError("fixed-route dependency bindings mismatch")


def _v1_outer_candidates() -> list[Path]:
    override = os.environ.get("WFOUTR_V1_OUTER_DECODE")
    rows = []
    if override:
        rows.append(Path(override))
    here = Path(__file__).resolve()
    rows.extend(
        [
            here.parents[2] / "outer_decoder_impl_v1" / "outer_decode.py",
            Path("/root/negative_gap_outer_decoder_v1/outer_decode.py"),
            Path("/root/patched_outer_reaudit/outer_decode.py"),
        ]
    )
    return rows


def load_v1_outer() -> ModuleType:
    for path in _v1_outer_candidates():
        if not path.is_file():
            continue
        actual_hash = sha256_path(path)
        if actual_hash != PINNED_V1_OUTER_DECODER_SHA256:
            raise ValueError(
                f"unaudited v1 outer decoder {path}: {actual_hash} != "
                f"{PINNED_V1_OUTER_DECODER_SHA256}"
            )
        name = "_wfoutr_fixed_route_v1_base"
        cached = sys.modules.get(name)
        if cached is not None and Path(cached.__file__).resolve() == path.resolve():
            return cached
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(
        "v1 outer_decode.py not found; set WFOUTR_V1_OUTER_DECODE"
    )


def dependency_bindings(
    base: ModuleType, entrypoint_key: str, entrypoint_path: Path
) -> dict[str, str]:
    base_path = Path(base.__file__).resolve()
    base_hash = sha256_path(base_path)
    if base_hash != PINNED_V1_OUTER_DECODER_SHA256:
        raise ValueError("loaded v1 outer decoder changed after audited import")
    return {
        entrypoint_key: sha256_path(entrypoint_path.resolve()),
        "fixed_route_codec_sha256": sha256_path(Path(__file__).resolve()),
        "audited_v1_outer_decoder_sha256": base_hash,
        "pinned_v1_outer_decoder_sha256": PINNED_V1_OUTER_DECODER_SHA256,
    }


def profile_code_offsets(base: ModuleType, blob: bytes) -> tuple[int, ...]:
    """Derive profile-code offsets from the canonical WFPLTE01 header."""

    if len(blob) < base.HEADER.size:
        raise ValueError("side shorter than WFPLTE01 header")
    (
        magic,
        version,
        block_count,
        _groups_per_chunk,
        _group_values,
        chunk_count,
        label_count,
        _lambda_variance,
        lut_count,
    ) = base.HEADER.unpack_from(blob)
    if magic != base.MAGIC or version != base.VERSION:
        raise ValueError("unsupported WFPLTE01 side header")
    if chunk_count != ROUTE_BITS:
        raise ValueError(f"fixed route requires exactly {ROUTE_BITS} chunks")
    if base.side_size_from_header(blob[: base.HEADER.size]) != len(blob):
        raise ValueError("WFPLTE01 side length/header mismatch")
    label_bytes = (int(label_count) * int(base.LABEL_BITS) + 7) // 8
    profile_start = (
        base.HEADER.size
        + int(lut_count) * 8
        + int(block_count) * 4
        + label_bytes
    )
    expected_eof = profile_start + int(chunk_count) * base.PROFILE.size
    if expected_eof != len(blob):
        raise ValueError("profile table is not at the canonical WFPLTE01 offset")
    return tuple(
        profile_start + index * base.PROFILE.size + PROFILE_ALPHABET_OFFSET
        for index in range(int(chunk_count))
    )


def route_from_literal_side(base: ModuleType, literal_side: bytes) -> tuple[bytes, bytes]:
    """Return canonical all-A64 side and exact 400-bit A64/A128 route."""

    base.parse_side(literal_side)
    canonical = bytearray(literal_side)
    route = bytearray(ROUTE_BYTES)
    for index, offset in enumerate(profile_code_offsets(base, literal_side)):
        code = canonical[offset]
        if code not in (0, 1):
            raise ValueError(
                f"chunk {index} alphabet code {code} is not A64/A128"
            )
        if code:
            route[index >> 3] |= 1 << (index & 7)
        canonical[offset] = 0
    parsed = base.parse_side(bytes(canonical))
    if len(parsed.profiles) != ROUTE_BITS or any(
        profile.alphabet_size != 64 for profile in parsed.profiles
    ):
        raise AssertionError("canonical side is not exactly 400 all-A64 profiles")
    return bytes(canonical), bytes(route)


def reconstruct_literal_side(
    base: ModuleType, canonical_side: bytes, route: bytes
) -> bytes:
    """Apply an exact LSB-first 400-bit route to canonical all-A64 profiles."""

    if len(route) != ROUTE_BYTES:
        raise ValueError(f"route is {len(route)} bytes, expected exactly {ROUTE_BYTES}")
    literal = bytearray(canonical_side)
    offsets = profile_code_offsets(base, canonical_side)
    for index, offset in enumerate(offsets):
        if literal[offset] != 0:
            raise ValueError(f"canonical profile {index} is not A64")
        literal[offset] = (route[index >> 3] >> (index & 7)) & 1
    parsed = base.parse_side(bytes(literal))
    if len(parsed.profiles) != ROUTE_BITS or any(
        profile.alphabet_size not in (64, 128) for profile in parsed.profiles
    ):
        raise AssertionError("route reconstructed a non-A64/A128 profile")
    return bytes(literal)


def encode_side_payload(base: ModuleType, literal_side: bytes) -> dict[str, bytes]:
    canonical, route = route_from_literal_side(base, literal_side)
    import lzma

    compressed = lzma.compress(canonical, format=lzma.FORMAT_XZ, preset=9)
    if base.decompress_lzma_xz_exact(compressed, len(canonical)) != canonical:
        raise AssertionError("canonical XZ round trip failed")
    payload = compressed + route
    reconstructed = reconstruct_literal_side(base, canonical, route)
    if reconstructed != literal_side:
        raise AssertionError("fixed route did not reproduce literal side")
    return {
        "canonical": canonical,
        "route": route,
        "canonical_xz": compressed,
        "payload": payload,
    }


def decode_side_payload(
    base: ModuleType,
    payload: bytes,
    expected_raw_bytes: int,
    expected_raw_sha256: bytes,
) -> tuple[object, bytes, bytes, bytes]:
    if len(payload) <= ROUTE_BYTES:
        raise ValueError("fixed-route side payload is too short")
    compressed = payload[:-ROUTE_BYTES]
    route = payload[-ROUTE_BYTES:]
    canonical = base.decompress_lzma_xz_exact(compressed, expected_raw_bytes)
    literal = reconstruct_literal_side(base, canonical, route)
    if hashlib.sha256(literal).digest() != expected_raw_sha256:
        raise ValueError("reconstructed literal raw-side SHA256 mismatch")
    return base.parse_side(literal), canonical, compressed, route


def read_bundle_prelude_v2(base: ModuleType, handle: BinaryIO):
    header = base.read_exact(handle, base.OUTER_HEADER.size, "outer bundle header")
    (
        magic,
        version,
        header_bytes,
        side_codec,
        side_raw_bytes,
        side_payload_bytes,
        mask_codec,
        mask_raw_bytes,
        mask_compressed_bytes,
        side_raw_hash,
        side_payload_hash,
        mask_raw_hash,
        mask_compressed_hash,
    ) = base.OUTER_HEADER.unpack(header)
    if (magic, version, header_bytes) != (
        base.OUTER_MAGIC,
        base.OUTER_VERSION,
        base.OUTER_HEADER.size,
    ):
        raise ValueError("unsupported outer magic/version/header size")
    if side_codec != SIDE_CODEC_XZ_CANONICAL_A64_ROUTE400:
        raise ValueError(f"unsupported fixed-route side codec {side_codec}")
    if mask_codec != base.MASK_CODEC_BZ2:
        raise ValueError(f"unsupported mask codec {mask_codec}")
    if not base.HEADER.size <= side_raw_bytes <= base.MAX_SIDE_RAW_BYTES:
        raise ValueError("invalid literal raw-side length")
    if not ROUTE_BYTES < side_payload_bytes <= base.MAX_SIDE_COMPRESSED_BYTES:
        raise ValueError("invalid canonical-XZ-plus-route payload length")
    expected_mask_bytes = base.BASE_MASK_LEVELS * (
        (base.NORMATIVE_BLOCK_LENGTH + 7) // 8
    )
    if (
        mask_raw_bytes != expected_mask_bytes
        or not 0 < mask_compressed_bytes <= base.MAX_MASK_COMPRESSED_BYTES
    ):
        raise ValueError("invalid embedded mask lengths")
    payload = base.read_exact(handle, side_payload_bytes, "canonical XZ plus route400")
    compressed_mask = base.read_exact(
        handle, mask_compressed_bytes, "BZ2-compressed raw mask"
    )
    if hashlib.sha256(payload).digest() != side_payload_hash:
        raise ValueError("canonical-XZ-plus-route payload SHA256 mismatch")
    if hashlib.sha256(compressed_mask).digest() != mask_compressed_hash:
        raise ValueError("compressed mask SHA256 mismatch")
    side, _canonical, _xz, _route = decode_side_payload(
        base, payload, side_raw_bytes, side_raw_hash
    )
    raw_mask = base.decompress_bz2_exact(compressed_mask, mask_raw_bytes)
    if hashlib.sha256(raw_mask).digest() != mask_raw_hash:
        raise ValueError("raw mask SHA256 mismatch")
    return base.BundlePrelude(
        header=header,
        side=side,
        compressed_side=payload,
        raw_mask=raw_mask,
        compressed_mask=compressed_mask,
        side_codec=side_codec,
        mask_codec=mask_codec,
    )
