#!/usr/bin/env python3
"""Fail-closed, source-free verifier for the VORPAL 400-block publication.

This verifier deliberately uses only the Python standard library.  It does
not import the encoder, decoder, CuPy, NumPy, or any model/source loader.  It
reparses the fixed-route WFOUTR01 wire image, binds the publication receipts
to physical files, redoes the exposed multiple-choice Pareto DP, and
recomputes the reported aggregate and strata.

The verifier cannot prove facts for which the publication supplies no
portable evidence.  In particular, source values are intentionally absent;
the exact-source evaluator receipt remains the authority for individual SSE
measurements.  Cross-file hashes and all formulas derived from those numbers
are checked independently here.
"""

from __future__ import annotations

import argparse
import bz2
import copy
import hashlib
import json
import lzma
import math
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CHUNK_COUNT = 400
BLOCK_VALUES = 1 << 18
GROUP_VALUES = 1 << 11
GROUPS_PER_CHUNK = BLOCK_VALUES // GROUP_VALUES
PANEL_VALUES = CHUNK_COUNT * BLOCK_VALUES
LABEL_COUNT = CHUNK_COUNT * GROUPS_PER_CHUNK
LABEL_BITS = 6
PROFILE = struct.Struct("<ddB")
SIDE_HEADER = struct.Struct("<8sIIIIIIdI")
OUTER_HEADER = struct.Struct("<8s8I32s32s32s32s")
SIDE_MAGIC = b"WFPLTE01"
OUTER_MAGIC = b"WFOUTR01"
WIRE_VERSION = 1
SIDE_CODEC_FIXED_ROUTE = 3
MASK_CODEC_BZ2 = 2
ROUTE_BITS = 400
ROUTE_BYTES = ROUTE_BITS // 8
BASE_MASK_LEVELS = 6
MASK_RAW_BYTES = BASE_MASK_LEVELS * ((BLOCK_VALUES + 7) // 8)
MAX_EMBEDDED_BYTES = 1 << 20

TARGET_GAP_DB = Decimal("-0.10")
RATE_LIMIT_BPW = Decimal("2.5")

CHECKPOINT = {
    "repo": "Qwen/Qwen3-30B-A3B",
    "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
}

# These are the exact audited implementations inspected for this publication
# contract.  Any production change must intentionally update the verifier and
# its tests; silently accepting a different implementation is not claim-grade.
PINNED = {
    "clean_decoder": "7589f4be6e784d8e5a0067303da389b6d982430eb84fda52f668808f322c25d9",
    "raw_mask": "11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6",
    "v1_outer_decoder": "15417800e16598b1fefe68b96796b5812b8294c0e53fc58a3092db3f6286b8fa",
    "v1_evaluator": "1fa3ba98529860d2e900b89d188f5451bf7b2b63becfb7b89469cf07f9b75f52",
    "side_packer": "3ac0cdfa4c3363c96e0b84ab4ea630de4696bb8f8e1e21ff0fd534bf651b6966",
    "fixed_route_codec": "75f66fd98e7ba8567ee3cfc4b87e00dc175024c02c88072e19550af88458ca76",
    "fixed_route_bundle_packer": "558553be948dfc6d9a7ef0045fd5987fbec747ac396215df68d9780f29f86139",
    "fixed_route_outer_decoder": "fb9732b70663807aab54fcb8d5e32003318a295d6ffd9a38f89f45c4ff633833",
    "fixed_route_evaluator": "72fb36b3448af9bf829a862261b13ed1fbcba07c7a5c0f764c2276f8825f123f",
}

LAYERWISE_ROLES = (
    "mlp.experts.{expert}.down_proj.weight",
    "mlp.experts.{expert}.gate_proj.weight",
    "mlp.experts.{expert}.up_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
GLOBAL_ROLE_COUNTS = {
    "model.embed_tokens.weight": 32,
    "lm_head.weight": 32,
}

REQUIRED_FILENAMES = {
    "construction_manifest": "construction.manifest.json",
    "selected_manifest": "selected.manifest.json",
    "source_ledger": "source_ledger.json",
    "base_receipt": "base.run.receipt.json",
    "candidate_receipt": "candidate.receipt.json",
    "selection_receipt": "selection.receipt.json",
    "selection_checksum": "selection.receipt.sha256",
    "side": "side.bin",
    "side_receipt": "side.receipt.json",
    "bundle": "selected.wfouter",
    "bundle_receipt": "bundle.receipt.json",
    "decode_receipt": "decode.receipt.json",
    "evaluation": "evaluation.json",
}


class VerificationError(ValueError):
    """A publication invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(actual == keys, f"{label} keys differ: missing={sorted(keys-actual)}, extra={sorted(actual-keys)}")
    return value


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON number {token!r} in {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot load canonical JSON {path}: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not canonical lowercase SHA-256",
    )
    return value


def as_int(value: object, label: str, *, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    result = int(value)
    if minimum is not None:
        require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def as_decimal(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise VerificationError(f"{label} is not decimal") from error
    require(result.is_finite(), f"{label} must be finite")
    if positive:
        require(result > 0, f"{label} must be positive")
    if nonnegative:
        require(result >= 0, f"{label} must be nonnegative")
    return result


def close_decimal(
    actual: object,
    expected: Decimal,
    label: str,
    *,
    absolute: Decimal = Decimal("2e-12"),
    relative: Decimal = Decimal("2e-12"),
) -> None:
    value = as_decimal(actual, label)
    tolerance = max(absolute, relative * max(abs(value), abs(expected)))
    require(abs(value - expected) <= tolerance, f"{label} mismatch: {value} != {expected}")


def canonical_sha256(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(type(item).__name__)

    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode,
    ).encode("utf-8")
    return sha256_bytes(blob)


def safe_child(root: Path, relative: object, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label} must be a nonempty relative path")
    candidate_rel = Path(relative)
    require(not candidate_rel.is_absolute(), f"{label} must be relative")
    require(".." not in candidate_rel.parts, f"{label} traverses outside publication")
    candidate = root / candidate_rel
    require(candidate.is_file(), f"missing {label}: {candidate_rel}")
    require(not candidate.is_symlink(), f"{label} must not be a symlink")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise VerificationError(f"{label} resolves outside publication") from error
    return candidate


def require_artifact(
    root: Path,
    relative: object,
    expected_bytes: object,
    expected_hash: object,
    label: str,
) -> Path:
    path = safe_child(root, relative, label)
    size = as_int(expected_bytes, f"{label} bytes", minimum=0)
    digest = require_sha256(expected_hash, f"{label} SHA-256")
    require(path.stat().st_size == size, f"{label} physical byte count mismatch")
    require(sha256_path(path) == digest, f"{label} physical SHA-256 mismatch")
    return path


def read_exact(handle: BinaryIO, count: int, label: str) -> bytes:
    require(count >= 0, f"negative byte count for {label}")
    payload = handle.read(count)
    require(len(payload) == count, f"truncated {label}: got {len(payload)}, expected {count}")
    return payload


def decompress_lzma_exact(payload: bytes, expected_bytes: int) -> bytes:
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        result = decoder.decompress(payload, max_length=expected_bytes + 1)
    except lzma.LZMAError as error:
        raise VerificationError(f"invalid canonical side XZ: {error}") from error
    require(len(result) == expected_bytes, "canonical side XZ length mismatch")
    require(decoder.eof, "canonical side XZ lacks exact EOF")
    require(not decoder.unused_data, "canonical side XZ has trailing/concatenated data")
    return result


def decompress_bz2_exact(payload: bytes, expected_bytes: int) -> bytes:
    decoder = bz2.BZ2Decompressor()
    try:
        result = decoder.decompress(payload, max_length=expected_bytes + 1)
    except OSError as error:
        raise VerificationError(f"invalid embedded BZ2 mask: {error}") from error
    require(len(result) == expected_bytes, "embedded mask length mismatch")
    require(decoder.eof, "embedded BZ2 mask lacks exact EOF")
    require(not decoder.unused_data, "embedded BZ2 mask has trailing/concatenated data")
    return result


@dataclass(frozen=True)
class Frame:
    index: int
    literal: bytes
    logical_bits: int
    escape_count: int
    scale: float
    arithmetic_padding_bits: int
    tail_padding_bits: int
    sha256: str


@dataclass(frozen=True)
class ParsedBundle:
    header: bytes
    literal_side: bytes
    canonical_side: bytes
    canonical_xz: bytes
    route: bytes
    raw_mask: bytes
    compressed_mask: bytes
    frames: tuple[Frame, ...]
    block_count: int
    groups_per_chunk: int
    group_values: int
    chunk_count: int
    label_count: int
    profile_offset: int
    profile_distortions: tuple[float, ...]
    profile_etas: tuple[float, ...]
    alphabet_codes: tuple[int, ...]

    @property
    def physical_prelude_bytes(self) -> int:
        return len(self.header) + len(self.canonical_xz) + len(self.route) + len(self.compressed_mask)

    @property
    def container_bytes(self) -> int:
        return sum(len(frame.literal) for frame in self.frames)


def parse_side_geometry(blob: bytes) -> tuple[int, int, int, int, int, int]:
    require(len(blob) >= SIDE_HEADER.size, "literal side shorter than header")
    (
        magic,
        version,
        block_count,
        groups_per_chunk,
        group_values,
        chunk_count,
        label_count,
        lambda_variance,
        lut_count,
    ) = SIDE_HEADER.unpack_from(blob)
    require(magic == SIDE_MAGIC and version == WIRE_VERSION, "unsupported WFPLTE side magic/version")
    require(math.isfinite(lambda_variance) and lambda_variance > 0, "invalid side lambda variance")
    require(lut_count == 64, "fixed publication requires exactly 64 LUT entries")
    require(
        (block_count, groups_per_chunk, group_values, chunk_count, label_count)
        == (CHUNK_COUNT, GROUPS_PER_CHUNK, GROUP_VALUES, CHUNK_COUNT, LABEL_COUNT),
        "literal side geometry is not the frozen 400-block panel",
    )
    label_bytes = (label_count * LABEL_BITS + 7) // 8
    profile_offset = SIDE_HEADER.size + lut_count * 8 + block_count * 4 + label_bytes
    expected = profile_offset + chunk_count * PROFILE.size
    require(len(blob) == expected, "literal side does not end exactly after 400 profiles")

    lut_offset = SIDE_HEADER.size
    lut = struct.unpack_from("<64d", blob, lut_offset)
    require(all(math.isfinite(value) and value > 0 for value in lut), "side LUT is not finite positive")
    scale_offset = lut_offset + 64 * 8
    scales = struct.unpack_from(f"<{block_count}f", blob, scale_offset)
    require(all(math.isfinite(value) and value > 0 for value in scales), "side block scales are not finite positive")
    return block_count, groups_per_chunk, group_values, chunk_count, label_count, profile_offset


def read_frame(handle: BinaryIO, index: int) -> Frame:
    fixed = read_exact(handle, 8, f"chunk {index} header")
    header_word, scale = struct.unpack("<If", fixed)
    logical_bits = header_word & ((1 << 20) - 1)
    escape_count = header_word >> 20
    require(escape_count <= BLOCK_VALUES, f"chunk {index} escape count exceeds block length")
    require(math.isfinite(scale) and scale > 0, f"chunk {index} has invalid FP32 scale")
    payload_bytes = (logical_bits + 7) // 8
    tail_bytes = (34 * escape_count + 7) // 8
    payload = read_exact(handle, payload_bytes, f"chunk {index} arithmetic payload")
    tail = read_exact(handle, tail_bytes, f"chunk {index} sparse tail")
    arithmetic_padding = payload_bytes * 8 - logical_bits
    if arithmetic_padding:
        require(payload and not (payload[-1] & ((1 << arithmetic_padding) - 1)), f"chunk {index} arithmetic padding is nonzero")
    tail_padding = tail_bytes * 8 - 34 * escape_count
    if tail_padding:
        require(tail and not (tail[-1] & ((1 << tail_padding) - 1)), f"chunk {index} sparse-tail padding is nonzero")
    combined = int.from_bytes(tail, "big")
    if tail_padding:
        combined >>= tail_padding
    positions = [0] * escape_count
    record_mask = (1 << 34) - 1
    for offset in range(escape_count - 1, -1, -1):
        record = combined & record_mask
        combined >>= 34
        positions[offset] = record >> 16
    require(not combined, f"chunk {index} tail parser left meaningful bits")
    require(all(0 <= value < BLOCK_VALUES for value in positions), f"chunk {index} tail position out of range")
    require(all(left < right for left, right in zip(positions, positions[1:])), f"chunk {index} tail positions not increasing")
    literal = fixed + payload + tail
    return Frame(
        index=index,
        literal=literal,
        logical_bits=logical_bits,
        escape_count=escape_count,
        scale=scale,
        arithmetic_padding_bits=arithmetic_padding,
        tail_padding_bits=tail_padding,
        sha256=sha256_bytes(literal),
    )


def parse_bundle(path: Path) -> ParsedBundle:
    with path.open("rb") as handle:
        header = read_exact(handle, OUTER_HEADER.size, "WFOUTR01 header")
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
        ) = OUTER_HEADER.unpack(header)
        require((magic, version, header_bytes) == (OUTER_MAGIC, WIRE_VERSION, OUTER_HEADER.size), "unsupported WFOUTR01 header")
        require(side_codec == SIDE_CODEC_FIXED_ROUTE, "publication bundle is not fixed-route side codec 3")
        require(mask_codec == MASK_CODEC_BZ2, "publication bundle does not use BZ2 mask codec 2")
        require(SIDE_HEADER.size <= side_raw_bytes <= MAX_EMBEDDED_BYTES, "invalid raw-side length")
        require(ROUTE_BYTES < side_payload_bytes <= MAX_EMBEDDED_BYTES, "invalid fixed-route payload length")
        require(mask_raw_bytes == MASK_RAW_BYTES, "invalid embedded raw-mask length")
        require(0 < mask_compressed_bytes <= MAX_EMBEDDED_BYTES, "invalid compressed-mask length")
        side_payload = read_exact(handle, side_payload_bytes, "canonical XZ plus route400")
        compressed_mask = read_exact(handle, mask_compressed_bytes, "compressed raw mask")
        require(hashlib.sha256(side_payload).digest() == side_payload_hash, "fixed-route payload hash mismatch")
        require(hashlib.sha256(compressed_mask).digest() == mask_compressed_hash, "compressed-mask hash mismatch")
        canonical_xz, route = side_payload[:-ROUTE_BYTES], side_payload[-ROUTE_BYTES:]
        canonical = decompress_lzma_exact(canonical_xz, side_raw_bytes)
        raw_mask = decompress_bz2_exact(compressed_mask, mask_raw_bytes)
        require(hashlib.sha256(raw_mask).digest() == mask_raw_hash, "raw-mask hash mismatch")
        require(sha256_bytes(raw_mask) == PINNED["raw_mask"], "embedded mask is not the pinned normative mask")
        geometry = parse_side_geometry(canonical)
        block_count, groups_per_chunk, group_values, chunk_count, label_count, profile_offset = geometry
        literal = bytearray(canonical)
        distortions: list[float] = []
        etas: list[float] = []
        codes: list[int] = []
        for index in range(chunk_count):
            offset = profile_offset + index * PROFILE.size
            distortion, eta, canonical_code = PROFILE.unpack_from(canonical, offset)
            require(math.isfinite(distortion) and 0 < distortion < 9, f"profile {index} distortion invalid")
            require(math.isfinite(eta) and eta > 0, f"profile {index} eta invalid")
            require(canonical_code == 0, f"canonical profile {index} is not A64")
            code = (route[index >> 3] >> (index & 7)) & 1
            literal[offset + 16] = code
            distortions.append(distortion)
            etas.append(eta)
            codes.append(code)
        literal_bytes = bytes(literal)
        require(hashlib.sha256(literal_bytes).digest() == side_raw_hash, "reconstructed literal-side hash mismatch")
        frames = tuple(read_frame(handle, index) for index in range(chunk_count))
        require(not handle.read(1), "bundle has trailing bytes after exactly 400 containers")
    require(path.stat().st_size == OUTER_HEADER.size + side_payload_bytes + mask_compressed_bytes + sum(len(frame.literal) for frame in frames), "physical bundle partition mismatch")
    return ParsedBundle(
        header=header,
        literal_side=literal_bytes,
        canonical_side=canonical,
        canonical_xz=canonical_xz,
        route=route,
        raw_mask=raw_mask,
        compressed_mask=compressed_mask,
        frames=frames,
        block_count=block_count,
        groups_per_chunk=groups_per_chunk,
        group_values=group_values,
        chunk_count=chunk_count,
        label_count=label_count,
        profile_offset=profile_offset,
        profile_distortions=tuple(distortions),
        profile_etas=tuple(etas),
        alphabet_codes=tuple(codes),
    )


def validate_manifest_pair(
    construction_path: Path,
    selected_path: Path,
    ledger: dict[str, Any],
    bundle: ParsedBundle,
) -> tuple[dict[str, Any], dict[str, Any]]:
    construction = load_json(construction_path)
    selected = load_json(selected_path)
    for label, manifest in (("construction", construction), ("selected", selected)):
        require(isinstance(manifest, dict), f"{label} manifest must be an object")
        require(
            manifest.get("format") == "continuous reverse-waterfilled PLTE exploratory manifest v1",
            f"unsupported {label} manifest format",
        )
        require(manifest.get("strict_ptq") is True, f"{label} manifest is not strict PTQ")
        require(manifest.get("training_or_retraining") is False, f"{label} manifest permits training/retraining")
        parameters = manifest.get("parameters")
        census = manifest.get("census")
        require(isinstance(parameters, dict) and isinstance(census, dict), f"{label} manifest omits geometry")
        require(as_int(parameters.get("block_values"), f"{label} block_values") == BLOCK_VALUES, f"{label} block length mismatch")
        require(as_int(parameters.get("group_values"), f"{label} group_values") == GROUP_VALUES, f"{label} group length mismatch")
        require(as_int(parameters.get("groups_per_polar_block"), f"{label} groups per chunk") == GROUPS_PER_CHUNK, f"{label} groups-per-chunk mismatch")
        expected_census = {
            "source_blocks": CHUNK_COUNT,
            "groups": LABEL_COUNT,
            "polar_chunks": CHUNK_COUNT,
            "values": PANEL_VALUES,
        }
        require(
            all(as_int(census.get(key), f"{label} census {key}") == value for key, value in expected_census.items()),
            f"{label} panel census mismatch",
        )
        blocks = manifest.get("blocks")
        chunks = manifest.get("chunks")
        require(isinstance(blocks, list) and len(blocks) == CHUNK_COUNT, f"{label} manifest must contain 400 blocks")
        require(isinstance(chunks, list) and len(chunks) == CHUNK_COUNT, f"{label} manifest must contain 400 chunks")
        require([as_int(row.get("ordinal"), f"{label} block ordinal") for row in blocks] == list(range(CHUNK_COUNT)), f"{label} block order is not canonical")
        require([as_int(row.get("chunk_index"), f"{label} chunk index") for row in chunks] == list(range(CHUNK_COUNT)), f"{label} chunk order is not canonical")
        require(len({str(row.get("id")) for row in blocks}) == CHUNK_COUNT, f"{label} block IDs are not unique")
        for index, chunk in enumerate(chunks):
            alphabet = as_int(chunk.get("alphabet_size"), f"{label} chunk {index} alphabet")
            require(alphabet in (64, 128), f"{label} chunk {index} is outside fixed-route A64/A128 domain")
            require_sha256(chunk.get("normalized_source_sha256"), f"{label} chunk {index} normalized-source hash")

    # The fixed selector deep-copies the construction manifest and changes only
    # per-chunk alphabet_size.  Any other mutation is outside the audited path.
    selected_normalized = copy.deepcopy(selected)
    construction_normalized = copy.deepcopy(construction)
    for index in range(CHUNK_COUNT):
        selected_normalized["chunks"][index]["alphabet_size"] = construction_normalized["chunks"][index]["alphabet_size"]
    require(selected_normalized == construction_normalized, "selected manifest changed fields other than chunk alphabets")

    ledger_rows = ledger["blocks"]
    for ordinal, (block, ledger_row) in enumerate(zip(construction["blocks"], ledger_rows, strict=True)):
        require(
            str(block.get("id")) == str(ledger_row.get("id"))
            and str(block.get("tensor")) == str(ledger_row.get("tensor"))
            and str(block.get("role")) == str(ledger_row.get("role"))
            and as_int(block.get("block_index"), f"manifest block {ordinal} index")
            == as_int(ledger_row.get("block_index"), f"ledger block {ordinal} index")
            and require_sha256(block.get("source_sha256"), f"manifest block {ordinal} source hash")
            == require_sha256(ledger_row.get("sha256"), f"ledger block {ordinal} source hash"),
            f"construction manifest/source-ledger identity mismatch at block {ordinal}",
        )

    for index, chunk in enumerate(selected["chunks"]):
        require(float(as_decimal(chunk.get("test_distortion"), f"chunk {index} distortion")) == bundle.profile_distortions[index], f"chunk {index} distortion differs from physical side")
        require(float(as_decimal(chunk.get("eta"), f"chunk {index} eta")) == bundle.profile_etas[index], f"chunk {index} eta differs from physical side")
        expected_code = 0 if as_int(chunk.get("alphabet_size"), f"chunk {index} alphabet") == 64 else 1
        require(bundle.alphabet_codes[index] == expected_code, f"chunk {index} physical route differs from selected manifest")
    return construction, selected


def validate_source_ledger(path: Path) -> dict[str, Any]:
    ledger = load_json(path)
    require(isinstance(ledger, dict), "source ledger must be an object")
    require(ledger.get("format") == "canonical BF16 source ledger v1", "unsupported source-ledger format")
    require(ledger.get("evaluator_only") is True, "source ledger is not evaluator-only")
    require(ledger.get("checkpoint") == CHECKPOINT, "source ledger does not pin the normative Qwen checkpoint")
    require_sha256(ledger.get("selection_manifest_sha256"), "ledger selection-manifest hash")
    require_sha256(ledger.get("construction_manifest_sha256"), "ledger construction-manifest hash")
    rows = ledger.get("blocks")
    require(isinstance(rows, list) and len(rows) == CHUNK_COUNT, "source ledger must have exactly 400 rows")
    require([as_int(row.get("canonical_block_ordinal"), "ledger ordinal") for row in rows] == list(range(CHUNK_COUNT)), "source ledger is not in canonical order")
    ids = [str(row.get("id", "")) for row in rows]
    require(all(ids) and len(set(ids)) == CHUNK_COUNT, "source-ledger IDs must be nonempty and unique")

    role_layers: dict[str, list[object]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        require(isinstance(row.get("tensor"), str) and row["tensor"], f"ledger row {ordinal} has empty tensor")
        require(isinstance(row.get("role"), str) and row["role"], f"ledger row {ordinal} has empty role")
        layer = row.get("layer")
        require(layer is None or (isinstance(layer, int) and not isinstance(layer, bool)), f"ledger row {ordinal} layer is not integer/null")
        require(isinstance(row.get("path"), str) and row["path"], f"ledger row {ordinal} has empty source path")
        require_sha256(row.get("sha256"), f"ledger row {ordinal} source hash")
        role_layers[row["role"]].append(layer)

    expected_roles = set(LAYERWISE_ROLES) | set(GLOBAL_ROLE_COUNTS)
    require(set(role_layers) == expected_roles, "source ledger does not cover every canonical tensor role")
    for role in LAYERWISE_ROLES:
        require(sorted(role_layers[role]) == list(range(48)), f"role {role} does not cover every layer 0..47 exactly once")
    for role, count in GLOBAL_ROLE_COUNTS.items():
        require(role_layers[role] == [None] * count, f"global role {role} must contain exactly {count} layer-null blocks")
    return ledger


def validate_base_receipt(path: Path, construction_path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "base receipt must be an object")
    require(receipt.get("status") == "complete", "base run is not complete")
    require(receipt.get("all_internal_roundtrips_passed") is True, "base run lacks internal round trips")
    require(not receipt.get("failures"), "base run records failures")
    require(as_int(receipt.get("chunks"), "base chunks") == CHUNK_COUNT, "base run is not 400 chunks")
    require(receipt.get("manifest_sha256") == sha256_path(construction_path), "base receipt does not bind construction manifest")
    require_sha256(receipt.get("encoder_sha256"), "base encoder hash")
    rows = receipt.get("rows")
    require(isinstance(rows, list) and len(rows) == CHUNK_COUNT, "base receipt must have 400 rows")
    require(sorted(as_int(row.get("chunk_index"), "base row index") for row in rows) == list(range(CHUNK_COUNT)), "base receipt rows are missing/duplicated")
    total = sum(as_int(row.get("container_bytes"), "base container bytes", minimum=8) for row in rows)
    require(as_int(receipt.get("actual_container_bytes"), "base total bytes") == total, "base receipt total container bytes mismatch")
    return receipt


def validate_candidate_receipt(
    path: Path,
    construction_path: Path,
    base_path: Path,
    construction: dict[str, Any],
) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "candidate receipt must be an object")
    require(receipt.get("format") == "continuous PLTE all-base adaptive candidate receipt v3", "unsupported candidate format")
    require(receipt.get("status") == "complete" and not receipt.get("failures"), "candidate receipt is incomplete/failed")
    require(receipt.get("strict_ptq") is True, "candidate generation is not strict PTQ")
    require(receipt.get("training_or_retraining") is False, "candidate generation permits retraining")
    require(receipt.get("manifest_sha256") == sha256_path(construction_path), "candidate receipt does not bind construction manifest")
    require(receipt.get("base_receipt_sha256") == sha256_path(base_path), "candidate receipt does not bind base receipt")
    require(receipt.get("base_receipt_status") == "complete", "candidate receipt did not consume a complete base run")
    require(receipt.get("raw_mask_sha256") == PINNED["raw_mask"], "candidate receipt used a non-normative mask")
    for key in (
        "implementation_sha256",
        "pinned_runner_core_sha256",
        "pinned_repacker_core_sha256",
        "encoder_sha256",
        "repacker_sha256",
        "scorer_sha256",
        "decoder_sha256",
    ):
        require_sha256(receipt.get(key), f"candidate {key}")
    require(receipt.get("decoder_sha256") == PINNED["clean_decoder"], "candidate clean decoder is not pinned")
    indices = list(range(CHUNK_COUNT))
    require(receipt.get("scanned_chunk_indices") == indices, "candidate receipt did not scan canonical chunks 0..399")
    require(as_int(receipt.get("base_reports_scanned"), "candidate base reports") == CHUNK_COUNT, "candidate receipt did not scan 400 base reports")
    require(receipt.get("trigger_predicate_universe") == "all 400 canonical validated base gaps", "candidate trigger universe is not exhaustive")
    expected_row_schema = {
        "base_alphabet_size": "required: 64 or 128",
        "base": "required and explicitly carries alphabet_size",
        "upgrade": "A64: required A128 object; A128: null",
        "tails": "required prefixes against the base alphabet",
    }
    require(receipt.get("row_schema") == expected_row_schema, "candidate row schema is not hardened v3")
    require(as_decimal(receipt.get("trigger_gap_db_strictly_greater_than"), "candidate trigger") == Decimal("0.1"), "candidate trigger is not strict >0.10 dB")
    tail_prefixes = receipt.get("tail_prefixes")
    require(
        isinstance(tail_prefixes, list)
        and tail_prefixes
        and [as_int(value, "tail prefix", minimum=1) for value in tail_prefixes] == sorted(set(tail_prefixes)),
        "candidate tail-prefix schedule is invalid",
    )
    triggered = receipt.get("triggered_chunk_indices")
    rows = receipt.get("rows")
    require(isinstance(triggered, list) and triggered == sorted(set(triggered)), "candidate triggered indices are noncanonical")
    require(isinstance(rows, list) and [row.get("chunk_index") for row in rows] == triggered, "candidate rows differ from triggered indices")

    counts = {"64": 0, "128": 0}
    for row in rows:
        index = as_int(row.get("chunk_index"), "candidate row index")
        require(0 <= index < CHUNK_COUNT, "candidate row index outside panel")
        base_alphabet = as_int(row.get("base_alphabet_size"), f"candidate row {index} base alphabet")
        require(base_alphabet == as_int(construction["chunks"][index].get("alphabet_size"), f"construction chunk {index} alphabet"), f"candidate row {index} base alphabet mismatch")
        counts[str(base_alphabet)] += 1
        expected_kinds = ["base", "alphabet-upgrade", "tail"] if base_alphabet == 64 else ["base", "tail"]
        require(row.get("available_option_kinds") == expected_kinds, f"candidate row {index} available kinds mismatch")
        require(as_decimal(row.get("trigger_gap_db"), f"candidate row {index} trigger gap") > Decimal("0.1"), f"candidate row {index} did not satisfy strict trigger")
        base = row.get("base")
        require(isinstance(base, dict), f"candidate row {index} lacks base")
        require(as_int(base.get("alphabet_size"), f"candidate row {index} base alphabet") == base_alphabet, f"candidate row {index} base object alphabet mismatch")
        require(as_int(base.get("container_bytes"), f"candidate row {index} base bytes", minimum=8) >= 8, f"candidate row {index} base container too short")
        require_sha256(base.get("container_sha256"), f"candidate row {index} base container hash")
        base_energy = as_decimal(base.get("raw_source_energy"), f"candidate row {index} base energy", positive=True)
        base_sse = as_decimal(base.get("raw_sse"), f"candidate row {index} base SSE", nonnegative=True)
        upgrade = row.get("upgrade")
        if base_alphabet == 64:
            require(isinstance(upgrade, dict), f"A64 candidate row {index} omits A128 upgrade")
            require(
                upgrade.get("kind") == "alphabet-upgrade"
                and upgrade.get("independent_decode_passed") is True
                and as_int(upgrade.get("from_alphabet_size"), "upgrade from") == 64
                and as_int(upgrade.get("to_alphabet_size"), "upgrade to") == 128,
                f"candidate row {index} has invalid A128 upgrade",
            )
            require_sha256(upgrade.get("container_sha256"), f"candidate row {index} upgrade hash")
            as_int(upgrade.get("container_bytes"), f"candidate row {index} upgrade bytes", minimum=8)
            upgrade_energy = as_decimal(upgrade.get("raw_source_energy"), f"candidate row {index} upgrade energy", positive=True)
            close_decimal(upgrade_energy, base_energy, f"candidate row {index} upgrade/base source energy", absolute=Decimal("2e-12"), relative=Decimal("2e-14"))
            as_decimal(upgrade.get("raw_sse"), f"candidate row {index} upgrade SSE", nonnegative=True)
        else:
            require(upgrade is None, f"A128 candidate row {index} illegally contains an upgrade")
            require(base.get("independent_clean_decode_passed") is True, f"A128 candidate row {index} lacks independent base decode")
        tails = row.get("tails")
        require(isinstance(tails, list) and [tail.get("escape_count") for tail in tails] == tail_prefixes, f"candidate row {index} tail schedule incomplete")
        for tail in tails:
            k = as_int(tail.get("escape_count"), f"candidate row {index} tail k", minimum=1)
            require(
                tail.get("payload_unchanged") is True
                and tail.get("independent_physical_reparse_passed") is True
                and tail.get("parsed_tail_applied_for_scoring") is True
                and tail.get("raw_gain_identity_passed") is True,
                f"candidate row {index} tail k={k} lacks physical/score evidence",
            )
            require_sha256(tail.get("container_sha256"), f"candidate row {index} tail k={k} hash")
            expected_delta = (34 * k + 7) // 8
            require(as_int(tail.get("container_bytes"), "tail bytes") == as_int(base.get("container_bytes"), "base bytes") + expected_delta, f"candidate row {index} tail k={k} physical delta mismatch")
            require(as_int(tail.get("incremental_tail_bytes"), "incremental tail bytes") == expected_delta, f"candidate row {index} tail k={k} incremental byte identity fails")
            require(as_int(tail.get("meaningful_tail_bits"), "meaningful tail bits") == 34 * k, f"candidate row {index} tail k={k} meaningful bit identity fails")
            require(as_int(tail.get("tail_padding_bits"), "tail padding bits") == (-34 * k) % 8, f"candidate row {index} tail k={k} padding identity fails")
            tail_energy = as_decimal(tail.get("raw_source_energy"), f"candidate row {index} tail energy", positive=True)
            close_decimal(tail_energy, base_energy, f"candidate row {index} tail/base source energy", absolute=Decimal("2e-12"), relative=Decimal("2e-14"))
            tail_sse = as_decimal(tail.get("raw_sse"), f"candidate row {index} tail SSE", nonnegative=True)
            close_decimal(tail.get("raw_sse_reduction"), base_sse - tail_sse, f"candidate row {index} tail k={k} raw SSE reduction", absolute=Decimal("2e-12"), relative=Decimal("2e-12"))
            close_decimal(tail.get("raw_relative_mse"), tail_sse / tail_energy, f"candidate row {index} tail k={k} raw relative MSE")
            close_decimal(tail.get("actual_container_bpw"), Decimal(8 * as_int(tail.get("container_bytes"), "tail bytes")) / Decimal(BLOCK_VALUES), f"candidate row {index} tail k={k} container bpw")
    require(receipt.get("triggered_base_alphabet_counts") == counts, "candidate triggered alphabet census mismatch")
    return receipt


@dataclass(frozen=True)
class Option:
    option_id: str
    kind: str
    alphabet_size: int
    container_bytes: int
    container_sha256: str
    escape_count: int
    base_raw_sse: Decimal
    raw_sse: Decimal
    savings: Decimal


@dataclass(frozen=True)
class State:
    byte_delta: int
    savings: Decimal
    choice_ids: tuple[str, ...]
    alphabet_mask: int


def parse_selection_options(
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[int], list[list[Option]], dict[int, list[Option]]]:
    dp = selection.get("dp")
    considered = selection.get("options_considered")
    require(isinstance(dp, dict) and isinstance(considered, dict), "selection omits DP/options evidence")
    triggered = dp.get("triggered_chunks")
    require(triggered == candidate.get("triggered_chunk_indices"), "selection/candidate triggered sets differ")
    require(isinstance(triggered, list) and triggered == sorted(set(triggered)), "selection triggered order is noncanonical")
    require(set(considered) == {str(index) for index in triggered}, "selection options do not exactly cover triggered chunks")
    candidate_rows = {as_int(row["chunk_index"], "candidate index"): row for row in candidate["rows"]}
    groups: list[list[Option]] = []
    by_index: dict[int, list[Option]] = {}
    for index in triggered:
        rows = considered[str(index)]
        require(isinstance(rows, list) and rows, f"selection chunk {index} has no options")
        options: list[Option] = []
        seen: set[str] = set()
        for row in rows:
            require(isinstance(row, dict), f"selection chunk {index} option is not an object")
            option_id = str(row.get("option_id", ""))
            kind = str(row.get("kind", ""))
            require(option_id and option_id not in seen, f"selection chunk {index} option IDs are empty/duplicated")
            seen.add(option_id)
            alphabet = as_int(row.get("alphabet_size"), f"selection chunk {index} option alphabet")
            require(alphabet in (64, 128), f"selection chunk {index} option alphabet outside fixed-route domain")
            base_sse = as_decimal(row.get("base_raw_sse_decimal"), f"selection chunk {index} base SSE", nonnegative=True)
            raw_sse = as_decimal(row.get("raw_sse_decimal"), f"selection chunk {index} option SSE", nonnegative=True)
            saving = as_decimal(row.get("raw_sse_savings_decimal"), f"selection chunk {index} saving")
            require(saving == base_sse - raw_sse, f"selection chunk {index} option saving identity fails")
            container_bytes = as_int(row.get("container_bytes"), f"selection chunk {index} option bytes", minimum=8)
            option = Option(
                option_id=option_id,
                kind=kind,
                alphabet_size=alphabet,
                container_bytes=container_bytes,
                container_sha256=require_sha256(row.get("container_sha256"), f"selection chunk {index} option hash"),
                escape_count=as_int(row.get("escape_count"), f"selection chunk {index} escape count", minimum=0),
                base_raw_sse=base_sse,
                raw_sse=raw_sse,
                savings=saving,
            )
            options.append(option)
        bases = [option for option in options if option.kind == "base" and option.option_id == "base"]
        require(len(bases) == 1 and bases[0].savings == 0, f"selection chunk {index} requires one zero-saving base")
        base = bases[0]
        for row, option in zip(rows, options, strict=True):
            require(as_int(row.get("byte_delta_from_base"), "option byte delta") == option.container_bytes - base.container_bytes, f"selection chunk {index} byte delta identity fails")

        # Candidate-v3 is the completeness witness for the exposed option set.
        candidate_row = candidate_rows[index]
        expected: dict[str, tuple[int, str, Decimal]] = {
            "base": (
                as_int(candidate_row["base"]["container_bytes"], "candidate base bytes"),
                require_sha256(candidate_row["base"]["container_sha256"], "candidate base hash"),
                as_decimal(candidate_row["base"]["raw_sse"], "candidate base SSE"),
            )
        }
        if candidate_row.get("upgrade") is not None:
            upgrade = candidate_row["upgrade"]
            expected["upgrade-a128"] = (
                as_int(upgrade["container_bytes"], "candidate upgrade bytes"),
                require_sha256(upgrade["container_sha256"], "candidate upgrade hash"),
                as_decimal(upgrade["raw_sse"], "candidate upgrade SSE"),
            )
        for tail in candidate_row["tails"]:
            k = as_int(tail["escape_count"], "candidate tail k")
            expected[f"tail-k{k}"] = (
                as_int(tail["container_bytes"], "candidate tail bytes"),
                require_sha256(tail["container_sha256"], "candidate tail hash"),
                as_decimal(tail["raw_sse"], "candidate tail SSE"),
            )
        require(set(expected) == {option.option_id for option in options}, f"selection chunk {index} omitted/invented candidate options")
        for option in options:
            expected_bytes, expected_hash, expected_sse = expected[option.option_id]
            require(option.container_bytes == expected_bytes and option.container_sha256 == expected_hash, f"selection chunk {index} option {option.option_id} candidate binding mismatch")
            close_decimal(option.raw_sse, expected_sse, f"selection chunk {index} option {option.option_id} candidate SSE", absolute=Decimal("5e-14"), relative=Decimal("5e-14"))
        by_index[index] = options
        groups.append(options)
    return triggered, groups, by_index


def pareto_dp(groups: Iterable[list[Option]]) -> list[State]:
    frontier = [State(0, Decimal(0), (), 0)]
    for group_index, group in enumerate(groups):
        require(bool(group), "empty DP option group")
        base_rows = [option for option in group if option.kind == "base"]
        require(len(base_rows) == 1, "DP option group does not have one base")
        base_bytes = base_rows[0].container_bytes
        best_at_delta: dict[int, State] = {}
        for state in frontier:
            for option in group:
                candidate = State(
                    state.byte_delta + option.container_bytes - base_bytes,
                    state.savings + option.savings,
                    state.choice_ids + (option.option_id,),
                    state.alphabet_mask | ((1 << group_index) if option.alphabet_size == 128 else 0),
                )
                incumbent = best_at_delta.get(candidate.byte_delta)
                if (
                    incumbent is None
                    or candidate.savings > incumbent.savings
                    or (candidate.savings == incumbent.savings and candidate.choice_ids < incumbent.choice_ids)
                ):
                    best_at_delta[candidate.byte_delta] = candidate
        pruned: list[State] = []
        best_saving: Decimal | None = None
        for delta in sorted(best_at_delta):
            state = best_at_delta[delta]
            if best_saving is None or state.savings > best_saving:
                pruned.append(state)
                best_saving = state.savings
        frontier = pruned
    return frontier


def objective_log(raw_sse: Decimal, physical_bytes: int, panel_values: int) -> Decimal:
    require(raw_sse > 0 and physical_bytes > 0 and panel_values > 0, "invalid objective inputs")
    with localcontext() as context:
        context.prec = 80
        return raw_sse.ln() + Decimal(2).ln() * Decimal(16 * physical_bytes) / Decimal(panel_values)


def choose_best_state(
    frontier: list[State],
    base_raw_sse: Decimal,
    base_container_bytes: int,
    physical_prelude_bytes: int,
    max_bpw: Decimal,
) -> tuple[State, Decimal, int]:
    feasible: list[tuple[Decimal, int, tuple[str, ...], State]] = []
    for state in frontier:
        selected_sse = base_raw_sse - state.savings
        physical_bytes = base_container_bytes + state.byte_delta + physical_prelude_bytes
        if selected_sse <= 0 or physical_bytes <= 0:
            continue
        rate = Decimal(8 * physical_bytes) / Decimal(PANEL_VALUES)
        if rate >= max_bpw:
            continue
        feasible.append((objective_log(selected_sse, physical_bytes, PANEL_VALUES), physical_bytes, state.choice_ids, state))
    require(bool(feasible), "no exposed DP state satisfies the strict rate budget")
    score, physical_bytes, _, state = min(feasible, key=lambda item: (item[0], item[1], item[2]))
    return state, score, physical_bytes


def validate_side_receipt(path: Path, side_path: Path, bundle: ParsedBundle) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "side receipt must be an object")
    require(receipt.get("format") == "continuous reverse-waterfilled PLTE side receipt v1", "unsupported side receipt")
    require(receipt.get("status") == "exact round-trip passed", "side pack did not pass exact round trip")
    require(as_int(receipt.get("side_bytes"), "side bytes") == len(bundle.literal_side) == side_path.stat().st_size, "side byte count mismatch")
    require(receipt.get("side_sha256") == sha256_path(side_path) == sha256_bytes(bundle.literal_side), "side hash does not match physical bundle")
    require(side_path.read_bytes() == bundle.literal_side, "published literal side is not bundle-reconstructed literal side")
    require(as_int(receipt.get("header_bytes"), "side header bytes") == SIDE_HEADER.size, "side header byte count mismatch")
    require(as_int(receipt.get("exp2_lut_bytes"), "side LUT bytes") == 64 * 8, "side LUT byte count mismatch")
    require(as_int(receipt.get("block_scale_bytes"), "side scale bytes") == CHUNK_COUNT * 4, "side scale byte count mismatch")
    require(as_int(receipt.get("packed_label_bytes"), "side label bytes") == (LABEL_COUNT * LABEL_BITS + 7) // 8, "side label byte count mismatch")
    require(as_int(receipt.get("profile_bytes"), "side profile bytes") == CHUNK_COUNT * PROFILE.size, "side profile byte count mismatch")
    require(receipt.get("stable_membership_reconstructed_from_side_only") is True and receipt.get("exact_eof") is True and receipt.get("profile_binary64_roundtrip") is True, "side receipt lacks canonical reconstruction/EOF evidence")
    close_decimal(receipt.get("side_bpw_over_panel"), Decimal(8 * len(bundle.literal_side)) / Decimal(PANEL_VALUES), "raw side diagnostic bpw")
    return receipt


def validate_bundle_receipt(path: Path, bundle_path: Path, bundle: ParsedBundle) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "bundle receipt must be an object")
    require(receipt.get("format") == "continuous PLTE WFOUTR fixed-route bundle experiment v2", "unsupported bundle receipt format")
    require(receipt.get("status") == "passed" and receipt.get("experimental_not_v1") is True, "bundle receipt did not pass fixed-route v2")
    require(receipt.get("source_free_reparse_passed") is True, "producer bundle reparse did not pass")
    physical_bytes = bundle_path.stat().st_size
    require(as_int(receipt.get("bundle_bytes"), "bundle receipt bytes") == physical_bytes, "bundle receipt physical bytes mismatch")
    require(receipt.get("bundle_sha256") == sha256_path(bundle_path), "bundle receipt physical hash mismatch")
    require(as_int(receipt.get("panel_values"), "bundle panel values") == PANEL_VALUES, "bundle panel values mismatch")
    close_decimal(receipt.get("physical_all_in_bpw"), Decimal(8 * physical_bytes) / Decimal(PANEL_VALUES), "bundle all-in bpw")
    require(as_int(receipt.get("header_bytes"), "bundle header bytes") == OUTER_HEADER.size, "bundle header size mismatch")
    require(receipt.get("header_sha256") == sha256_bytes(bundle.header), "bundle header hash mismatch")
    require(as_int(receipt.get("physical_prelude_bytes"), "bundle prelude bytes") == bundle.physical_prelude_bytes, "bundle prelude bytes mismatch")
    side = receipt.get("side")
    require(isinstance(side, dict), "bundle receipt omits side section")
    require(as_int(side.get("codec_id"), "bundle side codec") == SIDE_CODEC_FIXED_ROUTE, "bundle receipt side codec mismatch")
    require(as_int(side.get("raw_bytes"), "bundle side raw bytes") == len(bundle.literal_side), "bundle side raw length mismatch")
    require(side.get("raw_sha256") == sha256_bytes(bundle.literal_side), "bundle side raw hash mismatch")
    require(as_int(side.get("canonical_raw_bytes"), "canonical raw-side bytes") == len(bundle.canonical_side), "canonical side length mismatch")
    require(side.get("canonical_raw_sha256") == sha256_bytes(bundle.canonical_side), "canonical side hash mismatch")
    require(as_int(side.get("canonical_xz_bytes"), "canonical XZ bytes") == len(bundle.canonical_xz), "canonical XZ length mismatch")
    require(side.get("canonical_xz_sha256") == sha256_bytes(bundle.canonical_xz), "canonical XZ hash mismatch")
    require(as_int(side.get("route_bits"), "route bits") == ROUTE_BITS and as_int(side.get("route_bytes"), "route bytes") == ROUTE_BYTES, "fixed route size mismatch")
    require(side.get("route_sha256") == sha256_bytes(bundle.route), "route hash mismatch")
    require(as_int(side.get("compressed_bytes"), "side payload bytes") == len(bundle.canonical_xz) + ROUTE_BYTES, "side payload length mismatch")
    require(side.get("compressed_sha256") == sha256_bytes(bundle.canonical_xz + bundle.route), "side payload hash mismatch")
    require(side.get("literal_side_reconstructed_hash_verified") is True and side.get("canonical_profile_offsets_verified") is True, "bundle side receipt lacks reconstruction evidence")
    require(side.get("alphabet_domain") == [64, 128], "bundle side alphabet domain mismatch")
    mask = receipt.get("mask")
    require(isinstance(mask, dict), "bundle receipt omits mask")
    require(as_int(mask.get("raw_bytes"), "mask raw bytes") == len(bundle.raw_mask), "bundle mask raw bytes mismatch")
    require(mask.get("raw_sha256") == sha256_bytes(bundle.raw_mask), "bundle mask raw hash mismatch")
    require(as_int(mask.get("compressed_bytes"), "mask compressed bytes") == len(bundle.compressed_mask), "bundle mask compressed bytes mismatch")
    require(mask.get("compressed_sha256") == sha256_bytes(bundle.compressed_mask), "bundle mask compressed hash mismatch")
    containers = receipt.get("containers")
    require(isinstance(containers, dict), "bundle receipt omits containers")
    require(as_int(containers.get("count"), "bundle container count") == CHUNK_COUNT, "bundle container count mismatch")
    require(as_int(containers.get("bytes"), "bundle container bytes") == bundle.container_bytes, "bundle container bytes mismatch")
    require(containers.get("ordered_sha256") == [frame.sha256 for frame in bundle.frames], "bundle ordered container hashes mismatch")
    require(containers.get("all_arithmetic_padding_zero") is True and containers.get("all_sparse_tail_padding_zero") is True and containers.get("exact_eof") is True, "bundle container receipt lacks zero-padding/EOF evidence")

    bindings = receipt.get("dependency_bindings")
    expected = {
        "fixed_route_packer_wrapper_sha256": PINNED["fixed_route_bundle_packer"],
        "fixed_route_codec_sha256": PINNED["fixed_route_codec"],
        "audited_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
        "pinned_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
    }
    require(bindings == expected, "bundle dependency bindings differ from audited implementations")
    return receipt


def validate_selection_receipt(
    root: Path,
    path: Path,
    construction_path: Path,
    selected_path: Path,
    base_path: Path,
    candidate_path: Path,
    side_path: Path,
    side_receipt_path: Path,
    bundle_path: Path,
    bundle_receipt_path: Path,
    construction: dict[str, Any],
    candidate: dict[str, Any],
    bundle: ParsedBundle,
) -> tuple[dict[str, Any], State]:
    selection = load_json(path)
    require(isinstance(selection, dict), "selection receipt must be an object")
    require(selection.get("format") == "continuous PLTE exact adaptive selection receipt fixed-route v2", "selection is not fixed-route v2")
    require(selection.get("status") == "passed", "selection did not pass")
    require(selection.get("strict_ptq") is True, "selection is not strict PTQ")
    require(selection.get("write_once_output") is True, "selection output was not declared write-once")
    require(selection.get("physical_selection_mode") == "fixed-route", "selection did not use fixed-route physical objective")
    require(
        selection.get("selection_guarantee")
        == "global exact: side codec 3 has one fixed 50-byte route and an alphabet-invariant canonical all-A64 XZ member",
        "selection does not claim the audited fixed-route global guarantee",
    )
    require(
        selection.get("objective")
        == "(base_raw_sse - savings) * 2**(16 * WFOUTR01_bundle_bytes / panel_values)",
        "selection objective wording/formula changed",
    )
    arithmetic = selection.get("selection_arithmetic")
    require(isinstance(arithmetic, dict), "selection omits arithmetic contract")
    require(arithmetic.get("byte_axis") == "exact integer physical bytes", "selection byte axis is not exact")
    require(arithmetic.get("sse_axis") == "exact Decimal values parsed from serialized candidate JSON", "selection SSE axis is not Decimal")
    require(arithmetic.get("transcendental_comparison") == "Python Decimal ln at 80-digit precision", "selection transcendental precision changed")
    require(arithmetic.get("tie_break") == "objective, then total bytes, then lexicographic option IDs", "selection tie break changed")
    require("fixed 50-byte route400" in str(arithmetic.get("side_physical_model")), "selection side model is not fixed route")

    inputs = selection.get("inputs")
    require(isinstance(inputs, dict), "selection omits input bindings")
    require(inputs.get("manifest_sha256") == sha256_path(construction_path), "selection construction-manifest hash mismatch")
    require(inputs.get("base_run_receipt_sha256") == sha256_path(base_path), "selection base-receipt hash mismatch")
    require(inputs.get("candidate_receipt_sha256") == sha256_path(candidate_path), "selection candidate-receipt hash mismatch")
    require(inputs.get("packer_sha256") == PINNED["side_packer"], "selection used an unpinned side packer")
    require(inputs.get("bundle_packer_sha256") == PINNED["fixed_route_bundle_packer"], "selection used an unpinned fixed-route bundle packer")
    require(inputs.get("raw_mask_sha256") == PINNED["raw_mask"], "selection used a non-normative raw mask")
    require(as_int(inputs.get("raw_mask_bytes"), "selection raw-mask bytes") == MASK_RAW_BYTES, "selection raw-mask length mismatch")
    require(as_int(selection.get("validation", {}).get("canonical_chunks"), "selection canonical chunks") == CHUNK_COUNT, "selection validation is not 400 chunks")
    validation = selection["validation"]
    for flag in (
        "one_staged_option_per_chunk",
        "all_base_reports_and_containers_hash_validated",
        "all_adaptive_reports_decodes_and_container_hashes_validated",
        "tail_payload_identity_validated",
        "raw_side_length_invariant_after_alphabet_regeneration",
        "selected_raw_side_matches_reranked_signature_byte_for_byte",
        "selected_physical_side_payload_matches_rerank_hash_and_length",
        "selected_bz2_mask_matches_rerank_hash_and_length",
        "wfoutr01_source_free_reparse",
    ):
        require(validation.get(flag) is True, f"selection validation flag {flag} is not true")

    artifacts = selection.get("artifacts")
    require(isinstance(artifacts, dict), "selection omits artifacts")
    artifact_expected = {
        "selected_manifest": (selected_path, artifacts.get("selected_manifest_sha256")),
        "literal_side": (side_path, artifacts.get("literal_side_sha256")),
        "side_receipt": (side_receipt_path, artifacts.get("side_receipt_sha256")),
        "physical_bundle": (bundle_path, artifacts.get("physical_bundle_sha256")),
        "bundle_receipt": (bundle_receipt_path, artifacts.get("bundle_receipt_sha256")),
    }
    for key, (actual_path, expected_hash) in artifact_expected.items():
        declared = artifacts.get(key)
        require(isinstance(declared, str) and Path(declared).name == actual_path.name, f"selection artifact {key} filename mismatch")
        require(require_sha256(expected_hash, f"selection artifact {key} hash") == sha256_path(actual_path), f"selection artifact {key} hash mismatch")
    require(as_int(artifacts.get("literal_side_raw_bytes"), "selection literal-side bytes") == side_path.stat().st_size, "selection literal-side byte count mismatch")
    require(as_int(artifacts.get("physical_bundle_bytes"), "selection bundle bytes") == bundle_path.stat().st_size, "selection bundle byte count mismatch")
    require(artifacts.get("container_directory") == "containers", "selection container directory changed")

    mapping = selection.get("selection_map")
    require(isinstance(mapping, list) and len(mapping) == CHUNK_COUNT, "selection map must contain exactly 400 rows")
    require([as_int(row.get("chunk_index"), "selection map index") for row in mapping] == list(range(CHUNK_COUNT)), "selection map is not canonical 0..399")
    triggered, groups, options_by_index = parse_selection_options(selection, candidate)
    selected_option_ids = dict(zip(triggered, selection["dp"].get("selected_choice_ids", []), strict=True))
    selected_container_bytes = 0
    for index, (row, frame) in enumerate(zip(mapping, bundle.frames, strict=True)):
        require(as_int(row.get("base_alphabet_size"), f"selection map {index} base alphabet") == as_int(construction["chunks"][index].get("alphabet_size"), f"construction alphabet {index}"), f"selection map {index} base alphabet mismatch")
        selected_alphabet = as_int(row.get("selected_alphabet_size"), f"selection map {index} selected alphabet")
        require(selected_alphabet == (64 if bundle.alphabet_codes[index] == 0 else 128), f"selection map {index} alphabet differs from physical route")
        require(as_int(row.get("container_bytes"), f"selection map {index} bytes") == len(frame.literal), f"selection map {index} bytes differ from bundle")
        require(row.get("container_sha256") == frame.sha256, f"selection map {index} hash differs from bundle")
        require(as_int(row.get("escape_count"), f"selection map {index} escapes") == frame.escape_count, f"selection map {index} escape count differs from bundle")
        staged_path = safe_child(root, row.get("staged_container"), f"staged container {index}")
        require(staged_path.stat().st_size == len(frame.literal) and sha256_path(staged_path) == frame.sha256, f"staged container {index} differs from bundle frame")
        require(staged_path.read_bytes() == frame.literal, f"staged container {index} is not byte-identical to bundle frame")
        is_triggered = index in options_by_index
        require(row.get("triggered") is is_triggered, f"selection map {index} trigger flag mismatch")
        if is_triggered:
            option_id = selected_option_ids[index]
            matches = [option for option in options_by_index[index] if option.option_id == option_id]
            require(len(matches) == 1, f"selected option for chunk {index} is not unique")
            option = matches[0]
            require(
                row.get("option_id") == option.option_id
                and row.get("kind") == option.kind
                and selected_alphabet == option.alphabet_size
                and len(frame.literal) == option.container_bytes
                and frame.sha256 == option.container_sha256,
                f"selection map {index} does not materialize selected DP option",
            )
            close_decimal(row.get("selected_raw_sse_decimal_if_triggered"), option.raw_sse, f"selection map {index} selected SSE", absolute=Decimal(0), relative=Decimal(0))
            close_decimal(row.get("raw_sse_savings_decimal"), option.savings, f"selection map {index} saving", absolute=Decimal(0), relative=Decimal(0))
        else:
            require(row.get("option_id") == "base" and row.get("kind") == "base", f"untriggered chunk {index} is not base")
            require(row.get("selected_raw_sse_decimal_if_triggered") is None, f"untriggered chunk {index} carries selected SSE")
            require(as_decimal(row.get("raw_sse_savings_decimal"), f"untriggered chunk {index} saving") == 0, f"untriggered chunk {index} has nonzero saving")
        selected_container_bytes += len(frame.literal)

    dp = selection["dp"]
    require(as_int(dp.get("triggered_count"), "DP triggered count") == len(triggered), "DP triggered count mismatch")
    require(as_int(dp.get("option_count"), "DP option count") == sum(len(group) for group in groups), "DP option count mismatch")
    require(as_int(dp.get("fixed_route_bits"), "DP fixed-route bits") == ROUTE_BITS, "DP fixed-route bit count mismatch")
    frontier = pareto_dp(groups)
    require(as_int(dp.get("proxy_frontier_states"), "DP proxy frontier states") == len(frontier), "DP proxy frontier state count mismatch")
    require(as_int(dp.get("physical_frontier_states"), "DP physical frontier states") == len(frontier), "DP physical frontier state count mismatch")
    require(as_int(dp.get("peak_exact_signature_states"), "DP peak states") == len(frontier), "DP fixed-route peak-state count mismatch")
    require(as_int(dp.get("distinct_physical_side_signatures"), "DP side signatures") == len({state.alphabet_mask for state in frontier}), "DP side-signature count mismatch")
    frontier_rows = [
        {
            "alphabet_signature_hex": hex(state.alphabet_mask),
            "byte_delta": state.byte_delta,
            "savings_decimal": str(+state.savings),
            "choice_ids": list(state.choice_ids),
        }
        for state in frontier
    ]
    require(dp.get("frontier_sha256") == canonical_sha256(frontier_rows), "DP frontier hash does not reproduce")
    accounting = selection.get("accounting")
    require(isinstance(accounting, dict), "selection omits accounting")
    require(as_int(accounting.get("panel_values"), "accounting panel values") == PANEL_VALUES, "selection panel values mismatch")
    base_raw_sse = as_decimal(inputs.get("base_total_raw_sse_decimal"), "selection base total SSE", positive=True)
    max_bpw = as_decimal(accounting.get("strict_max_bpw_decimal"), "selection strict max bpw", positive=True)
    require(max_bpw <= RATE_LIMIT_BPW, "selection rate budget exceeds 2.5 bpw")
    base_container_bytes = as_int(accounting.get("base_container_bytes"), "selection base container bytes", minimum=1)
    require(base_container_bytes == as_int(load_json(base_path).get("actual_container_bytes"), "base receipt container bytes"), "selection/base total container bytes differ")
    require(as_int(accounting.get("wfoutr01_header_bytes"), "accounting header bytes") == OUTER_HEADER.size, "selection header accounting mismatch")
    require(as_int(accounting.get("side_physical_payload_bytes"), "accounting side payload bytes") == len(bundle.canonical_xz) + ROUTE_BYTES, "selection side payload accounting mismatch")
    require(as_int(accounting.get("side_canonical_xz_bytes"), "accounting canonical XZ bytes") == len(bundle.canonical_xz), "selection canonical XZ accounting mismatch")
    require(as_int(accounting.get("side_fixed_route_bytes"), "accounting route bytes") == ROUTE_BYTES, "selection route accounting mismatch")
    require(as_int(accounting.get("mask_bz2_compressed_bytes"), "accounting compressed-mask bytes") == len(bundle.compressed_mask), "selection mask accounting mismatch")
    require(as_int(accounting.get("physical_prelude_bytes"), "accounting prelude bytes") == bundle.physical_prelude_bytes, "selection prelude accounting mismatch")
    selected_state, selected_log, modeled_bytes = choose_best_state(frontier, base_raw_sse, base_container_bytes, bundle.physical_prelude_bytes, max_bpw)
    require(list(selected_state.choice_ids) == dp.get("selected_choice_ids"), "published DP choices are not the recomputed optimum")
    require(selected_state.byte_delta == as_int(dp.get("selected_byte_delta"), "selected DP byte delta"), "selected DP byte delta mismatch")
    require(selected_state.savings == as_decimal(dp.get("selected_raw_sse_savings_decimal"), "selected DP saving"), "selected DP saving mismatch")
    require(hex(selected_state.alphabet_mask) == dp.get("selected_alphabet_signature_hex"), "selected DP alphabet signature mismatch")
    require(modeled_bytes == bundle_path.stat().st_size, "DP physical optimum bytes differ from bundle bytes")
    require(selected_container_bytes == bundle.container_bytes == as_int(accounting.get("selected_container_bytes"), "selected container bytes"), "selected container accounting mismatch")
    require(selected_container_bytes == base_container_bytes + selected_state.byte_delta, "selected container bytes do not match DP delta")
    require(as_int(accounting.get("physical_bundle_bytes"), "accounting bundle bytes") == bundle_path.stat().st_size, "selection physical bundle bytes mismatch")
    actual_bpw = Decimal(8 * bundle_path.stat().st_size) / Decimal(PANEL_VALUES)
    close_decimal(accounting.get("physical_all_in_bpw_decimal"), actual_bpw, "selection physical all-in bpw", absolute=Decimal(0), relative=Decimal(0))
    require(accounting.get("strict_rate_pass") is True and actual_bpw < max_bpw <= RATE_LIMIT_BPW, "selection strict rate test fails")
    selected_sse = base_raw_sse - selected_state.savings
    require(selected_sse == as_decimal(accounting.get("selected_total_raw_sse_decimal"), "selection total SSE"), "selection total SSE identity fails")
    require(selected_log == objective_log(selected_sse, bundle_path.stat().st_size, PANEL_VALUES), "selection objective log mismatch")
    with localcontext() as context:
        context.prec = 80
        objective = selected_log.exp()
    close_decimal(accounting.get("objective_decimal"), objective, "selection objective", absolute=Decimal("1e-20"), relative=Decimal("2e-27"))
    return selection, selected_state


def validate_decode_receipt(
    path: Path,
    bundle_path: Path,
    bundle: ParsedBundle,
) -> dict[str, Any]:
    receipt = load_json(path)
    require(isinstance(receipt, dict), "decode receipt must be an object")
    require(
        receipt.get("format")
        == "continuous reverse-waterfilled PLTE independent outer decode fixed-route experiment v2",
        "unsupported independent decode receipt",
    )
    require(receipt.get("status") == "passed" and receipt.get("experimental_not_v1") is True, "independent decode did not pass fixed-route v2")
    require(receipt.get("strict_ptq") is True, "independent decode is not marked strict PTQ")
    independence = receipt.get("independence")
    expected_independence = {
        "read_encoder_json": False,
        "read_exploratory_manifest": False,
        "read_normalized_or_raw_source": False,
        "encoder_probability_arrays_used": False,
        "membership_derived_from_literal_side_only": True,
    }
    require(independence == expected_independence, "independent decoder source/encoder isolation contract changed")
    geometry = receipt.get("geometry")
    expected_geometry = {
        "canonical_blocks": CHUNK_COUNT,
        "groups_per_canonical_block": GROUPS_PER_CHUNK,
        "group_values": GROUP_VALUES,
        "polar_chunks": CHUNK_COUNT,
        "polar_block_values": BLOCK_VALUES,
        "panel_values": PANEL_VALUES,
    }
    require(geometry == expected_geometry, "decode receipt geometry mismatch")
    side = receipt.get("side")
    require(isinstance(side, dict), "decode receipt omits side")
    require(as_int(side.get("bytes"), "decode side bytes") == len(bundle.literal_side), "decode side byte count mismatch")
    require(side.get("sha256") == sha256_bytes(bundle.literal_side), "decode side hash mismatch")
    require(side.get("exact_eof") is True and as_int(side.get("packed_label_count"), "decode label count") == LABEL_COUNT, "decode side EOF/label count mismatch")
    stream = receipt.get("encoded_stream")
    require(isinstance(stream, dict), "decode receipt omits encoded stream")
    require(stream.get("self_contained_decoder_side_and_mask") is True, "decode did not use a self-contained bundle")
    require(as_int(stream.get("physical_prelude_bytes"), "decode physical prelude") == bundle.physical_prelude_bytes, "decode prelude byte count mismatch")
    require(as_int(stream.get("container_stream_bytes"), "decode container bytes") == bundle.container_bytes, "decode container byte count mismatch")
    require(as_int(stream.get("combined_encoded_bytes"), "decode encoded bytes") == bundle_path.stat().st_size, "decode physical bundle bytes mismatch")
    require(stream.get("combined_encoded_sha256") == sha256_path(bundle_path), "decode physical bundle hash mismatch")
    require(stream.get("exact_eof_after_declared_chunk_count") is True and stream.get("all_arithmetic_padding_zero") is True and stream.get("all_sparse_tail_padding_zero") is True, "decode receipt lacks exact EOF/zero padding")
    close_decimal(stream.get("actual_all_in_bpw"), Decimal(8 * bundle_path.stat().st_size) / Decimal(PANEL_VALUES), "decode all-in bpw")
    assets = receipt.get("decoder_assets")
    require(isinstance(assets, dict), "decode receipt omits decoder assets")
    require(as_int(assets.get("raw_mask_bytes"), "decode raw-mask bytes") == MASK_RAW_BYTES, "decode raw-mask bytes mismatch")
    require(assets.get("raw_mask_sha256") == PINNED["raw_mask"] == sha256_bytes(bundle.raw_mask), "decode raw-mask hash mismatch")
    require(assets.get("raw_mask_embedded_and_physically_charged") is True, "decode receipt does not charge embedded mask")
    require(assets.get("clean_decoder_sha256") == PINNED["clean_decoder"], "decode used non-normative clean decoder")
    require(assets.get("outer_decoder_sha256") == PINNED["v1_outer_decoder"], "delegated outer decoder hash mismatch")
    dependencies = {
        "fixed_route_decoder_wrapper_sha256": PINNED["fixed_route_outer_decoder"],
        "fixed_route_codec_sha256": PINNED["fixed_route_codec"],
        "audited_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
        "pinned_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
    }
    require(receipt.get("dependency_bindings") == dependencies, "decode dependency bindings mismatch")
    require(all(assets.get(key) == value for key, value in dependencies.items()), "decode assets do not repeat dependency bindings")
    outer = receipt.get("outer_bundle")
    require(isinstance(outer, dict), "decode receipt omits fixed-route outer bundle")
    require(as_int(outer.get("side_codec_id"), "decode side codec") == SIDE_CODEC_FIXED_ROUTE, "decode side codec mismatch")
    require(as_int(outer.get("canonical_side_raw_bytes"), "decode canonical side bytes") == len(bundle.canonical_side), "decode canonical side bytes mismatch")
    require(outer.get("canonical_side_raw_sha256") == sha256_bytes(bundle.canonical_side), "decode canonical side hash mismatch")
    require(as_int(outer.get("canonical_side_xz_bytes"), "decode canonical XZ bytes") == len(bundle.canonical_xz), "decode canonical XZ bytes mismatch")
    require(outer.get("canonical_side_xz_sha256") == sha256_bytes(bundle.canonical_xz), "decode canonical XZ hash mismatch")
    require(as_int(outer.get("route_bits"), "decode route bits") == ROUTE_BITS and as_int(outer.get("route_bytes"), "decode route bytes") == ROUTE_BYTES, "decode route length mismatch")
    require(outer.get("route_sha256") == sha256_bytes(bundle.route), "decode route hash mismatch")
    require(outer.get("reconstructed_literal_side_sha256") == sha256_bytes(bundle.literal_side), "decode reconstructed-side hash mismatch")
    require(outer.get("reconstructed_literal_side_hash_verified") is True and outer.get("canonical_profile_offsets_verified") is True and outer.get("decompressor_exact_eof") is True, "decode fixed-route reconstruction evidence missing")
    require(outer.get("alphabet_domain") == [64, 128], "decode alphabet domain mismatch")
    expected_census = {
        "64": sum(code == 0 for code in bundle.alphabet_codes),
        "128": sum(code == 1 for code in bundle.alphabet_codes),
    }
    # The base decoder may include a zero-count A256 key; reject nonzero values.
    census = receipt.get("alphabet_census")
    require(isinstance(census, dict), "decode receipt omits alphabet census")
    require(as_int(census.get("64", 0), "decode A64 census") == expected_census["64"], "decode A64 census mismatch")
    require(as_int(census.get("128", 0), "decode A128 census") == expected_census["128"], "decode A128 census mismatch")
    require(all(key in ("64", "128", "256") and as_int(value, "decode alphabet census") == (expected_census.get(key, 0)) for key, value in census.items()), "decode alphabet census contains unsupported/nonzero entries")
    chunks = receipt.get("chunks")
    require(isinstance(chunks, list) and len(chunks) == CHUNK_COUNT, "decode receipt must contain 400 chunk receipts")
    require([as_int(row.get("chunk_index"), "decode chunk index") for row in chunks] == list(range(CHUNK_COUNT)), "decode chunk receipts are noncanonical")
    for index, (row, frame) in enumerate(zip(chunks, bundle.frames, strict=True)):
        require(as_int(row.get("container_bytes"), f"decode chunk {index} bytes") == len(frame.literal), f"decode chunk {index} byte count mismatch")
        require(row.get("container_sha256") == frame.sha256, f"decode chunk {index} hash mismatch")
        require(as_int(row.get("logical_bits"), f"decode chunk {index} logical bits") == frame.logical_bits, f"decode chunk {index} logical length mismatch")
        require(as_int(row.get("escape_count"), f"decode chunk {index} escapes") == frame.escape_count, f"decode chunk {index} escape count mismatch")
        require(as_int(row.get("arithmetic_padding_bits"), f"decode chunk {index} arithmetic padding") == frame.arithmetic_padding_bits and row.get("arithmetic_padding_zero") is True, f"decode chunk {index} arithmetic padding evidence mismatch")
        require(as_int(row.get("tail_padding_bits"), f"decode chunk {index} tail padding") == frame.tail_padding_bits and row.get("tail_padding_zero") is True, f"decode chunk {index} tail padding evidence mismatch")
        require(float(as_decimal(row.get("fp32_decoder_scale"), f"decode chunk {index} scale")) == frame.scale, f"decode chunk {index} FP32 scale mismatch")
        require(as_int(row.get("alphabet_size"), f"decode chunk {index} alphabet") == (64 if bundle.alphabet_codes[index] == 0 else 128), f"decode chunk {index} alphabet mismatch")
        require(as_int(row.get("arithmetic_decoder_bits_read"), f"decode chunk {index} bits read") == frame.logical_bits, f"decode chunk {index} did not consume all arithmetic bits")
        require_sha256(row.get("frequency_u16_sha256"), f"decode chunk {index} frequency hash")
        require_sha256(row.get("reconstruction_indices_i16_sha256"), f"decode chunk {index} reconstruction-index hash")
    reconstruction = receipt.get("reconstruction")
    require(isinstance(reconstruction, dict), "decode receipt omits reconstruction binding")
    dtype = str(reconstruction.get("dtype"))
    require(dtype in ("<f4", "<f8"), "decode reconstruction dtype unsupported")
    element_bytes = 4 if dtype == "<f4" else 8
    require(reconstruction.get("shape") == [CHUNK_COUNT, BLOCK_VALUES], "decode reconstruction shape mismatch")
    require(as_int(reconstruction.get("bytes"), "decode reconstruction bytes") == PANEL_VALUES * element_bytes, "decode reconstruction byte count mismatch")
    require_sha256(reconstruction.get("sha256"), "decode reconstruction hash")
    require(reconstruction.get("every_canonical_group_written_exactly_once") is True, "decode scatter coverage did not pass")
    return receipt


def ordered_source_identity(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for ordinal, row in enumerate(rows):
        source_id = str(row["id"]).encode("utf-8")
        digest.update(ordinal.to_bytes(8, "little"))
        digest.update(len(source_id).to_bytes(4, "little"))
        digest.update(source_id)
        digest.update(bytes.fromhex(require_sha256(row["sha256"], f"ledger source {ordinal} hash")))
        for field in ("tensor", "role", "layer"):
            payload = json.dumps(row[field], sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest.update(len(payload).to_bytes(4, "little"))
            digest.update(payload)
    return digest.hexdigest()


def aggregate_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    output = []
    for key in sorted(buckets, key=lambda item: tuple("<global>" if value is None else str(value) for value in item)):
        members = buckets[key]
        energy = 0.0
        sse = 0.0
        for row in members:
            energy += float(as_decimal(row["source_energy"], "block energy"))
            sse += float(as_decimal(row["sse"], "block SSE"))
        output.append(
            {
                "key": key,
                "blocks": len(members),
                "values": len(members) * BLOCK_VALUES,
                "source_energy": energy,
                "sse": sse,
                "ids": [str(row["id"]) for row in members],
            }
        )
    return output


def validate_strata(
    published: object,
    block_rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    actual_bpw: Decimal,
    gaussian_reference: Decimal,
    label: str,
) -> None:
    require(isinstance(published, list), f"{label} must be a list")
    expected = aggregate_rows(block_rows, fields)
    require(len(published) == len(expected), f"{label} row count mismatch")
    for position, (row, want) in enumerate(zip(published, expected, strict=True)):
        require(isinstance(row, dict), f"{label} row {position} is not an object")
        require(tuple(row.get(field) for field in fields) == want["key"], f"{label} row {position} key/order mismatch")
        require(as_int(row.get("blocks"), f"{label} blocks") == want["blocks"], f"{label} block census mismatch")
        require(as_int(row.get("values"), f"{label} values") == want["values"], f"{label} value census mismatch")
        close_decimal(row.get("source_energy"), Decimal(str(want["source_energy"])), f"{label} source energy")
        close_decimal(row.get("sse"), Decimal(str(want["sse"])), f"{label} SSE")
        relative = Decimal(str(want["sse"])) / Decimal(str(want["source_energy"]))
        close_decimal(row.get("relative_mse"), relative, f"{label} relative MSE")
        close_decimal(row.get("effective_charged_panel_bpw"), actual_bpw, f"{label} charged bpw")
        close_decimal(row.get("gaussian_reference_mse_at_effective_panel_rate"), gaussian_reference, f"{label} Gaussian reference")
        with localcontext() as context:
            context.prec = 50
            gap = Decimal(10) * (relative / gaussian_reference).ln() / Decimal(10).ln()
        close_decimal(row.get("diagnostic_gap_db_at_effective_panel_rate"), gap, f"{label} gap")
        require(row.get("ids") == want["ids"], f"{label} ID coverage/order mismatch")


def validate_evaluation(
    path: Path,
    ledger_path: Path,
    decode_path: Path,
    bundle_path: Path,
    ledger: dict[str, Any],
    decode: dict[str, Any],
    selection: dict[str, Any],
    bundle: ParsedBundle,
) -> tuple[dict[str, Any], Decimal, Decimal]:
    result = load_json(path)
    require(isinstance(result, dict), "evaluation must be an object")
    require(
        result.get("format")
        == "continuous reverse-waterfilled PLTE exact-source evaluation fixed-route experiment v2",
        "unsupported exact-source evaluation format",
    )
    require(result.get("status") == "passed" and result.get("experimental_not_v1") is True, "exact-source evaluation did not pass fixed-route v2")
    require(result.get("strict_ptq") is True, "exact-source evaluation is not strict PTQ")
    require(result.get("source_is_evaluator_only") is True, "source was not restricted to evaluator-only use")
    require(result.get("source_ledger_sha256") == sha256_path(ledger_path), "evaluation source-ledger hash mismatch")
    require(result.get("ordered_source_identity_sha256") == ordered_source_identity(ledger["blocks"]), "evaluation ordered source identity hash mismatch")
    require(as_int(result.get("source_blocks"), "evaluation source blocks") == CHUNK_COUNT, "evaluation is not 400 source blocks")
    require(as_int(result.get("panel_values"), "evaluation panel values") == PANEL_VALUES, "evaluation panel values mismatch")
    require(result.get("all_source_hashes_and_ordinals_verified") is True and result.get("all_400_source_hashes_ordinals_and_scatter_coverage_verified") is True, "evaluation lacks complete source/scatter verification")
    require(result.get("decode_receipt_sha256") == sha256_path(decode_path), "evaluation decode-receipt hash mismatch")
    require(result.get("reconstruction_sha256") == decode["reconstruction"]["sha256"], "evaluation/decode reconstruction hashes differ")
    require(result.get("encoded_sha256") == sha256_path(bundle_path), "evaluation bundle hash mismatch")
    require(result.get("loaded_outer_decoder_sha256") == PINNED["fixed_route_outer_decoder"], "evaluation loaded an unpinned fixed-route decoder")
    require(result.get("normative_clean_decoder_sha256") == PINNED["clean_decoder"], "evaluation clean decoder hash mismatch")
    require(result.get("normative_raw_mask_sha256") == PINNED["raw_mask"], "evaluation raw-mask hash mismatch")
    require(as_int(result.get("encoded_bytes"), "evaluation encoded bytes") == bundle_path.stat().st_size, "evaluation encoded byte count mismatch")

    dependencies = {
        "fixed_route_evaluator_wrapper_sha256": PINNED["fixed_route_evaluator"],
        "delegated_v1_evaluator_sha256": PINNED["v1_evaluator"],
        "pinned_v1_evaluator_sha256": PINNED["v1_evaluator"],
        "fixed_route_decoder_wrapper_sha256": PINNED["fixed_route_outer_decoder"],
        "fixed_route_codec_sha256": PINNED["fixed_route_codec"],
        "audited_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
        "pinned_v1_outer_decoder_sha256": PINNED["v1_outer_decoder"],
    }
    require(result.get("evaluation_dependency_bindings") == dependencies, "evaluation dependency bindings mismatch")
    fixed_route = result.get("fixed_route_codec")
    require(
        isinstance(fixed_route, dict)
        and as_int(fixed_route.get("side_codec_id"), "evaluation side codec") == SIDE_CODEC_FIXED_ROUTE
        and as_int(fixed_route.get("route_bits"), "evaluation route bits") == ROUTE_BITS
        and fixed_route.get("literal_side_hash_verified_before_source_scoring") is True,
        "evaluation lacks fixed-route pre-score verification",
    )

    physical_bytes = bundle_path.stat().st_size
    actual_bpw = Decimal(8 * physical_bytes) / Decimal(PANEL_VALUES)
    close_decimal(result.get("actual_all_in_bpw"), actual_bpw, "evaluation actual all-in bpw")
    require(as_decimal(result.get("rate_limit_bpw"), "evaluation rate limit") == RATE_LIMIT_BPW, "evaluation rate limit is not 2.5 bpw")
    require(result.get("strict_rate_below_2p5_bpw_passed") is True and actual_bpw < RATE_LIMIT_BPW, "evaluation strict rate claim fails")

    blocks = result.get("blocks")
    require(isinstance(blocks, list) and len(blocks) == CHUNK_COUNT, "evaluation must expose 400 block rows")
    require([as_int(row.get("canonical_block_ordinal"), "evaluation block ordinal") for row in blocks] == list(range(CHUNK_COUNT)), "evaluation block rows are noncanonical")
    total_energy_float = 0.0
    total_sse_float = 0.0
    for ordinal, (row, ledger_row) in enumerate(zip(blocks, ledger["blocks"], strict=True)):
        require(
            row.get("id") == ledger_row.get("id")
            and row.get("tensor") == ledger_row.get("tensor")
            and row.get("role") == ledger_row.get("role")
            and row.get("layer") == ledger_row.get("layer")
            and row.get("source_sha256") == ledger_row.get("sha256"),
            f"evaluation block {ordinal} identity differs from source ledger",
        )
        energy = float(as_decimal(row.get("source_energy"), f"block {ordinal} energy", positive=True))
        sse = float(as_decimal(row.get("sse"), f"block {ordinal} SSE", nonnegative=True))
        total_energy_float += energy
        total_sse_float += sse
        relative = Decimal(str(sse)) / Decimal(str(energy))
        close_decimal(row.get("relative_mse"), relative, f"block {ordinal} relative MSE")
        close_decimal(row.get("effective_charged_panel_bpw"), actual_bpw, f"block {ordinal} charged bpw")

    # Match the evaluator's canonical sequential binary64 accumulation.
    close_decimal(result.get("source_energy"), Decimal(str(total_energy_float)), "evaluation aggregate source energy")
    close_decimal(result.get("sse"), Decimal(str(total_sse_float)), "evaluation aggregate SSE")
    source_energy = as_decimal(result.get("source_energy"), "evaluation source energy", positive=True)
    sse = as_decimal(result.get("sse"), "evaluation SSE", nonnegative=True)
    relative_mse = sse / source_energy
    close_decimal(result.get("relative_mse"), relative_mse, "evaluation aggregate relative MSE")
    with localcontext() as context:
        context.prec = 80
        gaussian_reference = (-(Decimal(2) * actual_bpw) * Decimal(2).ln()).exp()
        gap = Decimal(10) * (relative_mse / gaussian_reference).ln() / Decimal(10).ln()
    close_decimal(result.get("gaussian_reference_mse_at_actual_rate"), gaussian_reference, "evaluation Gaussian reference")
    close_decimal(result.get("gaussian_reference_gap_db"), gap, "evaluation signed Gaussian gap")
    require(as_decimal(result.get("target_gap_db"), "evaluation target gap") == TARGET_GAP_DB, "evaluation target is not signed <= -0.10 dB")
    require(result.get("target_gap_le_negative_0p10_db_passed") is True, "evaluation target-pass flag is false")
    require(gap <= TARGET_GAP_DB, f"signed Gaussian gap {gap} dB does not meet <= -0.10 dB")

    validate_strata(result.get("by_role"), blocks, ("role",), actual_bpw, gaussian_reference, "role strata")
    validate_strata(result.get("by_layer"), blocks, ("layer",), actual_bpw, gaussian_reference, "layer strata")
    validate_strata(result.get("by_role_and_layer"), blocks, ("role", "layer"), actual_bpw, gaussian_reference, "role/layer strata")

    diagnostics = result.get("mixed_chunk_diagnostics")
    require(isinstance(diagnostics, list) and len(diagnostics) == CHUNK_COUNT, "evaluation mixed-chunk diagnostics must contain 400 rows")
    require([as_int(row.get("chunk_index"), "mixed chunk index") for row in diagnostics] == list(range(CHUNK_COUNT)), "mixed-chunk diagnostics are noncanonical")
    chunk_energy = 0.0
    chunk_sse = 0.0
    shared_outer_bpw = Decimal(8 * bundle.physical_prelude_bytes) / Decimal(PANEL_VALUES)
    for index, (row, frame) in enumerate(zip(diagnostics, bundle.frames, strict=True)):
        energy = float(as_decimal(row.get("source_energy"), f"mixed chunk {index} energy", positive=True))
        row_sse = float(as_decimal(row.get("sse"), f"mixed chunk {index} SSE", nonnegative=True))
        chunk_energy += energy
        chunk_sse += row_sse
        require(as_int(row.get("container_bytes"), f"mixed chunk {index} bytes") == len(frame.literal), f"mixed chunk {index} container bytes mismatch")
        container_bpw = Decimal(8 * len(frame.literal)) / Decimal(BLOCK_VALUES)
        close_decimal(row.get("container_bpw"), container_bpw, f"mixed chunk {index} container bpw")
        close_decimal(row.get("allocated_shared_outer_bpw"), shared_outer_bpw, f"mixed chunk {index} shared outer bpw")
        effective = container_bpw + shared_outer_bpw
        close_decimal(row.get("effective_charged_chunk_bpw"), effective, f"mixed chunk {index} effective bpw")
        require(row.get("diagnostic_only") is True, f"mixed chunk {index} is not marked diagnostic-only")
    close_decimal(Decimal(str(chunk_energy)), source_energy, "mixed-chunk aggregate energy")
    close_decimal(Decimal(str(chunk_sse)), sse, "mixed-chunk aggregate SSE")

    selection_accounting = selection["accounting"]
    close_decimal(selection_accounting.get("selected_total_raw_sse_decimal"), sse, "selection/evaluation total SSE", absolute=Decimal("2e-10"), relative=Decimal("2e-12"))
    if selection_accounting.get("raw_relative_mse_decimal") is not None:
        close_decimal(selection_accounting.get("raw_relative_mse_decimal"), relative_mse, "selection/evaluation relative MSE", absolute=Decimal("2e-12"), relative=Decimal("2e-12"))
    if selection_accounting.get("gaussian_reference_gap_db_decimal") is not None:
        close_decimal(selection_accounting.get("gaussian_reference_gap_db_decimal"), gap, "selection/evaluation Gaussian gap", absolute=Decimal("2e-10"), relative=Decimal("2e-10"))
    return result, actual_bpw, gap


def validate_no_raw_sources(
    root: Path,
    ledger: dict[str, Any],
    construction: dict[str, Any],
) -> int:
    source_hashes = {
        require_sha256(row.get("sha256"), "ledger source hash")
        for row in ledger["blocks"]
    }
    normalized_hashes = {
        require_sha256(row.get("normalized_source_sha256"), "normalized source hash")
        for row in construction["chunks"]
    }
    source_basenames = {Path(str(row["path"])).name.casefold() for row in ledger["blocks"]}
    forbidden_components = {"sources", "raw_sources", "normalized_sources", "model_weights", "checkpoint"}
    forbidden_suffixes = (
        ".bf16.bin",
        ".safetensors",
        ".ckpt",
        ".pth",
        ".pt",
        ".npy",
        ".npz",
        ".raw",
        ".bf16",
    )
    files = 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"publication contains symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        files += 1
        relative = path.relative_to(root)
        lowered_parts = {part.casefold() for part in relative.parts[:-1]}
        require(not lowered_parts.intersection(forbidden_components), f"publication contains raw-source directory component: {relative}")
        name = path.name.casefold()
        require(name not in source_basenames, f"publication contains a ledger source basename: {relative}")
        require(not any(name.endswith(suffix) for suffix in forbidden_suffixes), f"publication contains forbidden raw/model array extension: {relative}")
        digest = sha256_path(path)
        require(digest not in source_hashes, f"publication contains exact raw BF16 source bytes: {relative}")
        require(digest not in normalized_hashes, f"publication contains exact normalized encoder source bytes: {relative}")
        # Every normative raw or normalized block is exactly 2*2^18 bytes.
        # A file of that size is not part of the declared source-free layout.
        require(path.stat().st_size != BLOCK_VALUES * 2, f"publication contains undeclared block-sized payload: {relative}")
    return files


def validate_selection_checksum(path: Path, selection_path: Path) -> None:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"cannot read selection checksum: {error}") from error
    expected = f"{sha256_path(selection_path)}  selection.receipt.json\n"
    require(text == expected, "selection.receipt.sha256 is not the exact canonical checksum line")


def validate_optional_reconstruction(root: Path, decode: dict[str, Any]) -> str:
    dtype = decode["reconstruction"]["dtype"]
    filename = "reconstruction.f32" if dtype == "<f4" else "reconstruction.f64"
    path = root / filename
    if not path.exists():
        return "not materialized; receipt hash is cross-bound but reconstruction bytes were not rehashed"
    require(path.is_file() and not path.is_symlink(), "published reconstruction must be a regular file")
    require(path.stat().st_size == as_int(decode["reconstruction"]["bytes"], "reconstruction bytes"), "published reconstruction byte count mismatch")
    require(sha256_path(path) == decode["reconstruction"]["sha256"], "published reconstruction SHA-256 mismatch")
    return "materialized bytes and SHA-256 verified"


def publication_paths(root: Path) -> dict[str, Path]:
    paths = {key: root / filename for key, filename in REQUIRED_FILENAMES.items()}
    for key, path in paths.items():
        require(path.is_file(), f"required publication artifact missing: {path.name}")
        require(not path.is_symlink(), f"required publication artifact is a symlink: {path.name}")
    containers = root / "containers"
    require(containers.is_dir() and not containers.is_symlink(), "required containers/ directory missing or symlinked")
    return paths


def verify_publication(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), f"publication directory does not exist: {root}")
    paths = publication_paths(root)
    validate_selection_checksum(paths["selection_checksum"], paths["selection_receipt"])

    ledger = validate_source_ledger(paths["source_ledger"])
    require(ledger["construction_manifest_sha256"] == sha256_path(paths["construction_manifest"]), "source ledger does not bind published construction manifest")
    bundle = parse_bundle(paths["bundle"])
    construction, _selected = validate_manifest_pair(
        paths["construction_manifest"],
        paths["selected_manifest"],
        ledger,
        bundle,
    )
    base = validate_base_receipt(paths["base_receipt"], paths["construction_manifest"])
    candidate = validate_candidate_receipt(
        paths["candidate_receipt"],
        paths["construction_manifest"],
        paths["base_receipt"],
        construction,
    )
    validate_side_receipt(paths["side_receipt"], paths["side"], bundle)
    validate_bundle_receipt(paths["bundle_receipt"], paths["bundle"], bundle)
    selection, selected_state = validate_selection_receipt(
        root,
        paths["selection_receipt"],
        paths["construction_manifest"],
        paths["selected_manifest"],
        paths["base_receipt"],
        paths["candidate_receipt"],
        paths["side"],
        paths["side_receipt"],
        paths["bundle"],
        paths["bundle_receipt"],
        construction,
        candidate,
        bundle,
    )
    decode = validate_decode_receipt(paths["decode_receipt"], paths["bundle"], bundle)
    _evaluation, actual_bpw, gap = validate_evaluation(
        paths["evaluation"],
        paths["source_ledger"],
        paths["decode_receipt"],
        paths["bundle"],
        ledger,
        decode,
        selection,
        bundle,
    )
    reconstruction_status = validate_optional_reconstruction(root, decode)
    file_count = validate_no_raw_sources(root, ledger, construction)
    require(as_int(base["actual_container_bytes"], "base total container bytes") > 0, "base receipt has no physical bytes")
    return {
        "status": "passed",
        "verifier_contract": "VORPAL continuous 400-block source-free fixed-route publication v1",
        "publication_directory": str(root),
        "physical_bundle_bytes": paths["bundle"].stat().st_size,
        "physical_all_in_bpw_decimal": str(actual_bpw),
        "signed_gaussian_gap_db_decimal": str(gap),
        "target_signed_gap_db_lte": str(TARGET_GAP_DB),
        "strict_rate_bpw_lt": str(RATE_LIMIT_BPW),
        "canonical_blocks": CHUNK_COUNT,
        "canonical_roles": len(LAYERWISE_ROLES) + len(GLOBAL_ROLE_COUNTS),
        "layerwise_roles_with_exact_0_through_47_coverage": len(LAYERWISE_ROLES),
        "triggered_chunks": len(candidate["triggered_chunk_indices"]),
        "selected_choice_count": len(selected_state.choice_ids),
        "source_free_files_scanned": file_count,
        "raw_or_normalized_sources_found": 0,
        "reconstruction_artifact": reconstruction_status,
        "receipt_sha256": {
            "candidate": sha256_path(paths["candidate_receipt"]),
            "selection": sha256_path(paths["selection_receipt"]),
            "bundle": sha256_path(paths["bundle_receipt"]),
            "decode": sha256_path(paths["decode_receipt"]),
            "evaluation": sha256_path(paths["evaluation"]),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication_dir", type=Path, help="directory containing the canonical source-free publication layout")
    parser.add_argument("--compact", action="store_true", help="emit one-line JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_publication(args.publication_dir)
    except (VerificationError, KeyError, TypeError, ValueError, OSError, struct.error) as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(failure, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2), file=sys.stderr)
        return 1
    print(json.dumps(report, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
