#!/usr/bin/env python3
"""Independent decoder for the continuous-waterfilled PLTE outer stream.

The decoder consumes only the literal ``WFPLTE01`` side stream, the raw
concatenation of self-delimiting PLTE containers, the pinned raw six-level
polar mask, and the existing clean PLTE decoder implementation.  It never
reads an encoder report, encoder probabilities, normalized sources, or the
exploratory membership manifest.

Wire format v1 deliberately fixes sigma_source=3, frozen_seed=20260831, and
trial=0 as codec constants.  They are not runtime side information.
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import hashlib
import importlib.util
import json
import lzma
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


MAGIC = b"WFPLTE01"
VERSION = 1
HEADER = struct.Struct("<8sIIIIIIdI")
PROFILE = struct.Struct("<ddB")
LABEL_BITS = 6
BASE_MASK_LEVELS = 6
NORMATIVE_BLOCK_LENGTH = 1 << 18
NORMATIVE_GROUP_VALUES = 1 << 11
NORMATIVE_GROUPS_PER_CHUNK = NORMATIVE_BLOCK_LENGTH // NORMATIVE_GROUP_VALUES
NORMATIVE_SIGMA_SOURCE = 3.0
NORMATIVE_FROZEN_SEED = 20260831
NORMATIVE_TRIAL = 0
ALPHABET_BY_CODE = {0: 64, 1: 128, 2: 256}
OUTPUT_DTYPES = {"f64": np.dtype("<f8"), "f32": np.dtype("<f4")}
OUTER_MAGIC = b"WFOUTR01"
OUTER_VERSION = 1
SIDE_CODEC_LZMA_XZ = 1
MASK_CODEC_BZ2 = 2
# magic, version, header_bytes, side codec/raw/compressed bytes, mask
# codec/raw/compressed bytes, then SHA256(raw side, compressed side, raw mask,
# compressed mask).  The 168-byte header itself is part of the charged rate.
OUTER_HEADER = struct.Struct("<8s8I32s32s32s32s")
MAX_SIDE_RAW_BYTES = 1 << 20
MAX_SIDE_COMPRESSED_BYTES = 1 << 20
MAX_MASK_COMPRESSED_BYTES = 1 << 20


@dataclass(frozen=True)
class Profile:
    distortion: float
    eta: float
    alphabet_size: int
    literal_bytes_hex: str


@dataclass(frozen=True)
class SideInfo:
    blob: bytes
    block_count: int
    groups_per_chunk: int
    group_values: int
    chunk_count: int
    label_count: int
    lambda_variance: float
    lut: np.ndarray
    block_scales: np.ndarray
    labels: np.ndarray
    profiles: tuple[Profile, ...]
    qscales: np.ndarray
    stable_order: np.ndarray
    adjacent_qvariance_ties: int
    tie_boundaries_split_by_chunks: int


@dataclass(frozen=True)
class ContainerFrame:
    chunk_index: int
    literal: bytes
    logical_bits: int
    scale: float
    payload: bytes
    escape_positions: np.ndarray
    escape_values_u16: np.ndarray
    arithmetic_padding_bits: int
    tail_padding_bits: int
    sha256: str


@dataclass(frozen=True)
class BundlePrelude:
    header: bytes
    side: SideInfo
    compressed_side: bytes
    raw_mask: bytes
    compressed_mask: bytes
    side_codec: int
    mask_codec: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_decoder(path: Path):
    spec = importlib.util.spec_from_file_location("outer_clean_plte_decoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_exact(handle: BinaryIO, count: int, description: str) -> bytes:
    if count < 0:
        raise ValueError(f"negative byte count for {description}")
    payload = handle.read(count)
    if len(payload) != count:
        raise ValueError(
            f"truncated {description}: expected {count} bytes, found {len(payload)}"
        )
    return payload


def side_size_from_header(header: bytes) -> int:
    if len(header) != HEADER.size:
        raise ValueError(f"side header must contain exactly {HEADER.size} bytes")
    (
        magic,
        version,
        block_count,
        groups_per_chunk,
        group_values,
        chunk_count,
        label_count,
        _lambda_variance,
        lut_count,
    ) = HEADER.unpack(header)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"unsupported side magic/version: {magic!r}/{version}")
    counts = (block_count, groups_per_chunk, group_values, chunk_count, label_count)
    if any(value <= 0 for value in counts):
        raise ValueError(f"side counts must all be positive: {counts}")
    if lut_count != 64:
        raise ValueError(f"v1 requires exactly 64 serialized LUT entries, got {lut_count}")
    if block_count > 1_000_000 or label_count > 100_000_000 or chunk_count > 1_000_000:
        raise ValueError("side counts exceed defensive v1 limits")
    label_bytes = (label_count * LABEL_BITS + 7) // 8
    return (
        HEADER.size
        + lut_count * 8
        + block_count * 4
        + label_bytes
        + chunk_count * PROFILE.size
    )


def unpack_labels(payload: bytes, count: int) -> np.ndarray:
    expected = (count * LABEL_BITS + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"label payload is {len(payload)} bytes, expected {expected}")
    values = np.empty(count, dtype=np.int8)
    bit_offset = 0
    for index in range(count):
        byte_index = bit_offset >> 3
        shift = bit_offset & 7
        word = payload[byte_index]
        if byte_index + 1 < len(payload):
            word |= payload[byte_index + 1] << 8
        value = (word >> shift) & 0x3F
        values[index] = value - 64 if value & 0x20 else value
        bit_offset += LABEL_BITS
    padding = 8 * expected - count * LABEL_BITS
    if padding and payload[-1] & (((1 << padding) - 1) << (8 - padding)):
        # Six-bit labels are packed least-significant-bit first, so unused bits
        # occupy the high end of the final byte.
        raise ValueError("nonzero six-bit label padding")
    return values


def parse_side(blob: bytes) -> SideInfo:
    if len(blob) < HEADER.size:
        raise ValueError("side stream is shorter than its fixed header")
    expected_size = side_size_from_header(blob[: HEADER.size])
    if len(blob) != expected_size:
        raise ValueError(f"side stream expected {expected_size} bytes, found {len(blob)}")
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
    ) = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or version != VERSION:
        raise AssertionError("side header changed after size validation")
    if groups_per_chunk != NORMATIVE_GROUPS_PER_CHUNK:
        raise ValueError(
            f"v1 requires {NORMATIVE_GROUPS_PER_CHUNK} groups/chunk, got {groups_per_chunk}"
        )
    if group_values != NORMATIVE_GROUP_VALUES:
        raise ValueError(
            f"v1 requires {NORMATIVE_GROUP_VALUES} values/group, got {group_values}"
        )
    if label_count != block_count * groups_per_chunk:
        raise ValueError("label count does not equal block_count * groups_per_chunk")
    if chunk_count * groups_per_chunk != label_count:
        raise ValueError("v1 requires full fixed-length polar chunks")
    if not math.isfinite(lambda_variance) or lambda_variance <= 0.0:
        raise ValueError(f"invalid serialized water level {lambda_variance}")

    offset = HEADER.size
    lut = np.frombuffer(blob, dtype="<f8", count=lut_count, offset=offset).copy()
    offset += lut_count * 8
    block_scales = np.frombuffer(
        blob, dtype="<f4", count=block_count, offset=offset
    ).copy()
    offset += block_count * 4
    label_bytes = (label_count * LABEL_BITS + 7) // 8
    labels = unpack_labels(blob[offset : offset + label_bytes], label_count)
    offset += label_bytes
    profiles: list[Profile] = []
    for chunk_index in range(chunk_count):
        raw = blob[offset : offset + PROFILE.size]
        if len(raw) != PROFILE.size:
            raise ValueError(f"truncated binary64 profile {chunk_index}")
        distortion, eta, code = PROFILE.unpack(raw)
        offset += PROFILE.size
        if code not in ALPHABET_BY_CODE:
            raise ValueError(f"unsupported alphabet code {code} in chunk {chunk_index}")
        if not math.isfinite(distortion) or not 0.0 < distortion < NORMATIVE_SIGMA_SOURCE**2:
            raise ValueError(f"invalid test distortion {distortion} in chunk {chunk_index}")
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError(f"invalid eta {eta} in chunk {chunk_index}")
        profiles.append(
            Profile(distortion, eta, ALPHABET_BY_CODE[code], raw.hex())
        )
    if offset != len(blob):
        raise AssertionError("side parser did not consume exact EOF")
    if not np.all(np.isfinite(lut)) or np.any(lut <= 0.0):
        raise ValueError("serialized qscale LUT must be finite and positive")
    if not np.all(np.diff(lut) > 0.0):
        raise ValueError("serialized qscale LUT must be strictly increasing")
    if not np.all(np.isfinite(block_scales)) or np.any(block_scales <= 0.0):
        raise ValueError("serialized FP32 block scales must be finite and positive")

    canonical = np.arange(label_count, dtype=np.int64)
    block_ordinals = canonical // groups_per_chunk
    qscales = (
        block_scales[block_ordinals].astype(np.float64)
        * lut[labels.astype(np.int16) + 32]
    )
    qvariances = np.square(qscales)
    if not np.all(np.isfinite(qvariances)) or np.any(qvariances <= 0.0):
        raise ValueError("reconstructed qvariances must be finite and positive")
    stable_order = np.lexsort((canonical, qvariances)).astype(np.int64, copy=False)
    if np.unique(stable_order).size != label_count:
        raise AssertionError("stable membership order is not a permutation")
    sorted_qv = qvariances[stable_order]
    equal_adjacent = sorted_qv[1:] == sorted_qv[:-1]
    adjacent_ties = int(np.count_nonzero(equal_adjacent))
    boundary_indices = np.arange(groups_per_chunk, label_count, groups_per_chunk)
    split_ties = int(
        np.count_nonzero(sorted_qv[boundary_indices - 1] == sorted_qv[boundary_indices])
    )
    return SideInfo(
        blob=blob,
        block_count=block_count,
        groups_per_chunk=groups_per_chunk,
        group_values=group_values,
        chunk_count=chunk_count,
        label_count=label_count,
        lambda_variance=lambda_variance,
        lut=lut,
        block_scales=block_scales,
        labels=labels,
        profiles=tuple(profiles),
        qscales=qscales,
        stable_order=stable_order,
        adjacent_qvariance_ties=adjacent_ties,
        tie_boundaries_split_by_chunks=split_ties,
    )


def read_side_file(path: Path) -> SideInfo:
    with path.open("rb") as handle:
        header = read_exact(handle, HEADER.size, "side header")
        expected = side_size_from_header(header)
        blob = header + read_exact(handle, expected - HEADER.size, "side body")
        if handle.read(1):
            raise ValueError("standalone side file has trailing bytes")
    return parse_side(blob)


def read_side_prefix(handle: BinaryIO) -> SideInfo:
    header = read_exact(handle, HEADER.size, "bundle side header")
    expected = side_size_from_header(header)
    blob = header + read_exact(handle, expected - HEADER.size, "bundle side body")
    return parse_side(blob)


def decompress_lzma_xz_exact(payload: bytes, expected_bytes: int) -> bytes:
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    result = decoder.decompress(payload, max_length=expected_bytes + 1)
    if len(result) != expected_bytes:
        raise ValueError(
            f"LZMA side expanded to {len(result)} bytes, expected {expected_bytes}"
        )
    if not decoder.eof or decoder.unused_data:
        raise ValueError("LZMA side did not consume one exact XZ stream")
    return result


def decompress_bz2_exact(payload: bytes, expected_bytes: int) -> bytes:
    decoder = bz2.BZ2Decompressor()
    result = decoder.decompress(payload, max_length=expected_bytes + 1)
    if len(result) != expected_bytes:
        raise ValueError(
            f"BZ2 mask expanded to {len(result)} bytes, expected {expected_bytes}"
        )
    if not decoder.eof or decoder.unused_data:
        raise ValueError("BZ2 mask did not consume one exact stream")
    return result


def read_bundle_prelude(handle: BinaryIO) -> BundlePrelude:
    header = read_exact(handle, OUTER_HEADER.size, "outer bundle header")
    (
        magic,
        version,
        header_bytes,
        side_codec,
        side_raw_bytes,
        side_compressed_bytes,
        mask_codec,
        mask_raw_bytes,
        mask_compressed_bytes,
        side_raw_hash,
        side_compressed_hash,
        mask_raw_hash,
        mask_compressed_hash,
    ) = OUTER_HEADER.unpack(header)
    if (magic, version, header_bytes) != (
        OUTER_MAGIC,
        OUTER_VERSION,
        OUTER_HEADER.size,
    ):
        raise ValueError(
            f"unsupported outer header {magic!r}/{version}/{header_bytes}"
        )
    if side_codec != SIDE_CODEC_LZMA_XZ or mask_codec != MASK_CODEC_BZ2:
        raise ValueError(
            f"unsupported side/mask codecs {side_codec}/{mask_codec}"
        )
    if not HEADER.size <= side_raw_bytes <= MAX_SIDE_RAW_BYTES:
        raise ValueError("invalid or excessive raw side length in outer header")
    if not 0 < side_compressed_bytes <= MAX_SIDE_COMPRESSED_BYTES:
        raise ValueError("invalid side lengths in outer header")
    expected_mask_bytes = BASE_MASK_LEVELS * (
        (NORMATIVE_BLOCK_LENGTH + 7) // 8
    )
    if (
        mask_raw_bytes != expected_mask_bytes
        or not 0 < mask_compressed_bytes <= MAX_MASK_COMPRESSED_BYTES
    ):
        raise ValueError(
            f"invalid mask lengths {mask_raw_bytes}/{mask_compressed_bytes}; "
            f"expected raw {expected_mask_bytes}"
        )
    compressed_side = read_exact(
        handle, side_compressed_bytes, "LZMA-compressed literal side"
    )
    compressed_mask = read_exact(
        handle, mask_compressed_bytes, "BZ2-compressed raw mask"
    )
    if hashlib.sha256(compressed_side).digest() != side_compressed_hash:
        raise ValueError("compressed side SHA256 mismatch")
    if hashlib.sha256(compressed_mask).digest() != mask_compressed_hash:
        raise ValueError("compressed mask SHA256 mismatch")
    raw_side = decompress_lzma_xz_exact(compressed_side, side_raw_bytes)
    raw_mask = decompress_bz2_exact(compressed_mask, mask_raw_bytes)
    if hashlib.sha256(raw_side).digest() != side_raw_hash:
        raise ValueError("raw side SHA256 mismatch")
    if hashlib.sha256(raw_mask).digest() != mask_raw_hash:
        raise ValueError("raw mask SHA256 mismatch")
    side = parse_side(raw_side)
    return BundlePrelude(
        header=header,
        side=side,
        compressed_side=compressed_side,
        raw_mask=raw_mask,
        compressed_mask=compressed_mask,
        side_codec=side_codec,
        mask_codec=mask_codec,
    )


def read_container_frame(
    handle: BinaryIO, chunk_index: int, block_length: int
) -> ContainerFrame:
    fixed = read_exact(handle, 8, f"chunk {chunk_index} PLTE header")
    header_word, scale = struct.unpack("<If", fixed)
    logical_bits = header_word & ((1 << 20) - 1)
    escape_count = header_word >> 20
    payload_bytes = (logical_bits + 7) // 8
    tail_bytes = (34 * escape_count + 7) // 8
    payload = read_exact(handle, payload_bytes, f"chunk {chunk_index} arithmetic payload")
    tail = read_exact(handle, tail_bytes, f"chunk {chunk_index} sparse tail")
    arithmetic_padding = payload_bytes * 8 - logical_bits
    if arithmetic_padding and payload[-1] & ((1 << arithmetic_padding) - 1):
        raise ValueError(f"chunk {chunk_index} has nonzero arithmetic padding")
    meaningful_tail_bits = 34 * escape_count
    tail_padding = tail_bytes * 8 - meaningful_tail_bits
    if tail_padding and tail[-1] & ((1 << tail_padding) - 1):
        raise ValueError(f"chunk {chunk_index} has nonzero sparse-tail padding")
    if escape_count > block_length:
        raise ValueError(f"chunk {chunk_index} escape count exceeds block length")
    combined = int.from_bytes(tail, "big")
    if tail_padding:
        combined >>= tail_padding
    positions = np.empty(escape_count, dtype=np.int32)
    values = np.empty(escape_count, dtype=np.uint16)
    record_mask = (1 << 34) - 1
    for index in range(escape_count - 1, -1, -1):
        record = combined & record_mask
        combined >>= 34
        positions[index] = record >> 16
        values[index] = record & 0xFFFF
    if combined:
        raise AssertionError(f"chunk {chunk_index} sparse-tail parser left bits")
    if escape_count:
        if int(positions[-1]) >= block_length:
            raise ValueError(f"chunk {chunk_index} sparse-tail position exceeds block length")
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError(f"chunk {chunk_index} sparse-tail positions are not increasing")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"chunk {chunk_index} has invalid FP32 decoder scale {scale}")
    literal = fixed + payload + tail
    return ContainerFrame(
        chunk_index=chunk_index,
        literal=literal,
        logical_bits=logical_bits,
        scale=float(scale),
        payload=payload,
        escape_positions=positions,
        escape_values_u16=values,
        arithmetic_padding_bits=arithmetic_padding,
        tail_padding_bits=tail_padding,
        sha256=sha256_bytes(literal),
    )


def read_all_frames(handle: BinaryIO, chunk_count: int) -> tuple[ContainerFrame, ...]:
    frames = tuple(
        read_container_frame(handle, chunk_index, NORMATIVE_BLOCK_LENGTH)
        for chunk_index in range(chunk_count)
    )
    if handle.read(1):
        raise ValueError(f"container stream has trailing bytes after {chunk_count} chunks")
    return frames


_WORKER: dict[str, object] = {}


def raw_frozen_flags_from_bytes(
    payload: bytes, n: int, levels: int
) -> list[np.ndarray]:
    stride = (n + 7) // 8
    if len(payload) != levels * stride:
        raise ValueError(
            f"raw mask bytes {len(payload)} != expected {levels * stride}"
        )
    return [
        np.unpackbits(
            np.frombuffer(
                payload[level * stride : (level + 1) * stride], dtype=np.uint8
            ),
            bitorder="big",
        )[:n].astype(np.uint8)
        for level in range(levels)
    ]


def initialize_worker(clean_decoder: str, raw_mask: bytes) -> None:
    decoder = load_clean_decoder(Path(clean_decoder))
    n = NORMATIVE_BLOCK_LENGTH
    _WORKER.clear()
    _WORKER.update(
        decoder=decoder,
        reverse=decoder.bit_reverse_indices(n),
        layers=decoder.sc_layers(n),
        base_flags=raw_frozen_flags_from_bytes(raw_mask, n, BASE_MASK_LEVELS),
        zero_flag=np.zeros(n, dtype=np.uint8),
    )


def decode_chunk(task: tuple[Profile, ContainerFrame]) -> dict[str, object]:
    started = time.perf_counter()
    profile, frame = task
    decoder = _WORKER["decoder"]
    reverse = _WORKER["reverse"]
    layers = _WORKER["layers"]
    base_flags = _WORKER["base_flags"]
    zero_flag = _WORKER["zero_flag"]
    alphabet_size = profile.alphabet_size
    levels = int(math.log2(alphabet_size))
    if (1 << levels) != alphabet_size or levels < BASE_MASK_LEVELS:
        raise ValueError(f"invalid lattice alphabet size {alphabet_size}")
    flags = list(base_flags) + [zero_flag] * (levels - BASE_MASK_LEVELS)
    sigma_recon = math.sqrt(NORMATIVE_SIGMA_SOURCE**2 - profile.distortion)
    alphabet = profile.eta * np.arange(
        -alphabet_size // 2 + 1,
        alphabet_size // 2 + 1,
        dtype=np.float64,
    )
    weights = np.exp(-0.5 * np.square(alphabet / sigma_recon))
    arithmetic = decoder.ArithmeticBinaryDecoder(frame.payload, frame.logical_bits)
    previous = np.zeros(NORMATIVE_BLOCK_LENGTH, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected_count = 0
    for level_index in range(levels):
        level = level_index + 1
        frozen_rng = np.random.default_rng(
            NORMATIVE_FROZEN_SEED
            + 104729 * NORMATIVE_TRIAL
            + 1000003 * level
        )
        frozen = frozen_rng.integers(
            0, 2, size=NORMATIVE_BLOCK_LENGTH, dtype=np.uint8
        )
        prior_lr = decoder.leaf_prior_ratios(weights, previous, level)
        decoded_x, frequencies = decoder.decode_sc_level(
            prior_lr,
            flags[level_index],
            frozen,
            reverse,
            layers,
            arithmetic,
        )
        previous += (1 << level_index) * decoded_x.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected_count += int(frequencies.size)
    if int(arithmetic.cursor) != frame.logical_bits:
        raise AssertionError(
            f"chunk {frame.chunk_index} arithmetic decoder consumed "
            f"{arithmetic.cursor} bits, expected {frame.logical_bits}"
        )
    reconstruction = alphabet[previous] * frame.scale
    if frame.escape_positions.size:
        escaped = (
            frame.escape_values_u16.astype(np.uint32) << np.uint32(16)
        ).view(np.float32)
        reconstruction[frame.escape_positions] = escaped.astype(np.float64)
    return {
        "chunk_index": frame.chunk_index,
        "reconstruction": reconstruction,
        "selected_symbols": selected_count,
        "frequency_u16_sha256": frequency_hash.hexdigest(),
        "reconstruction_indices_i16_sha256": sha256_bytes(
            previous.astype("<i2", copy=False).tobytes()
        ),
        "arithmetic_decoder_bits_read": int(arithmetic.cursor),
        "procedural_zero_mask_levels": levels - BASE_MASK_LEVELS,
        "decode_wall_seconds": time.perf_counter() - started,
    }


def decode_tasks(
    tasks: list[tuple[Profile, ContainerFrame]],
    clean_decoder: Path,
    raw_mask: bytes,
    workers: int,
) -> Iterator[dict[str, object]]:
    if workers == 1:
        initialize_worker(str(clean_decoder), raw_mask)
        for task in tasks:
            yield decode_chunk(task)
        return
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(str(clean_decoder), raw_mask),
    ) as pool:
        yield from pool.map(decode_chunk, tasks, chunksize=1)


def atomic_json(path: Path, value: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = Path(str(path) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    encoded = parser.add_mutually_exclusive_group(required=True)
    encoded.add_argument(
        "--bundle", type=Path, help="side bytes followed by raw PLTE containers"
    )
    encoded.add_argument("--side", type=Path, help="standalone literal side stream")
    parser.add_argument(
        "--containers",
        type=Path,
        help="raw self-delimiting PLTE stream; required with --side",
    )
    parser.add_argument(
        "--raw-mask",
        type=Path,
        help="development-only external mask; required with --side, forbidden with self-contained --bundle",
    )
    parser.add_argument("--clean-decoder", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reconstruction-dtype", choices=tuple(OUTPUT_DTYPES), default="f64")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.side is not None and args.containers is None:
        parser.error("--containers is required with --side")
    if args.side is not None and args.raw_mask is None:
        parser.error("--raw-mask is required with the development --side mode")
    if args.bundle is not None and args.containers is not None:
        parser.error("--containers cannot be combined with --bundle")
    if args.bundle is not None and args.raw_mask is not None:
        parser.error("--raw-mask is embedded in --bundle and must not be supplied")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.clean_decoder.is_file():
        raise FileNotFoundError(args.clean_decoder)
    expected_mask_bytes = BASE_MASK_LEVELS * ((NORMATIVE_BLOCK_LENGTH + 7) // 8)

    if args.bundle is not None:
        with args.bundle.open("rb") as handle:
            prelude = read_bundle_prelude(handle)
            side = prelude.side
            frames = read_all_frames(handle, side.chunk_count)
        raw_mask = prelude.raw_mask
        bundle_path = args.bundle
        container_path = None
        physical_prelude_bytes = (
            len(prelude.header)
            + len(prelude.compressed_side)
            + len(prelude.compressed_mask)
        )
        self_contained = True
    else:
        if not args.raw_mask.is_file():
            raise FileNotFoundError(args.raw_mask)
        if args.raw_mask.stat().st_size != expected_mask_bytes:
            raise ValueError(
                f"pinned raw mask is {args.raw_mask.stat().st_size} bytes, "
                f"expected {expected_mask_bytes}"
            )
        raw_mask = args.raw_mask.read_bytes()
        side = read_side_file(args.side)
        with args.containers.open("rb") as handle:
            frames = read_all_frames(handle, side.chunk_count)
        bundle_path = None
        container_path = args.containers
        prelude = None
        physical_prelude_bytes = len(side.blob)
        self_contained = False

    stream_digest = hashlib.sha256()
    development_combined_digest = hashlib.sha256(side.blob)
    stream_bytes = 0
    for frame in frames:
        stream_digest.update(frame.literal)
        development_combined_digest.update(frame.literal)
        stream_bytes += len(frame.literal)
    encoded_bytes = physical_prelude_bytes + stream_bytes
    if bundle_path is not None and bundle_path.stat().st_size != encoded_bytes:
        raise AssertionError("parsed physical bundle byte count does not match filesystem size")
    if container_path is not None and container_path.stat().st_size != stream_bytes:
        raise AssertionError("parsed container byte count does not match filesystem size")

    if args.reconstruction.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {args.reconstruction}; pass --overwrite"
        )
    partial_reconstruction = Path(str(args.reconstruction) + ".partial")
    if partial_reconstruction.exists():
        raise FileExistsError(f"stale partial output exists: {partial_reconstruction}")
    args.reconstruction.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    dtype = OUTPUT_DTYPES[args.reconstruction_dtype]
    canonical = np.memmap(
        partial_reconstruction,
        dtype=dtype,
        mode="w+",
        shape=(side.label_count, side.group_values),
    )
    seen = np.zeros(side.label_count, dtype=np.uint8)
    tasks = list(zip(side.profiles, frames, strict=True))
    chunk_receipts: list[dict[str, object]] = []
    digest_of_frequency_digests = hashlib.sha256()
    digest_of_index_digests = hashlib.sha256()
    for completed, result in enumerate(
        decode_tasks(tasks, args.clean_decoder, raw_mask, args.workers), start=1
    ):
        chunk_index = int(result["chunk_index"])
        if chunk_index != completed - 1:
            raise AssertionError("parallel decoder returned chunks out of canonical order")
        begin = chunk_index * side.groups_per_chunk
        end = begin + side.groups_per_chunk
        members = side.stable_order[begin:end]
        if members.size != side.groups_per_chunk:
            raise AssertionError(f"chunk {chunk_index} membership is incomplete")
        if np.any(seen[members]):
            raise AssertionError(f"chunk {chunk_index} repeats canonical group membership")
        normalized = np.asarray(result.pop("reconstruction"), dtype=np.float64)
        if normalized.size != NORMATIVE_BLOCK_LENGTH:
            raise AssertionError(f"chunk {chunk_index} reconstruction length mismatch")
        raw = normalized.reshape(side.groups_per_chunk, side.group_values)
        raw *= side.qscales[members, None]
        canonical[members, :] = raw.astype(dtype, copy=False)
        seen[members] = 1
        frame = frames[chunk_index]
        profile = side.profiles[chunk_index]
        frequency_digest = str(result["frequency_u16_sha256"])
        index_digest = str(result["reconstruction_indices_i16_sha256"])
        digest_of_frequency_digests.update(bytes.fromhex(frequency_digest))
        digest_of_index_digests.update(bytes.fromhex(index_digest))
        chunk_receipts.append(
            {
                "chunk_index": chunk_index,
                "container_bytes": len(frame.literal),
                "container_sha256": frame.sha256,
                "logical_bits": frame.logical_bits,
                "escape_count": int(frame.escape_positions.size),
                "fp32_decoder_scale": frame.scale,
                "arithmetic_padding_bits": frame.arithmetic_padding_bits,
                "arithmetic_padding_zero": True,
                "tail_padding_bits": frame.tail_padding_bits,
                "tail_padding_zero": True,
                "test_distortion_binary64": profile.distortion,
                "eta_binary64": profile.eta,
                "alphabet_size": profile.alphabet_size,
                "profile_literal_bytes_hex": profile.literal_bytes_hex,
                "first_stable_member_ordinal": int(members[0]),
                "last_stable_member_ordinal": int(members[-1]),
                **result,
            }
        )
        print(
            f"[{completed}/{side.chunk_count}] decoded chunk {chunk_index:03d}",
            file=sys.stderr,
            flush=True,
        )
    if not np.all(seen == 1):
        raise AssertionError("canonical scatter did not write every group exactly once")
    canonical.flush()
    del canonical
    expected_reconstruction_bytes = side.label_count * side.group_values * dtype.itemsize
    if partial_reconstruction.stat().st_size != expected_reconstruction_bytes:
        raise AssertionError("canonical reconstruction byte count mismatch")
    reconstruction_sha256 = sha256_path(partial_reconstruction)
    os.replace(partial_reconstruction, args.reconstruction)

    panel_values = side.label_count * side.group_values
    side_sha256 = sha256_bytes(side.blob)
    encoded_sha256 = (
        sha256_path(bundle_path)
        if bundle_path is not None
        else development_combined_digest.hexdigest()
    )
    receipt = {
        "format": "continuous reverse-waterfilled PLTE independent outer decode v1",
        "status": "passed",
        "strict_ptq": True,
        "independence": {
            "read_encoder_json": False,
            "read_exploratory_manifest": False,
            "read_normalized_or_raw_source": False,
            "encoder_probability_arrays_used": False,
            "membership_derived_from_literal_side_only": True,
        },
        "normative_codec_constants": {
            "sigma_source": NORMATIVE_SIGMA_SOURCE,
            "frozen_seed": NORMATIVE_FROZEN_SEED,
            "trial_per_independently_encoded_chunk": NORMATIVE_TRIAL,
            "base_serialized_mask_levels": BASE_MASK_LEVELS,
            "higher_levels": "procedural all-zero freeze flags (fully open)",
        },
        "geometry": {
            "canonical_blocks": side.block_count,
            "groups_per_canonical_block": side.groups_per_chunk,
            "group_values": side.group_values,
            "polar_chunks": side.chunk_count,
            "polar_block_values": NORMATIVE_BLOCK_LENGTH,
            "panel_values": panel_values,
        },
        "side": {
            "bytes": len(side.blob),
            "sha256": side_sha256,
            "exact_eof": True,
            "lambda_variance_binary64": side.lambda_variance,
            "serialized_lut_sha256": sha256_bytes(side.lut.astype("<f8", copy=False).tobytes()),
            "serialized_fp32_scales_sha256": sha256_bytes(side.block_scales.astype("<f4", copy=False).tobytes()),
            "packed_label_count": side.label_count,
            "decoded_labels_i8_sha256": sha256_bytes(side.labels.astype("i1", copy=False).tobytes()),
            "stable_order_i64_sha256": sha256_bytes(side.stable_order.astype("<i8", copy=False).tobytes()),
            "adjacent_equal_qvariance_pairs": side.adjacent_qvariance_ties,
            "chunk_boundaries_splitting_exact_qvariance_ties": side.tie_boundaries_split_by_chunks,
            "binary64_profile_bytes": side.chunk_count * PROFILE.size,
        },
        "encoded_stream": {
            "layout": (
                "WFOUTR01 header, LZMA-XZ side, BZ2 six-mask, raw self-delimiting PLTE containers"
                if self_contained
                else "development-only raw WFPLTE01 side plus external mask and raw containers"
            ),
            "self_contained_decoder_side_and_mask": self_contained,
            "physical_prelude_bytes": physical_prelude_bytes,
            "container_stream_bytes": stream_bytes,
            "container_stream_sha256": stream_digest.hexdigest(),
            "combined_encoded_bytes": encoded_bytes,
            "combined_encoded_sha256": encoded_sha256,
            "exact_eof_after_declared_chunk_count": True,
            "all_arithmetic_padding_zero": True,
            "all_sparse_tail_padding_zero": True,
            "actual_all_in_bpw": encoded_bytes * 8.0 / panel_values,
        },
        "decoder_assets": {
            "raw_mask_bytes": len(raw_mask),
            "raw_mask_sha256": sha256_bytes(raw_mask),
            "raw_mask_embedded_and_physically_charged": self_contained,
            "clean_decoder_sha256": sha256_path(args.clean_decoder),
            "outer_decoder_sha256": sha256_path(Path(__file__)),
        },
        "reconstruction": {
            "path": str(args.reconstruction),
            "storage_order": "canonical block, then canonical group, then value",
            "shape": [side.block_count, side.groups_per_chunk * side.group_values],
            "dtype": dtype.str,
            "bytes": expected_reconstruction_bytes,
            "sha256": reconstruction_sha256,
            "every_canonical_group_written_exactly_once": True,
            "digest_of_ordered_chunk_frequency_digests": digest_of_frequency_digests.hexdigest(),
            "digest_of_ordered_chunk_index_digests": digest_of_index_digests.hexdigest(),
        },
        "alphabet_census": {
            str(size): sum(profile.alphabet_size == size for profile in side.profiles)
            for size in sorted(set(ALPHABET_BY_CODE.values()))
        },
        "workers": args.workers,
        "chunks": chunk_receipts,
    }
    if prelude is not None:
        receipt["outer_bundle"] = {
            "magic": OUTER_MAGIC.decode("ascii"),
            "version": OUTER_VERSION,
            "header_bytes": len(prelude.header),
            "header_sha256": sha256_bytes(prelude.header),
            "side_codec": "LZMA-XZ preset 9",
            "side_raw_bytes": len(side.blob),
            "side_compressed_bytes": len(prelude.compressed_side),
            "side_compressed_sha256": sha256_bytes(prelude.compressed_side),
            "mask_codec": "BZ2 level 9",
            "mask_raw_bytes": len(raw_mask),
            "mask_compressed_bytes": len(prelude.compressed_mask),
            "mask_compressed_sha256": sha256_bytes(prelude.compressed_mask),
            "decompressor_exact_eof": True,
        }
    atomic_json(args.receipt, receipt, args.overwrite)
    print(json.dumps({key: value for key, value in receipt.items() if key != "chunks"}, indent=2))


if __name__ == "__main__":
    main()
