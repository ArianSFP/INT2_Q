#!/usr/bin/env python3
"""Strict-PTQ polar-lattice Gaussian source-coding gate.

This is an independent Python/CuPy reproduction scaffold for the public
PolarLatticeQuantization construction (Liu, Shi, Ling).  It deliberately
starts with a synthetic Gaussian source and reports both distortion and the
decoder-visible prior codelength of the polar decisions.  No model weights,
calibration data, retraining, QAT, or task loss are used.

The first gate reuses the published Tal--Vardy reliability order at D=0.20
while allowing the test-channel distortion to move.  It is therefore a
screen, not yet the final normative codec.  A passing screen must later be
reconstructed with reliability tables generated for the exact target and an
actual arithmetic-coded bitstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cupy as cp
import numpy as np
from scipy.io import loadmat


def bit_reverse_indices(n: int) -> np.ndarray:
    k = int(math.log2(n))
    if 1 << k != n:
        raise ValueError("block length must be a power of two")
    x = np.arange(n, dtype=np.uint32)
    out = np.zeros(n, dtype=np.uint32)
    for _ in range(k):
        out = (out << 1) | (x & 1)
        x >>= 1
    return out.astype(np.int64)


def sc_layers(n: int) -> np.ndarray:
    """Zero-based translation of PolarSCDecodePrepare.m."""
    k = int(math.log2(n))
    out = np.ones(n + 1, dtype=np.int32)
    out[0] = k
    for i_one in range(2, n + 1):
        end_layer = 1
        idx = i_one
        while idx % 2 == 1:
            end_layer += 1
            idx = (idx + 1) // 2
        out[i_one - 1] = end_layer
    return out


def polar_transform(bits: np.ndarray) -> np.ndarray:
    """Exact vectorized translation of Encoder4Polar.m."""
    x = np.asarray(bits, dtype=np.uint8).copy()
    n = x.size
    step = 1
    while step < n:
        view = x.reshape(-1, 2 * step)
        view[:, :step] ^= view[:, step:]
        step *= 2
    return x


@dataclass
class SCResult:
    external_u: np.ndarray
    internal_u: np.ndarray
    selected_nll_bits: float
    selected_count: int
    selected_bits: np.ndarray
    selected_freq1_u16: np.ndarray


def sc_encode_ratio(
    leaf_lr: np.ndarray,
    freeze_flag: np.ndarray,
    frozen_external: np.ndarray,
    reverse: np.ndarray,
    layers: np.ndarray,
    *,
    rng: np.random.Generator,
    decision: str,
    forced_internal: np.ndarray | None = None,
    score_selected: bool = False,
    arithmetic_decoder: object | None = None,
) -> SCResult:
    """Successive-cancellation encoder in likelihood-ratio form.

    The register updates are a direct zero-based port of the authors'
    PolarNewLossySCEncoder.m, LRCalc4PolarSC.m, and MiuCalc4PolarSC.m.
    If ``forced_internal`` is supplied, it evaluates the causal probability
    of an already chosen path, which is the decoder-visible entropy model.
    """
    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    n = lr_in.size
    depth = int(math.log2(n))
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    u = np.zeros(n, dtype=np.uint8)
    nll = 0.0
    selected_count = 0
    selected_bits: list[int] = []
    selected_freq1: list[int] = []

    def adjust(a: np.ndarray | float) -> np.ndarray | float:
        return np.clip(a, 1e-30, 1e30)

    for i0 in range(n):
        i_one = i0 + 1
        if i_one == 1:
            end = int(layers[i0])
            col = end - 1
            a = lr_in[0::2]
            b = lr_in[1::2]
            lr_reg[:, col] = adjust((a * b + 1.0) / (a + b))
            for lev_one in range(end - 1, 0, -1):
                src_col = lev_one
                dst_col = lev_one - 1
                count = 1 << lev_one
                a = lr_reg[0:count:2, src_col]
                b = lr_reg[1:count:2, src_col]
                lr_reg[: count // 2, dst_col] = adjust((a * b + 1.0) / (a + b))
        elif i_one == n // 2 + 1:
            end = int(layers[i0])
            col = end - 1
            a = lr_in[0::2]
            b = lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, col] = adjust(np.power(a, 1 - 2 * used) * b)
            for lev_one in range(end - 1, 0, -1):
                src_col = lev_one
                dst_col = lev_one - 1
                count = 1 << lev_one
                a = lr_reg[0:count:2, src_col]
                b = lr_reg[1:count:2, src_col]
                lr_reg[: count // 2, dst_col] = adjust((a * b + 1.0) / (a + b))
        elif i_one % 2 == 0:
            end = int(layers[i0])
            dst_col = end - 1
            src_col = end
            a = float(lr_reg[0, src_col])
            b = float(lr_reg[1, src_col])
            used = int(mu_reg[0, 0])
            lr_reg[0, dst_col] = adjust((a ** (1 - 2 * used)) * b)
        else:
            end = int(layers[i0])
            dst_col = end - 1
            src_col = end
            count = 1 << end
            a = lr_reg[0:count:2, src_col]
            b = lr_reg[1:count:2, src_col]
            used = mu_reg[: count // 2, dst_col].astype(np.int8)
            lr_reg[: count // 2, dst_col] = adjust(np.power(a, 1 - 2 * used) * b)
            for lev_one in range(end - 1, 0, -1):
                src_col2 = lev_one
                dst_col2 = lev_one - 1
                count2 = 1 << lev_one
                a2 = lr_reg[0:count2:2, src_col2]
                b2 = lr_reg[1:count2:2, src_col2]
                lr_reg[: count2 // 2, dst_col2] = adjust((a2 * b2 + 1.0) / (a2 + b2))

        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        p1 = 1.0 / (1.0 + root_lr)
        if arithmetic_decoder is not None and freeze_flag[i0]:
            bit = int(frozen_external[reverse[i0]])
        elif arithmetic_decoder is not None:
            freq1 = min(65535, max(1, int(math.floor(p1 * 65536.0 + 0.5))))
            bit = int(arithmetic_decoder.decode(freq1))
        elif forced_internal is not None:
            bit = int(forced_internal[i0])
        elif freeze_flag[i0]:
            bit = int(frozen_external[reverse[i0]])
        elif decision == "map":
            bit = int(root_lr < 1.0)
        elif decision == "random":
            bit = int(rng.random() < p1)
        else:
            raise ValueError(f"unknown decision {decision}")
        u[i0] = bit

        if score_selected and not freeze_flag[i0]:
            prob = p1 if bit else (1.0 - p1)
            nll -= math.log2(max(prob, 1e-300))
            selected_count += 1
            selected_bits.append(bit)
            selected_freq1.append(min(65535, max(1, int(math.floor(p1 * 65536.0 + 0.5)))))

        # Direct port of MiuCalc4PolarSC.m.
        if i_one % 2 == 1:
            mu_reg[0, 0] = bit
        else:
            end = int(layers[i_one])  # MATLAB indexes SCLayer(I+1).
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = bit
            for j_one in range(1, end):
                length = 1 << (j_one - 1)
                left = mu_reg[:length, j_one - 1]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[: 2 * length] = merged
            mu_reg[: 1 << max(end - 1, 0), end - 1] = temp

    return SCResult(
        external_u=u[reverse].copy(),
        internal_u=u,
        selected_nll_bits=nll,
        selected_count=selected_count,
        selected_bits=np.asarray(selected_bits, dtype=np.uint8),
        selected_freq1_u16=np.asarray(selected_freq1, dtype=np.uint16),
    )


def arithmetic_encode_binary(bits: np.ndarray, freq1: np.ndarray) -> tuple[bytes, int]:
    """32-bit binary arithmetic coder with a fixed 16-bit frequency total."""
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for bit_u8, f1_u16 in zip(bits, freq1, strict=True):
        bit = int(bit_u8)
        f1 = int(f1_u16)
        f0 = 65536 - f1
        width = high - low + 1
        split = low + (width * f0 // 65536) - 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1

    pending += 1
    emit(0 if low < quarter else 1)
    logical_bits = len(output)
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, logical_bits


def arithmetic_decode_binary(payload: bytes, logical_bits: int, freq1: np.ndarray) -> np.ndarray:
    decoder = ArithmeticBinaryDecoder(payload, logical_bits)
    return np.asarray([decoder.decode(int(f)) for f in freq1], dtype=np.uint8)


LOGICAL_LENGTH_BITS = 20
ESCAPE_POSITION_BITS = 18
ESCAPE_VALUE_BITS = 16
ESCAPE_RECORD_BITS = ESCAPE_POSITION_BITS + ESCAPE_VALUE_BITS
MAX_ESCAPE_RECORDS = (1 << (32 - LOGICAL_LENGTH_BITS)) - 1


def pack_escape_records(positions: np.ndarray, bf16_values: np.ndarray) -> bytes:
    """Pack sorted (18-bit position, 16-bit BF16) lossless tail escapes."""
    if positions.size != bf16_values.size:
        raise ValueError("escape position/value length mismatch")
    combined = 0
    for position, value in zip(positions, bf16_values, strict=True):
        p = int(position)
        if p < 0 or p >= (1 << ESCAPE_POSITION_BITS):
            raise ValueError(f"escape position outside 18-bit block: {p}")
        combined = (combined << ESCAPE_RECORD_BITS) | (p << ESCAPE_VALUE_BITS) | int(value)
    meaningful_bits = ESCAPE_RECORD_BITS * int(positions.size)
    padding_bits = (-meaningful_bits) % 8
    combined <<= padding_bits
    return combined.to_bytes((meaningful_bits + padding_bits) // 8, "big")


def unpack_escape_records(payload: bytes, count: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Inverse of pack_escape_records, including a zero-padding audit."""
    meaningful_bits = ESCAPE_RECORD_BITS * count
    expected_bytes = (meaningful_bits + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError((len(payload), expected_bytes))
    padding_bits = expected_bytes * 8 - meaningful_bits
    combined = int.from_bytes(payload, "big")
    padding_is_zero = (combined & ((1 << padding_bits) - 1)) == 0 if padding_bits else True
    combined >>= padding_bits
    positions = np.empty(count, dtype=np.int32)
    values = np.empty(count, dtype=np.uint16)
    mask = (1 << ESCAPE_RECORD_BITS) - 1
    for index in range(count - 1, -1, -1):
        record = combined & mask
        combined >>= ESCAPE_RECORD_BITS
        positions[index] = record >> ESCAPE_VALUE_BITS
        values[index] = record & ((1 << ESCAPE_VALUE_BITS) - 1)
    if count and (np.any(positions[1:] <= positions[:-1]) or np.any(positions < 0)):
        raise ValueError("escape positions are not strictly increasing")
    return positions, values, padding_is_zero


class ArithmeticBinaryDecoder:
    """Streaming counterpart: probabilities are supplied causally by the decoder."""

    def __init__(self, payload: bytes, logical_bits: int):
        self.full = 1 << 32
        self.half = 1 << 31
        self.quarter = 1 << 30
        self.three_quarters = 3 << 30
        self.packed_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
        self.logical_bits = logical_bits
        self.cursor = 0
        self.low = 0
        self.high = self.full - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) & (self.full - 1)) | self._read_bit()

    def _read_bit(self) -> int:
        if self.cursor >= self.logical_bits:
            return 0
        value = int(self.packed_bits[self.cursor])
        self.cursor += 1
        return value

    def decode(self, freq1: int) -> int:
        f1 = int(freq1)
        f0 = 65536 - f1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * 65536 - 1) // width
        split = self.low + (width * f0 // 65536) - 1
        if scaled < f0:
            bit = 0
            self.high = split
        else:
            bit = 1
            self.low = split + 1
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & (self.full - 1)
            self.high = ((self.high << 1) & (self.full - 1)) | 1
            self.code = ((self.code << 1) & (self.full - 1)) | self._read_bit()
        return bit


def _legacy_arithmetic_decode_binary_removed(payload: bytes, logical_bits: int, freq1: np.ndarray) -> np.ndarray:
    """Retained only as unreachable source history during the audit transition."""
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    packed_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    cursor = 0

    def read_bit() -> int:
        nonlocal cursor
        if cursor >= logical_bits:
            return 0
        value = int(packed_bits[cursor])
        cursor += 1
        return value

    low = 0
    high = full - 1
    code = 0
    for _ in range(32):
        code = ((code << 1) & (full - 1)) | read_bit()
    decoded = np.empty(freq1.size, dtype=np.uint8)
    for index, f1_u16 in enumerate(freq1):
        f1 = int(f1_u16)
        f0 = 65536 - f1
        width = high - low + 1
        scaled = ((code - low + 1) * 65536 - 1) // width
        split = low + (width * f0 // 65536) - 1
        if scaled < f0:
            decoded[index] = 0
            high = split
        else:
            decoded[index] = 1
            low = split + 1
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
            code = ((code << 1) & (full - 1)) | read_bit()
    return decoded


def periodic_binary_capacity(sigma: float, grid: int = 1 << 17, neighbors: int = 16) -> float:
    """Capacity of uniform binary input through (X+N(0,sigma^2)) mod 2."""
    y = (np.arange(grid, dtype=np.float64) + 0.5) * (2.0 / grid)
    ks = np.arange(-neighbors, neighbors + 1, dtype=np.float64)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    p0 = np.exp(-0.5 * ((y[:, None] + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    p1 = np.exp(-0.5 * ((y[:, None] - 1.0 + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    mix = 0.5 * (p0 + p1)
    post = np.divide(p1, p0 + p1, out=np.full_like(p1, 0.5), where=(p0 + p1) > 0)
    h = -(post * np.log2(np.maximum(post, 1e-300)) + (1.0 - post) * np.log2(np.maximum(1.0 - post, 1e-300)))
    return float(1.0 - np.sum(mix * h) * (2.0 / grid))


def reliability_freeze_flags(repo: Path, n: int, capacities: Iterable[float]) -> list[np.ndarray]:
    reverse = bit_reverse_indices(n)
    logn = int(math.log2(n))
    flags: list[np.ndarray] = []
    for level, capacity in enumerate(capacities, start=1):
        # The authors' high-SNR numerical construction explicitly leaves
        # levels 4--6 fully open; their near-one numerical capacities are not
        # used to freeze an arbitrary handful of positions.
        if level >= 4:
            flags.append(np.zeros(n, dtype=np.uint8))
            continue
        keep = min(n, max(0, int(math.ceil(n * float(capacity)))))
        flag = np.zeros(n, dtype=np.uint8)
        if keep == n:
            flags.append(flag)
            continue
        if keep == 0:
            flag[:] = 1
            flags.append(flag)
            continue
        if level <= 3:
            path = repo / f"Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_{level}_n_{logn}.mat"
            pe = np.asarray(loadmat(path)["PeLast"]).ravel()
            zn = pe[reverse]
            ordered = np.argsort(zn, kind="stable")
            freeze_index = np.sort(ordered[keep:])
            freeze_resolved = reverse[freeze_index]
            flag[freeze_resolved] = 1
        else:
            # The public reference omits tables for nearly-perfect levels.
            # Freeze the earliest internal positions only as a conservative
            # screen; a promoted run must build exact target reliability sets.
            flag[: n - keep] = 1
        flags.append(flag)
    return flags


def leaf_likelihood_ratios_gpu(
    y: cp.ndarray,
    alphabet: cp.ndarray,
    weights: cp.ndarray,
    distortion: float,
    previous: cp.ndarray,
    level: int,
) -> np.ndarray:
    dens = cp.exp(-0.5 * ((y[:, None] - alphabet[None, :]) ** 2) / distortion) * weights[None, :]
    lower_mod = 1 << (level - 1)
    bit = 1 << (level - 1)
    out = cp.empty(y.size, dtype=cp.float64)
    for context in range(lower_mod):
        pos = cp.where(previous == context)[0]
        if pos.size == 0:
            continue
        idx0 = cp.asarray([j for j in range(alphabet.size) if (j % lower_mod == context and (j & bit) == 0)])
        idx1 = cp.asarray([j for j in range(alphabet.size) if (j % lower_mod == context and (j & bit) != 0)])
        p0 = dens[pos[:, None], idx0[None, :]].sum(axis=1)
        p1 = dens[pos[:, None], idx1[None, :]].sum(axis=1)
        out[pos] = cp.clip(p0 / cp.maximum(p1, 1e-300), 1e-30, 1e30)
    return cp.asnumpy(out)


def leaf_prior_ratios(weights: np.ndarray, previous: np.ndarray, level: int) -> np.ndarray:
    lower_mod = 1 << (level - 1)
    bit = 1 << (level - 1)
    values = np.empty(lower_mod, dtype=np.float64)
    for context in range(lower_mod):
        idx0 = [j for j in range(weights.size) if j % lower_mod == context and (j & bit) == 0]
        idx1 = [j for j in range(weights.size) if j % lower_mod == context and (j & bit) != 0]
        values[context] = weights[idx0].sum() / max(weights[idx1].sum(), 1e-300)
    return np.clip(values[previous], 1e-30, 1e30)


def run_trial(args: argparse.Namespace, trial: int, capacities: list[float], flags: list[np.ndarray]) -> dict:
    n = args.block_length
    reverse = bit_reverse_indices(n)
    layers = sc_layers(n)
    rng = np.random.default_rng(args.seed + 104729 * trial)
    cp_rng = cp.random.RandomState(args.seed + 104729 * trial)
    source_row: dict[str, object]
    source_bf16_u16: np.ndarray | None = None
    if args.input_bf16 is None:
        y_gpu = cp_rng.normal(0.0, args.sigma_source, size=n, dtype=cp.float64)
        literal_source_gpu = y_gpu
        source_row = {"kind": "synthetic_gaussian", "trial_seed": args.seed + 104729 * trial}
    else:
        raw = np.fromfile(args.input_bf16, dtype="<u2")
        values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        block_index = args.input_block_start + trial
        begin = block_index * n
        end = begin + n
        if end > values.size:
            raise ValueError(
                f"input block {block_index} ends at {end}, beyond {values.size} values"
            )
        block = cp.asarray(values[begin:end], dtype=cp.float64)
        source_bf16_u16 = raw[begin:end].copy()
        literal_source_gpu = block
        block_rms = float(cp.sqrt(cp.mean(block * block)).get())
        if not math.isfinite(block_rms) or block_rms <= 0:
            raise ValueError(f"invalid block RMS {block_rms}")
        y_gpu = block * (args.sigma_source / block_rms)
        source_row = {
            "kind": "frozen_bf16_weight_block",
            "path": str(args.input_bf16),
            "block_index": block_index,
            "values": n,
            "block_bf16_sha256": hashlib.sha256(source_bf16_u16.tobytes()).hexdigest(),
            "block_rms_fp64": block_rms,
            "decoder_scale_fp32": float(np.float32(block_rms / args.sigma_source)),
        }

    sigma_recon = math.sqrt(args.sigma_source**2 - args.test_distortion)
    alphabet_np = args.eta * np.arange(-args.alphabet_size // 2 + 1, args.alphabet_size // 2 + 1, dtype=np.float64)
    weights_np = np.exp(-0.5 * (alphabet_np / sigma_recon) ** 2)
    alphabet_gpu = cp.asarray(alphabet_np)
    weights_gpu = cp.asarray(weights_np)

    previous_gpu = cp.zeros(n, dtype=cp.int16)
    previous_np = np.zeros(n, dtype=np.int16)
    x_levels: list[np.ndarray] = []
    ideal_bits = 0.0
    selected = 0
    level_rows = []
    level_audit: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for level in range(1, int(math.log2(args.alphabet_size)) + 1):
        posterior_lr = leaf_likelihood_ratios_gpu(
            y_gpu, alphabet_gpu, weights_gpu, args.test_distortion, previous_gpu, level
        )
        frozen_rng = np.random.default_rng(args.seed + 104729 * trial + 1000003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        chosen = sc_encode_ratio(
            posterior_lr,
            flags[level - 1],
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision=args.decision,
        )
        x_bit = polar_transform(chosen.external_u)

        prior_lr = leaf_prior_ratios(weights_np, previous_np, level)
        scored = sc_encode_ratio(
            prior_lr,
            flags[level - 1],
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision="map",
            forced_internal=chosen.internal_u,
            score_selected=True,
        )
        ideal_bits += scored.selected_nll_bits
        selected += scored.selected_count
        level_audit.append((frozen_external, chosen.internal_u, scored.selected_freq1_u16))
        x_levels.append(x_bit)
        previous_np += (1 << (level - 1)) * x_bit.astype(np.int16)
        previous_gpu = cp.asarray(previous_np)
        level_rows.append(
            {
                "level": level,
                "capacity_schedule": capacities[level - 1],
                "selected_fraction": float((flags[level - 1] == 0).mean()),
                "selected_nll_bits": scored.selected_nll_bits,
                "selected_nll_bpw": scored.selected_nll_bits / n,
            }
        )

    reconstruct_gpu = alphabet_gpu[previous_gpu]
    squared = cp.square(y_gpu - reconstruct_gpu)
    source_energy = cp.square(y_gpu)
    distortion_abs = float(cp.mean(squared).get())
    normalized_relative_mse = float((cp.sum(squared) / cp.sum(source_energy)).get())

    all_selected_bits = np.concatenate(
        [
            chosen_internal[flags[level_index] == 0]
            for level_index, (_, chosen_internal, _) in enumerate(level_audit)
        ]
    )
    all_freq1 = np.concatenate([row[2] for row in level_audit])
    payload, arithmetic_logical_bits = arithmetic_encode_binary(all_selected_bits, all_freq1)
    decoded_selected = arithmetic_decode_binary(payload, arithmetic_logical_bits, all_freq1)
    arithmetic_bits_match = bool(np.array_equal(decoded_selected, all_selected_bits))

    # Normative streaming decode: each probability is generated before its
    # bit is read, using only fixed frozen bits, already-decoded decisions,
    # and reconstructed lower levels. No encoder probability list is passed
    # to this decoder.
    online_decoder = ArithmeticBinaryDecoder(payload, arithmetic_logical_bits)
    decoded_previous = np.zeros(n, dtype=np.int16)
    regenerated_freqs: list[np.ndarray] = []
    online_selected_parts: list[np.ndarray] = []
    decoded_x_levels: list[np.ndarray] = []
    causal_frozen_bits_match = True
    for level_index in range(len(x_levels)):
        encoder_frozen_external, _, original_freqs = level_audit[level_index]
        decoder_frozen_rng = np.random.default_rng(
            args.seed + 104729 * trial + 1000003 * (level_index + 1)
        )
        frozen_external = decoder_frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        causal_frozen_bits_match &= bool(
            np.array_equal(frozen_external, encoder_frozen_external)
        )
        flag = flags[level_index]
        prior_lr = leaf_prior_ratios(weights_np, decoded_previous, level_index + 1)
        rescored = sc_encode_ratio(
            prior_lr,
            flag,
            frozen_external,
            reverse,
            layers,
            rng=rng,
            decision="map",
            score_selected=True,
            arithmetic_decoder=online_decoder,
        )
        regenerated_freqs.append(rescored.selected_freq1_u16)
        online_selected_parts.append(rescored.selected_bits)
        decoded_x = polar_transform(rescored.external_u)
        decoded_x_levels.append(decoded_x)
        decoded_previous += (1 << level_index) * decoded_x.astype(np.int16)
        if not np.array_equal(rescored.selected_freq1_u16, original_freqs):
            raise AssertionError(f"causal frequency mismatch at level {level_index + 1}")

    causal_frequencies_match = bool(np.array_equal(np.concatenate(regenerated_freqs), all_freq1))
    online_arithmetic_bits_match = bool(
        np.array_equal(np.concatenate(online_selected_parts), all_selected_bits)
    )
    reconstruction_indices_match = bool(np.array_equal(decoded_previous, previous_np))
    if not (
        arithmetic_bits_match
        and online_arithmetic_bits_match
        and causal_frequencies_match
        and causal_frozen_bits_match
        and reconstruction_indices_match
    ):
        raise AssertionError("arithmetic round-trip audit failed")

    # Literal per-block container.  The low 20 bits of the first u32 hold the
    # arithmetic logical length and the high 12 bits hold a sparse escape
    # count.  Each escape stores an absolute 18-bit position and the exact
    # 16-bit BF16 source value.  This lossless tail side-channel spends only
    # entropy slack up to the configured container cap and repairs the rare
    # heavy-tail coordinates for which a finite 64-point lattice clips.
    decoder_scale = float(source_row.get("decoder_scale_fp32", np.float32(1.0)))
    decoded_reconstruct_gpu = alphabet_gpu[cp.asarray(decoded_previous)] * decoder_scale
    base_literal_squared = cp.square(literal_source_gpu - decoded_reconstruct_gpu)
    base_literal_relative_mse = float(
        (cp.sum(base_literal_squared) / cp.sum(cp.square(literal_source_gpu))).get()
    )
    base_container_bytes = 8 + len(payload)
    if args.container_cap_bytes and base_container_bytes > args.container_cap_bytes:
        raise RuntimeError(
            f"base polar container {base_container_bytes} exceeds enforced "
            f"{args.container_cap_bytes}-byte slot"
        )
    escape_positions = np.empty(0, dtype=np.int32)
    escape_values = np.empty(0, dtype=np.uint16)
    if source_bf16_u16 is not None and args.container_cap_bytes > base_container_bytes:
        available_bits = 8 * (args.container_cap_bytes - base_container_bytes)
        escape_count = min(MAX_ESCAPE_RECORDS, available_bits // ESCAPE_RECORD_BITS)
        while (ESCAPE_RECORD_BITS * escape_count + 7) // 8 > available_bits // 8:
            escape_count -= 1
        if escape_count:
            errors = cp.asnumpy(base_literal_squared)
            # Stable ordering makes equal-error selection normative.
            chosen = np.argsort(-errors, kind="stable")[:escape_count]
            escape_positions = np.sort(chosen.astype(np.int32))
            escape_values = source_bf16_u16[escape_positions].copy()

    if arithmetic_logical_bits >= (1 << LOGICAL_LENGTH_BITS):
        raise ValueError("arithmetic payload length does not fit the 20-bit header field")
    escape_payload = pack_escape_records(escape_positions, escape_values)
    header_word = arithmetic_logical_bits | (int(escape_positions.size) << LOGICAL_LENGTH_BITS)
    container = struct.pack("<If", header_word, decoder_scale) + payload + escape_payload

    # Parse the exact byte container and reconstruct only from decoded fields.
    decoded_header_word, decoded_scale = struct.unpack("<If", container[:8])
    decoded_logical_bits = decoded_header_word & ((1 << LOGICAL_LENGTH_BITS) - 1)
    decoded_escape_count = decoded_header_word >> LOGICAL_LENGTH_BITS
    decoded_payload_bytes = (decoded_logical_bits + 7) // 8
    decoded_payload = container[8 : 8 + decoded_payload_bytes]
    decoded_escape_payload = container[8 + decoded_payload_bytes :]
    decoded_escape_positions, decoded_escape_values, escape_padding_is_zero = unpack_escape_records(
        decoded_escape_payload, decoded_escape_count
    )
    container_header_roundtrip = bool(
        decoded_logical_bits == arithmetic_logical_bits
        and decoded_escape_count == escape_positions.size
        and decoded_payload == payload
    )
    escape_records_roundtrip = bool(
        np.array_equal(decoded_escape_positions, escape_positions)
        and np.array_equal(decoded_escape_values, escape_values)
    )
    if not (container_header_roundtrip and escape_records_roundtrip and escape_padding_is_zero):
        raise AssertionError("literal tail-escape container round-trip failed")

    literal_reconstruct_gpu = alphabet_gpu[cp.asarray(decoded_previous)] * float(decoded_scale)
    if decoded_escape_count:
        decoded_escape_float = (
            (decoded_escape_values.astype(np.uint32) << np.uint32(16)).view(np.float32)
        )
        literal_reconstruct_gpu[cp.asarray(decoded_escape_positions)] = cp.asarray(
            decoded_escape_float, dtype=cp.float64
        )
    literal_squared = cp.square(literal_source_gpu - literal_reconstruct_gpu)
    literal_absolute_mse = float(cp.mean(literal_squared).get())
    relative_mse = float(
        (cp.sum(literal_squared) / cp.sum(cp.square(literal_source_gpu))).get()
    )
    framing_bits = 64
    total_bits = len(container) * 8
    rate = total_bits / n
    gaussian = 2.0 ** (-2.0 * rate)
    gap_db = 10.0 * math.log10(relative_mse / gaussian)
    threshold = (10.0 ** (0.10 / 10.0)) * gaussian
    return {
        "trial": trial,
        "source": source_row,
        "absolute_mse": distortion_abs,
        "literal_decoded_absolute_mse": literal_absolute_mse,
        "normalized_relative_mse_before_fp32_scale": normalized_relative_mse,
        "base_literal_relative_mse_before_escapes": base_literal_relative_mse,
        "literal_decoded_relative_mse": relative_mse,
        "fp32_decoder_scale_in_mse_audit": True,
        "relative_mse": relative_mse,
        "ideal_entropy_bits": ideal_bits,
        "arithmetic_logical_bits": arithmetic_logical_bits,
        "arithmetic_payload_bytes": len(payload),
        "arithmetic_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "arithmetic_roundtrip_bits_match": arithmetic_bits_match,
        "online_causal_arithmetic_bits_match": online_arithmetic_bits_match,
        "causal_decoder_frequencies_match": causal_frequencies_match,
        "causal_decoder_frozen_bits_match": causal_frozen_bits_match,
        "reconstruction_indices_match": reconstruction_indices_match,
        "tail_escape_count": int(escape_positions.size),
        "tail_escape_record_bits": ESCAPE_RECORD_BITS,
        "tail_escape_payload_bytes": len(escape_payload),
        "tail_escape_payload_sha256": hashlib.sha256(escape_payload).hexdigest(),
        "tail_escape_records_roundtrip": escape_records_roundtrip,
        "tail_escape_padding_is_zero": escape_padding_is_zero,
        "container_header_roundtrip": container_header_roundtrip,
        "base_literal_container_bytes": base_container_bytes,
        "container_cap_bytes": args.container_cap_bytes,
        "passes_container_cap": args.container_cap_bytes <= 0 or len(container) <= args.container_cap_bytes,
        "literal_container_bytes": len(container),
        "literal_container_sha256": hashlib.sha256(container).hexdigest(),
        "framing_bits": framing_bits,
        "total_screen_bits": total_bits,
        "screen_bpw": rate,
        "gaussian_limit_mse_at_screen_rate": gaussian,
        "threshold_mse_0p10db": threshold,
        "gap_db": gap_db,
        "passes_rate_lt_2p5": rate < 2.5,
        "passes_gap_lt_0p10db": gap_db < 0.10,
        "selected_polar_bits": selected,
        "levels": level_rows,
        "_container_hex": container.hex() if args.emit_container_hex else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    # Defaults are the final scale-invariant N=2^18 MAP profile.  Earlier
    # development defaults (N=1024, random decisions, D=0.28, eta=0.5) are
    # intentionally not retained: silently falling back to them produces a
    # different codec and an incomparable finite-length result.
    ap.add_argument("--block-length", type=int, default=1 << 18)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--sigma-source", type=float, default=3.0)
    ap.add_argument("--test-distortion", type=float, default=0.29)
    ap.add_argument("--eta", type=float, default=0.5989929996555583)
    ap.add_argument("--alphabet-size", type=int, default=64)
    ap.add_argument("--decision", choices=("map", "random"), default="map")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--input-bf16", type=Path)
    ap.add_argument("--input-block-start", type=int, default=0)
    ap.add_argument(
        "--container-cap-bytes",
        type=int,
        default=81242,
        help="spend unused bytes below this per-block cap on exact sparse BF16 tail escapes; 0 disables",
    )
    ap.add_argument("--emit-container-hex", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    n = args.block_length
    if n not in (1 << k for k in range(10, 19)):
        raise ValueError("public reliability tables cover powers 2^10 through 2^18")
    if args.container_cap_bytes < 0:
        raise ValueError("container cap must be non-negative")
    sigma_recon = math.sqrt(args.sigma_source**2 - args.test_distortion)
    tilde_sigma = sigma_recon * math.sqrt(args.test_distortion) / args.sigma_source
    levels = int(math.log2(args.alphabet_size))
    capacities = [
        periodic_binary_capacity(tilde_sigma / args.eta / (1 << level0))
        for level0 in range(levels)
    ]
    flags = reliability_freeze_flags(args.polar_repo, n, capacities)

    started = time.perf_counter()
    rows = [run_trial(args, trial, capacities, flags) for trial in range(args.trials)]
    elapsed = time.perf_counter() - started
    mean_rate = float(np.mean([r["screen_bpw"] for r in rows]))
    mean_mse = float(np.mean([r["relative_mse"] for r in rows]))
    aggregate_gap = 10.0 * math.log10(mean_mse / (2.0 ** (-2.0 * mean_rate)))
    result = {
        "architecture": "entropy-coded polar-lattice PTQ with rate-neutral sparse BF16 tail escapes",
        "claim_boundary": (
            "literal entropy-coded two-set realization of the authors' MAP SC simulation; "
            "distinct from the paper's fixed-length F/I/S construction"
        ),
        "strict_ptq": True,
        "source_training_or_retraining": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {
            "block_length": n,
            "trials": args.trials,
            "sigma_source": args.sigma_source,
            "test_channel_distortion": args.test_distortion,
            "eta": args.eta,
            "alphabet_size": args.alphabet_size,
            "decision": args.decision,
            "tilde_sigma": tilde_sigma,
            "capacity_schedule": capacities,
            "seed": args.seed,
            "container_cap_bytes": args.container_cap_bytes,
        },
        "aggregate": {
            "mean_relative_mse": mean_mse,
            "mean_screen_bpw": mean_rate,
            "gaussian_limit_mse": 2.0 ** (-2.0 * mean_rate),
            "threshold_mse_0p10db": (10.0 ** 0.01) * (2.0 ** (-2.0 * mean_rate)),
            "gap_db": aggregate_gap,
            "passes_rate_lt_2p5": mean_rate < 2.5,
            "passes_gap_lt_0p10db": aggregate_gap < 0.10,
        },
        "trials": rows,
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "seconds": elapsed,
    }
    containers: list[bytes] = []
    for row in rows:
        encoded = row.pop("_container_hex")
        if encoded is not None:
            containers.append(bytes.fromhex(encoded))
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.emit_container_hex and args.output:
        bitstream_path = args.output.with_suffix(".polar.bin")
        bitstream_path.write_bytes(b"".join(containers))


if __name__ == "__main__":
    main()
