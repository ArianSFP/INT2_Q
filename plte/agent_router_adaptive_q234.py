#!/usr/bin/env python3
"""Strict-PTQ adaptive router codec: per-row Lloyd Q2 -> Q3 -> Q4 -> BF16.

The encoder tries the lowest-rate literal codec first.  A quantized candidate
is accepted only after decoding its serialized FP16 absolute centroids and
packed labels and measuring original-domain SSE.  Routers that miss the
configured relative-SSE threshold are stored losslessly as their original
BF16 bytes.

Container (all little endian):
  global: <8sHHHHdHH20s  (48 bytes)
  record: <HBBII          (12 bytes), then payload

For q in {2,3,4}, payload is ROWS*(2**q) FP16 centroids followed by a dense
little-endian q-bit label stream.  Tag 16 stores the original BF16 payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

import cupy as cp
import numpy as np


ROWS = 128
COLS = 2048
VALUES = ROWS * COLS
LAYERS = 48
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MAGIC = b"RQA2V001"
VERSION = 1
GLOBAL_FORMAT = "<8sHHHHdHH20s"
RECORD_FORMAT = "<HBBII"
GLOBAL_BYTES = struct.calcsize(GLOBAL_FORMAT)
RECORD_BYTES = struct.calcsize(RECORD_FORMAT)
FLAG_ABSOLUTE_FP16_ROW_CODEBOOK = 1


def bf16_to_float32(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != VALUES:
        raise ValueError((words.size, VALUES))
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(ROWS, COLS)


def fit_literal_candidate(
    weights: cp.ndarray, q: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    k = 1 << q
    quantiles = cp.asarray((np.arange(k, dtype=np.float64) + 0.5) / k)
    centroids = cp.quantile(weights, quantiles, axis=1).T.astype(cp.float32)
    for _ in range(iterations):
        labels = cp.argmin(
            cp.square(weights[:, :, None] - centroids[:, None, :]), axis=2
        )
        updated = centroids.copy()
        for code in range(k):
            mask = labels == code
            count = mask.sum(axis=1)
            total = cp.where(mask, weights, cp.float32(0)).sum(axis=1)
            updated[:, code] = cp.where(
                count > 0, total / cp.maximum(count, 1), centroids[:, code]
            )
        centroids = updated

    # The decoder sees these exact FP16 values, so assignments must be redone.
    literal = centroids.astype(cp.float16)
    labels = cp.argmin(
        cp.square(weights[:, :, None] - literal.astype(cp.float32)[:, None, :]),
        axis=2,
    ).astype(cp.uint8)
    return (
        cp.asnumpy(literal).astype("<f2", copy=False),
        cp.asnumpy(labels).astype(np.uint8, copy=False),
    )


def pack_labels(labels: np.ndarray, q: int) -> bytes:
    flat = labels.reshape(-1).astype(np.uint64, copy=False)
    if np.any(flat >= (1 << q)):
        raise ValueError("label outside q-bit alphabet")
    bit_count = flat.size * q
    output = np.zeros((bit_count + 7) // 8, dtype=np.uint8)
    bit_offsets = np.arange(flat.size, dtype=np.uint64) * np.uint64(q)
    byte_offsets = (bit_offsets >> np.uint64(3)).astype(np.int64)
    shifts = bit_offsets & np.uint64(7)
    words = flat << shifts
    np.bitwise_or.at(output, byte_offsets, (words & np.uint64(255)).astype(np.uint8))
    spill = shifts + np.uint64(q) > np.uint64(8)
    if np.any(spill):
        np.bitwise_or.at(
            output,
            byte_offsets[spill] + 1,
            (words[spill] >> np.uint64(8)).astype(np.uint8),
        )
    return output.tobytes()


def unpack_labels(payload: bytes, q: int, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    bit_offsets = np.arange(count, dtype=np.uint64) * np.uint64(q)
    byte_offsets = (bit_offsets >> np.uint64(3)).astype(np.int64)
    shifts = bit_offsets & np.uint64(7)
    low = packed[byte_offsets].astype(np.uint16)
    high = np.zeros(count, dtype=np.uint16)
    has_high = byte_offsets + 1 < packed.size
    high[has_high] = packed[byte_offsets[has_high] + 1].astype(np.uint16)
    words = low | (high << np.uint16(8))
    return ((words >> shifts.astype(np.uint16)) & np.uint16((1 << q) - 1)).astype(
        np.uint8
    )


def quantized_payload(centroids: np.ndarray, labels: np.ndarray, q: int) -> bytes:
    if centroids.shape != (ROWS, 1 << q) or labels.shape != (ROWS, COLS):
        raise ValueError((centroids.shape, labels.shape, q))
    return centroids.astype("<f2", copy=False).tobytes() + pack_labels(labels, q)


def decode_payload(payload: bytes, tag: int) -> tuple[np.ndarray, np.ndarray | None]:
    if tag == 16:
        return bf16_to_float32(payload), None
    if tag not in (2, 3, 4):
        raise ValueError(tag)
    k = 1 << tag
    centroid_bytes = ROWS * k * 2
    expected_label_bytes = (VALUES * tag + 7) // 8
    if len(payload) != centroid_bytes + expected_label_bytes:
        raise ValueError((len(payload), centroid_bytes + expected_label_bytes))
    centroids = np.frombuffer(payload, dtype="<f2", count=ROWS * k).reshape(ROWS, k)
    labels = unpack_labels(payload[centroid_bytes:], tag, VALUES).reshape(ROWS, COLS)
    reconstructed = centroids.astype(np.float32)[np.arange(ROWS)[:, None], labels]
    return reconstructed, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-bin", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.0327167447673)
    args = parser.parse_args()

    started = time.perf_counter()
    global_header = struct.pack(
        GLOBAL_FORMAT,
        MAGIC,
        VERSION,
        LAYERS,
        ROWS,
        COLS,
        args.threshold,
        args.iterations,
        0,
        bytes.fromhex(REVISION),
    )
    if len(global_header) != GLOBAL_BYTES:
        raise AssertionError(len(global_header))

    records: list[bytes] = []
    rows: list[dict[str, object]] = []
    aggregate_energy = 0.0
    aggregate_sse = 0.0

    for layer in range(LAYERS):
        name = f"model.layers.{layer}.mlp.gate.weight.block0.bf16.bin"
        path = args.input_dir / name
        raw = path.read_bytes()
        weights_np = bf16_to_float32(raw)
        weights = cp.asarray(weights_np)
        energy = float(np.sum(np.square(weights_np, dtype=np.float64), dtype=np.float64))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError((path, energy))

        attempts: list[dict[str, object]] = []
        selected_tag = 16
        selected_payload = raw
        selected_sse = 0.0
        selected_relative_mse = 0.0
        selected_labels: np.ndarray | None = None

        for q in (2, 3, 4):
            centroids, labels = fit_literal_candidate(weights, q, args.iterations)
            payload = quantized_payload(centroids, labels, q)
            reconstructed, decoded_labels = decode_payload(payload, q)
            sse = float(
                np.sum(np.square(weights_np - reconstructed, dtype=np.float64), dtype=np.float64)
            )
            relative_mse = sse / energy
            labels_match = bool(np.array_equal(labels, decoded_labels))
            attempts.append(
                {
                    "q": q,
                    "payload_bytes": len(payload),
                    "sse": sse,
                    "relative_mse": relative_mse,
                    "labels_roundtrip": labels_match,
                }
            )
            if not labels_match:
                raise AssertionError((layer, q, "label roundtrip"))
            if relative_mse <= args.threshold:
                selected_tag = q
                selected_payload = payload
                selected_sse = sse
                selected_relative_mse = relative_mse
                selected_labels = labels
                break

        reconstructed, decoded_labels = decode_payload(selected_payload, selected_tag)
        decoded_sse = float(
            np.sum(np.square(weights_np - reconstructed, dtype=np.float64), dtype=np.float64)
        )
        if selected_tag == 16:
            selected_sse = decoded_sse
            selected_relative_mse = decoded_sse / energy
        elif not np.array_equal(selected_labels, decoded_labels):
            raise AssertionError((layer, selected_tag, "selected labels"))
        if decoded_sse != selected_sse:
            raise AssertionError((layer, decoded_sse, selected_sse))

        crc = zlib.crc32(selected_payload) & 0xFFFFFFFF
        record_header = struct.pack(
            RECORD_FORMAT,
            layer,
            selected_tag,
            FLAG_ABSOLUTE_FP16_ROW_CODEBOOK if selected_tag in (2, 3, 4) else 0,
            len(selected_payload),
            crc,
        )
        records.append(record_header + selected_payload)
        aggregate_energy += energy
        aggregate_sse += selected_sse
        rows.append(
            {
                "layer": layer,
                "tensor": f"model.layers.{layer}.mlp.gate.weight",
                "input": str(path),
                "input_bytes": len(raw),
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "energy": energy,
                "rms": (energy / VALUES) ** 0.5,
                "attempts": attempts,
                "selected_tag": selected_tag,
                "selected_payload_bytes": len(selected_payload),
                "selected_record_bytes": len(record_header) + len(selected_payload),
                "selected_sse": selected_sse,
                "selected_relative_mse": selected_relative_mse,
                "payload_crc32": f"{crc:08x}",
                "inverse_decode_exact": bool(decoded_sse == selected_sse),
            }
        )
        print(
            json.dumps(
                {
                    "layer": layer,
                    "tag": selected_tag,
                    "relative_mse": selected_relative_mse,
                    "record_bytes": len(record_header) + len(selected_payload),
                }
            ),
            flush=True,
        )

    container = global_header + b"".join(records)
    args.output_bin.write_bytes(container)

    # Independent sequential framing/CRC/inverse audit of the literal file.
    literal = args.output_bin.read_bytes()
    header = struct.unpack(GLOBAL_FORMAT, literal[:GLOBAL_BYTES])
    if header[0] != MAGIC or header[1:5] != (VERSION, LAYERS, ROWS, COLS):
        raise AssertionError(header[:5])
    cursor = GLOBAL_BYTES
    decoded_energy = 0.0
    decoded_sse = 0.0
    for layer, row in enumerate(rows):
        record = struct.unpack(RECORD_FORMAT, literal[cursor : cursor + RECORD_BYTES])
        cursor += RECORD_BYTES
        layer_id, tag, flags, payload_bytes, crc = record
        payload = literal[cursor : cursor + payload_bytes]
        cursor += payload_bytes
        if layer_id != layer or zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise AssertionError((layer, record))
        if tag in (2, 3, 4) and flags != FLAG_ABSOLUTE_FP16_ROW_CODEBOOK:
            raise AssertionError((layer, flags))
        source = bf16_to_float32(Path(row["input"]).read_bytes())
        reconstruction, _ = decode_payload(payload, tag)
        energy = float(np.sum(np.square(source, dtype=np.float64), dtype=np.float64))
        sse = float(
            np.sum(np.square(source - reconstruction, dtype=np.float64), dtype=np.float64)
        )
        decoded_energy += energy
        decoded_sse += sse
    if cursor != len(literal):
        raise AssertionError((cursor, len(literal)))

    tag_counts = {str(tag): sum(row["selected_tag"] == tag for row in rows) for tag in (2, 3, 4, 16)}
    result = {
        "architecture": "adaptive per-row Lloyd Q2 -> Q3 -> Q4 -> lossless BF16 router exception",
        "strict_ptq": True,
        "model_training_or_retraining": False,
        "repo": "Qwen/Qwen3-30B-A3B",
        "revision": REVISION,
        "shape_each": [ROWS, COLS],
        "router_count": LAYERS,
        "router_values": LAYERS * VALUES,
        "promotion_threshold_relative_sse": args.threshold,
        "iterations": args.iterations,
        "format": {
            "global_format": GLOBAL_FORMAT,
            "global_bytes": GLOBAL_BYTES,
            "record_format": RECORD_FORMAT,
            "record_header_bytes": RECORD_BYTES,
            "quantized_payload": "absolute row-major FP16 centroids, then dense little-endian q-bit labels",
            "bf16_tag": 16,
        },
        "selected_tag_counts": tag_counts,
        "aggregate": {
            "source_energy": aggregate_energy,
            "sse": aggregate_sse,
            "relative_mse": aggregate_sse / aggregate_energy,
            "container_bytes": len(container),
            "container_bits": len(container) * 8,
            "router_bpw": len(container) * 8 / (LAYERS * VALUES),
            "container_sha256": hashlib.sha256(container).hexdigest(),
        },
        "independent_literal_decode": {
            "bytes_consumed": cursor,
            "exact_file_length": cursor == len(literal),
            "source_energy": decoded_energy,
            "sse": decoded_sse,
            "relative_mse": decoded_sse / decoded_energy,
            "aggregate_energy_match": decoded_energy == aggregate_energy,
            "aggregate_sse_match": decoded_sse == aggregate_sse,
        },
        "routers": rows,
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "seconds": time.perf_counter() - started,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["aggregate"], "tag_counts": tag_counts}), flush=True)


if __name__ == "__main__":
    main()
