# Independent continuous-waterfill outer decoder v1

This ignored development directory closes the oracle dependencies in the
exploratory continuous-waterfilled PLTE path. The final `WFOUTR01` bundle is
self-contained. `outer_decode.py` consumes only:

1. losslessly compressed exact `WFPLTE01` bytes emitted by
   `pack_continuous_side.py`;
2. the losslessly compressed pinned six-level raw reliability mask; and
3. raw self-delimiting PLTE containers concatenated in side-profile order; and
4. the existing clean PLTE decoder implementation.

It does **not** read encoder JSON, the ~13 MB exploratory manifest, normalized
sources, raw Qwen sources, or encoder probability arrays. Membership is
reconstructed as a stable sort of `(qscale**2, canonical_group_ordinal)`, where
`qscale` comes from the literal FP32 block scale, literal six-bit label, and
literal serialized binary64 LUT. Per-chunk test distortion and eta are read as
their exact serialized binary64 values, and the alphabet code is checked.

The v1 codec fixes `sigma_source=3.0`, frozen seed `20260831`, and trial `0` for
each independently encoded chunk. The raw mask supplies levels 1--6; levels 7
and 8 use procedural all-zero freeze flags (fully open), exactly as the encoder.

## Stream framing

The final physically charged bundle is:

```text
WFOUTR01 header | LZMA-XZ(WFPLTE01 side) | BZ2(raw mask) | PLTE chunk 0 | ... | PLTE chunk N-1
```

The 168-byte outer header carries codec IDs, raw and compressed lengths, and
SHA256 hashes of both forms of the side and mask. Both decompressors must reach
exact EOF. There is no container TOC. For each PLTE chunk, the first
little-endian u32 has
`logical_bits = word & ((1<<20)-1)` and `escape_count = word >> 20`. Its exact
size is `8 + ceil(logical_bits/8) + ceil(34*escape_count/8)` bytes. The decoder
checks arithmetic padding, sparse-tail padding, monotone/in-range escapes, all
declared chunks, and exact EOF.

Assemble the final bundle without reading encoder JSON or the manifest:

```bash
python pack_bundle.py \
  --side side.bin \
  --container-dir full_jobs \
  --raw-mask frozen_flags_6x262144.raw \
  --output panel.wfouter \
  --receipt pack.receipt.json
```

Decode the self-contained form (the mask is deliberately not a CLI input):

```bash
python outer_decode.py \
  --bundle panel.wfouter \
  --clean-decoder ../../plte/agent_polar_codec_audit_independent_decoder.py \
  --reconstruction canonical.f64 \
  --receipt decode.receipt.json \
  --workers 8
```

Separate raw `--side/--containers/--raw-mask` inputs remain available only as
a clearly receipted development mode; they are not the final rate claim. The
default output is little-endian float64 in canonical
`[block, group, value]` order, preserving the exact decoder arithmetic before
source scoring. `--reconstruction-dtype f32` is available only as an explicitly
receipted storage tradeoff.

## Separate exact-source evaluation

`evaluate_sources.py` is deliberately separate. It requires CuPy and a compact
evaluation-only ledger, never the exploratory membership manifest:

```json
{
  "format": "canonical BF16 source ledger v1",
  "blocks": [
    {
      "canonical_block_ordinal": 0,
      "id": "the exact frozen block id",
      "tensor": "the exact tensor name",
      "role": "the frozen tensor role",
      "layer": 0,
      "path": "relative/or/absolute.bf16.bin",
      "sha256": "64 lowercase hex digits"
    }
  ]
}
```

Use JSON `null` for the layer of global embedding/head tensors. Every one of
the 400 ordinals, IDs, roles, layers, byte counts, and source SHA256 values is
checked before scoring. The evaluator also re-parses and re-hashes the
reconstruction and physical encoded bundle. It emits the primary aggregate,
all 400 original block rows, role/layer/role-layer aggregates, and explicitly
diagnostic mixed-chunk rows. Its pass criterion is the new target,
`Gaussian-reference gap <= -0.10 dB`, together with strict physical all-in rate
`< 2.5 bpw`.

```bash
python evaluate_sources.py \
  --decode-receipt decode.receipt.json \
  --reconstruction canonical.f64 \
  --source-ledger sources.json \
  --expected-source-ledger-sha256 <published-ledger-sha256> \
  --source-root /root/int2/INT2_Q_stratified_run \
  --bundle bundle.bin \
  --output score.receipt.json
```

Claim evaluation deliberately accepts only the physical `--bundle` form. It
rejects the decoder's separate-file development mode, recomputes panel geometry
from the reparsed literal side, and pins both the clean decoder and immutable
Qwen checkpoint before calculating rate or distortion.

The decoder receipt binds the raw and compressed side, raw and compressed mask,
physical bundle, container stream, clean decoder, decoder script, stable
membership, every container, and canonical reconstruction with SHA256 hashes.
Only physical `WFOUTR01` bytes are charged; reconstruction and source ledger are
evaluation artifacts, not encoded side information.
