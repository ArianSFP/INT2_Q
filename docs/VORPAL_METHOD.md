# VORPAL method and metric contract

## Strict-PTQ boundary

The encoder may inspect only immutable checkpoint weights. It may compute and
transmit statistics, choose a transmitted alphabet, and select transmitted
tail coordinates. It may not update weights or use gradients, an optimizer,
QAT, distillation, activations, prompts, labels, task loss, or Hessian/Fisher
calibration. Every weight-dependent decoder input is charged in the bundle.
Fixed public algorithms and versioned constants are decoder state; fitted
per-panel values are not free side information.

## Geometry and variance description

Each source block contains `N = 2^18` weights and is split into 128 contiguous
groups of `G = 2^11` values. The 400-block panel therefore has 51,200 groups.
For source block `b` and group `g`, the encoder computes the second moment
`v[b,g]` and block second moment `v_block[b]`. It emits one FP32-RNE block RMS
and one signed six-bit relative log-variance label per group:

```text
s_block[b] = FP32_RNE(sqrt(v_block[b]))
c[b,g] = clip(RNE(8 log2(v[b,g] / v_block[b])), -32, 31)
qscale[b,g] = binary64(s_block[b]) * binary64(2^(c[b,g]/16))
qvar[b,g] = qscale[b,g]^2.
```

The binary64 64-entry exponential lookup table is serialized. The literal
`WFPLTE01` side also contains the 400 block scales, packed labels, and 400
binary64 profile pairs plus alphabet codes. Its raw size is 47,356 bytes.

## Stable cross-tensor pooling

Every group receives canonical ordinal `128*b + g`. Encoder and decoder sort
the pair `(qvar, ordinal)` ascending, so equal reconstructed variances have a
normative tie-break. Consecutive sets of 128 sorted groups form 400 polar
chunks of length `2^18`. Groups may cross tensor, role, and layer boundaries.
The normalized source is rounded to BF16-RNE before encoding; decoding applies
the reconstructed scales and inverse stable scatter.

## Reverse-waterfilled polar-lattice core

For sorted chunk `j`, the profile control rate is

```text
r[j] = clip(0.5 log2(mean(qvar[j]) / lambda), 1.45, 3.05)
D_test[j] = 3^2 * 2^(-2r[j])
eta[j] = 1.1306675421666137
         * sqrt((3^2 - D_test[j]) * D_test[j] / 3^2),
```

with frozen binary64 `lambda = 1.8252209629460492e-5`. `D_test` and `eta` are
serialized as binary64 values, avoiding decoder-side transcendental drift.
These are allocation controls, not measured rate or distortion.

Each normalized chunk is coded by a six-level, block-length-`2^18`
polar-lattice source codec with causal arithmetic coding. The routed alphabets
are A64 (six lattice levels) and A128 (a procedural all-open seventh level).
Sparse tail records repair selected normalized BF16 coordinates. Each record
contains an 18-bit coordinate and exact 16-bit BF16 word, for 34 logical bits
before byte padding; coordinates are strictly increasing.

## Completed adaptive scan

The candidate runner reparsed and hash-validated all 400 base reports,
containers, sources, the manifest, encoder, independent decoder, scorer, and
mask. Its trigger universe is all 400 validated base chunks and its predicate
is strict `base_gap_db > 0.10`.

| Item | Count or rule |
|---|---:|
| Base chunks scanned | 400 |
| Triggered chunks | 38 |
| Triggered base alphabet | 38 A64, 0 A128 |
| Tail prefix grid | 1, 3, 7, 15, 30, 60, 120, 240, 480 |
| Tail candidates | 342 |
| A64-to-A128 upgrades | 38 |
| Candidate receipt SHA-256 | `091537886638d5b024aca11b6d00d16330396f4cad845e0604ffa594d1aeeec3` |

Tail ranking uses original-coordinate SSE gain. Candidates are physical,
independently decoded/reparsed objects; the selector still decides whether
their savings justify their exact byte deltas. The hardened schema also
supports tail-only candidates for a triggered A128 base, although no A128 base
crossed the trigger in this completed panel.

## Exact fixed-route selection over the frozen candidate set

The `WFOUTR01` side codec ID 3 represents the alphabet route as

```text
XZ(WFPLTE01 with all 400 alphabet codes canonicalized to A64)
|| 50-byte LSB-first A64/A128 route bitmap.
```

The canonical XZ length is independent of the chosen A64/A128 pattern. This
removes state-dependent side-compression cost from the multiple-choice problem:
the selector Pareto-prunes exact integer container-byte deltas against exact
Decimal raw-SSE savings, then evaluates

```text
(base_SSE - savings) * 2^(16 * bundle_bytes / 104857600)
```

with a deterministic tie-break: objective, total bytes, then lexicographic
option IDs. Source energy is constant and therefore does not affect selection.

The frozen base prelude accounting is:

| Physical component | Bytes |
|---|---:|
| `WFOUTR01` header | 168 |
| Canonical all-A64 XZ member | 33,756 |
| A64/A128 route | 50 |
| BZ2-compressed six-level mask | 1,248 |
| Total prelude | 35,222 |

After the prelude come exactly 400 self-delimiting PLTE frames and physical
EOF. Each frame is `8 + ceil(arithmetic_bits/8) + ceil(34*tails/8)` bytes.
All arithmetic and tail padding must be zero.

## Exact distortion and target

Let `B` be the final physical bundle bytes, `M = 104857600`, `w` the frozen
BF16 values interpreted exactly, and `w_hat` the independent float64
reconstruction in canonical order:

```text
SSE      = sum_i (w_i - w_hat_i)^2
E        = sum_i w_i^2
D        = SSE / E
R        = 8B / M
D_G(R)   = 2^(-2R)
gap_dB   = 10 log10(D / D_G(R))
PASS     = (R < 2.5) and (gap_dB <= -0.10).
```

The selection is globally exact over the frozen enumerated candidate set under
the fixed-route physical objective. It is not a claim of global optimality over
all quantizers, profile families, or sparse-tail allocations.

This is an energy-weighted panel aggregate, not the mean of per-block relative
MSEs. The release publishes per-block, per-role, per-layer, and role-layer
strata beside it.

## Interpretation and provenance

`D_G(R)` is the i.i.d. unit-variance Gaussian squared-error reference. Qwen
weights are finite, structured, heterogeneous, and not assumed Gaussian.
Going below this reference does not beat the unknown rate-distortion function
of the Qwen source and does not violate Shannon theory.

Polar lattices, reverse water-filling, entropy coding, and sparse exceptions
are established ideas. The provisional contribution is their charged,
decoder-complete combination with local log-variance labels, deterministic
cross-tensor pooling, continuous profile assignment, A64/A128 routing, and
exact sparse repairs. No exhaustive patent or prior-art search supports a
universal “first” claim. Relevant primary sources include
[Shannon (1959)](https://ieeexplore.ieee.org/document/5311476),
[Liu, Shi, and Ling (2015)](https://arxiv.org/abs/1501.05683), and
[Liu et al. (2024)](https://arxiv.org/abs/2405.04051).
