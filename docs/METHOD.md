# PLTE method

## Design objective

The target is finite-length weight distortion within 0.10 dB of the unit-variance Gaussian rate-distortion reference while keeping the checkpoint-level budget below 2.5 bits per original parameter. Everything is post-training: the encoder sees frozen BF16 weights, and all weight-fitted state needed by a decoder is explicitly transmitted.

## Non-router block codec

Non-router rank-two tensors are partitioned into blocks of `N = 2^18 = 262,144` weights. Each tested block follows this path:

1. Convert immutable BF16 words to floating point and scale the block so its RMS equals `sigma_source = 3`.
2. Encode six nested binary levels over a 64-point one-dimensional lattice alphabet.
3. Use MAP successive cancellation with the published `N=2^18` reliability ordering.
4. Emit every non-frozen decision through a literal 32-bit arithmetic coder. The decoder regenerates each 16-bit causal frequency before consuming the associated bit.
5. Store one FP32 inverse scale.
6. Spend otherwise unused bytes below the 81,242-byte slot on stable largest-squared-error exceptions. Each exception carries an 18-bit absolute coordinate and the exact 16-bit BF16 source word.

Pinned numerical parameters:

| Parameter | Value |
|---|---:|
| Block length | 262,144 |
| Source RMS after scaling | 3.0 |
| Test-channel distortion | 0.29 |
| Lattice scale `eta` | 0.5989929996555583 |
| Alphabet | 64 points / 6 binary levels |
| Decision rule | MAP |
| Frozen-bit seed | 20260831 |
| Slot cap | 81,242 bytes |

The six serialized masks occupy 196,608 bytes and have SHA-256 `11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6`.

## Literal tail-container layout

The first 32-bit little-endian word packs two values:

- low 20 bits: arithmetic payload length in logical bits;
- high 12 bits: exact-tail record count.

It is followed by an FP32 scale, arithmetic payload bytes, and packed 34-bit tail records. Padding bits in the final tail byte must be zero. The current encoder fails closed if the base stream already exceeds the slot cap; it does not implement an overflow codec.

The tail path is a finite-length repair, not a learned residual model. On the heavy-tailed first embedding block, it changes direct-polar relative MSE `0.1175481613` to `0.0322666653` using 564 exact BF16 exceptions without exceeding the slot.

## Router exception

The 48 MoE router matrices have a strongly non-Gaussian distribution and are handled separately. Each of 128 rows carries 16 absolute FP16 centroids and dense little-endian four-bit labels. Centroids are fitted only to that frozen row and are transmitted.

The literal router artifact contains:

- 12,582,912 router weights;
- 6,488,688 bytes;
- 4.1253967285 router-only bpw;
- relative MSE 0.03321895008;
- per-record CRC32 and exact sequential decode audits.

Q2, Q3, and Q4 were evaluated over all 48 routers. Q3 is a Pareto alternative; Q4 is the quality-priority selection within the allowed global budget.

## Rank-one tensors and global accounting

All 210,944 rank-one values remain lossless BF16. The conditional checkpoint budget additionally charges one 64-bit entry per tensor, the six masks, and a 4,096-bit global-format header.

| Component | Bits |
|---|---:|
| 116,422 non-router slots | 75,666,848,992 |
| All-Q4 router container | 51,909,504 |
| Rank-one BF16 values | 3,375,104 |
| 64-bit entries for 18,867 tensors | 1,207,488 |
| Six raw frozen masks | 1,572,864 |
| Global-format header | 4,096 |
| **Total** | **75,724,918,048** |

This arithmetic is exact under the slot assumption. It is not an emitted checkpoint filesize: no concatenated packer currently writes the padding and headers, and untested blocks may overflow.

## Why this is strict PTQ

- The source is the frozen checkpoint only.
- No weights are updated.
- No activation data or task loss is used.
- No QAT, distillation, or calibration optimization is performed.
- Router centroids are fitted to frozen weights and transmitted.
- Decoder probabilities are regenerated causally from pinned codec state.

Development-time selection of parameters and formats is disclosed in the evidence; it is not model retraining.
