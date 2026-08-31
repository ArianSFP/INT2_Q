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
| Original Tier-0 cap `T0` | 81,242 bytes |

The six serialized masks occupy 196,608 bytes and have SHA-256 `11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6`.

## Literal tail-container layout

The first 32-bit little-endian word packs two values:

- low 20 bits: arithmetic payload length in logical bits;
- high 12 bits: exact-tail record count.

It is followed by an FP32 scale, arithmetic payload bytes, and packed 34-bit tail records. Padding bits in the final arithmetic and tail bytes must be zero. At the original Tier-0 cap, the encoder fails closed if the arithmetic base stream alone exceeds 81,242 bytes.

The tail path is a finite-length repair, not a learned residual model. On the heavy-tailed first embedding block, it changes direct-polar relative MSE `0.1175481613` to `0.0322666653` using 564 exact BF16 exceptions without exceeding the slot.

## Deterministic post-hoc rate reservoir

The 400-block broad evaluation falsified the universal Tier-0 fit assumption: 385 blocks fit `T0`, while 15 base streams exceeded it by 2--36 bytes. The original endpoint therefore failed and remains recorded as failed. It was not repaired retroactively.

For a separate engineering evaluation, the codec uses deterministic reservoir tiers:

```text
T0 = 81,242 bytes
Tk = T0 + 64 k bytes
k  = max(0, ceil((base_container_bytes - T0) / 64))
```

Only the exact base-container-overflow exception may select `k > 0`. Quality, MSE, layer, and tensor role do not participate in tier selection. Successful Tier-0 artifacts remain byte-for-byte unchanged. A retry must reproduce the original base length and changes only `container_cap_bytes`; the weights, polar parameters, coset, masks, and decoder remain fixed. All 15 observed overflows selected Tier 1.

Each non-router block is charged four bits for its tier-map entry. For a block assigned tier `k`, the all-in charged rate and quality gap are

```text
R_k   = (8 T_k + 4) / 262144
gap_k = 10 log10(D_k / 2^(-2 R_k)).
```

The published panel map follows the ID-sorted `results.json` order with the first result in the low nibble. The conditional full-checkpoint design instead requires a four-bit map in canonical non-router block order. The panel artifact validates map decoding, header-derived literal-prefix extraction, container hashes, zero bit/byte padding, contiguous offsets, and exact end of file.

The conditional checkpoint rate is

```text
(75,724,918,048 + 4*116,422 + 512*sum(k_i)) / 30,532,122,624 bpw.
```

With no tier increments this is `2.4801873315049345 bpw`. The strict condition for remaining below 2.5 bpw is `sum(k_i) <= 1,181,489`; even the conditional all-Tier-10 inventory is `2.499710397337621 bpw`. These are exact inventory calculations, not a measured checkpoint tier distribution. The 400-block panel has `sum(k_i) = 15`, which cannot be substituted for the unencoded checkpoint total.

On the amended 400-block panel, every all-in charged gap is below 0.10 dB. The maximum is `0.08667984346279214 dB`; the energy-weighted aggregate relative MSE is `0.03271539785114697`, and its aggregate all-in charged gap is `0.07498293435240821 dB`. This is post-hoc engineering evidence, not an untouched confirmatory result.

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

The base arithmetic above is exact for uniform Tier-0 inventory. The reservoir formula adds the full tier map and every 64-byte increment. A literal map and padded slot image were emitted and read back for the 400-block panel, but no complete checkpoint has been encoded or packed.

## Measured coverage

The broad panel adds 400 previously untested PLTE blocks spanning all 48 layers, seven layer-specific rank-two roles, 32 embedding blocks, and 32 LM-head blocks. Together with the earlier 47 unique blocks, measured PLTE coverage is `447 / 116,422 = 0.383948%` of non-router full blocks. The remaining 115,975 blocks are unencoded. The router and rank-one exception paths are complete censuses, but that does not convert sampled PLTE distortion into a whole-checkpoint measurement.

## Why this is strict PTQ

- The source is the frozen checkpoint only.
- No weights are updated.
- No activation data or task loss is used.
- No QAT, distillation, or calibration optimization is performed.
- Router centroids are fitted to frozen weights and transmitted.
- Decoder probabilities are regenerated causally from pinned codec state.

Development-time selection of parameters and formats is disclosed in the evidence; it is not model retraining.

The reservoir remains strict PTQ for the same reason: its tier is a deterministic function of the frozen block's decoder-visible base-stream length, and every extra tail word is transmitted. However, the tier design was introduced after observing the 15 original cap failures, so the amended result is explicitly non-confirmatory.
