# VORPAL role-joint procedural residual extension

## Result

The frozen Qwen development artifact puts the expert-MLP up-projection,
down-projection, and gate-projection role aggregates individually below the
declared i.i.d. unit-variance Gaussian reference. It remains strict PTQ and
physically below 2.5 bits per panel weight.

| Independently replayed quantity | Up | Down | Gate |
|---|---:|---:|---:|
| Blocks | 48 | 48 | 48 |
| Values | 12,582,912 | 12,582,912 | 12,582,912 |
| Source energy | 7,126.617850162894 | 7,344.938468006613 | 6,911.452696096183 |
| VORPAL-base SSE | 233.83067163158927 | 233.72836074725691 | 231.7005451408534 |
| Corrected SSE | 222.61419508188922 | 229.43461240212838 | 215.91720395036748 |
| Corrected relative MSE | 0.031237004672111178 | 0.031237104762893406 | 0.031240495080335957 |
| Coordinate pulses | 100 | 100 | 100 |
| Refinement stages | 269 | 100 | 387 |
| Signed gap at the actual wrapper rate | **-0.0018119033858966 dB** | **-0.0017979875814763 dB** | **-0.0013266518828134 dB** |

The exact physical endpoint is:

| Quantity | Value |
|---|---:|
| Panel values, `M` | 104,857,600 |
| Preserved VORPAL base | 32,583,835 bytes |
| Provenance-bound wrapper header | 224 bytes |
| Residual extension | 183,929 bytes |
| Complete wrapper, `B` | 32,767,988 bytes |
| Exact rate, `R = 8B/M` | 2.49999908447265625 bpw |
| Distance to the 2.5-bpw equality point | 96 bits |
| Additional whole bytes allowed while remaining strictly below 2.5 | 11 bytes |
| Gaussian reference, `2^(-2R)` | 0.031250039662224983 |
| Corrected full-panel SSE | 1,919.2523903771403 |
| Corrected full-panel relative MSE | 0.02935406047291881 |
| Corrected full-panel gap | **-0.2718238828655344 dB** |

The up/down/gate goal is a strict signed `< 0 dB` predicate. The worst of the
three independently measured gaps is `-0.0013266518828134 dB`, so the artifact
passes. The small margin is reported explicitly; it should not be interpreted
as a robustness guarantee outside these exact bytes and this exact panel.

## Strict-PTQ boundary

The encoder reads immutable checkpoint weights and the already decoded VORPAL
reconstruction. It may select and transmit residual atoms, but it does not
train or modify the source model. The experiment uses no optimizer, gradient,
QAT, distillation, activations, prompts, labels, Hessian or Fisher calibration,
task loss, or calibration corpus.

Every weight-dependent quantity needed by the decoder is physically present:
the original VORPAL base, role masks, coordinate supports and signs, FP16
amplitudes, and every refinement symbol. The four signed transform bases are
fixed public algorithms generated from versioned integer constants. They are
not fitted codebooks or uncharged model state.

## Residual architecture

Let `x` be a role's frozen weights and `x_hat_0` its VORPAL reconstruction. The
encoder starts from `r_0 = x - x_hat_0`. Each role comprises 48 blocks of
262,144 values, or 12,582,912 values. It is viewed as 96 transform groups of
`N = 131,072 = 2^17` values.

### Coordinate pulse prefix

The encoder selects the 100 largest-magnitude residual coordinates in a role,
uses one shared FP16 amplitude, and transmits sorted coordinate gaps with a
canonical Rice code plus one sign bit per coordinate. The emitted Rice
parameters are `b = 16` for all three roles. Exact payload sizes are 245 bytes
for up and 244 bytes each for down and gate. The independent decoder checks
support order, bounds, meaningful bit count, zero padding, and that the Rice
parameter minimizes the exact emitted length over `b = 0..20`.

### Procedural multi-basis refinement

For refinement stage `s`, role `q`, group `g`, and bank `b in {0,1,2,3}`, the
decoder constructs a deterministic diagonal sign vector `D(q,s,b,g)` with a
fixed 32-bit integer mixing function. Multiplying it by the normalized
Walsh-Hadamard matrix yields a signed orthonormal basis. For every group, the
encoder chooses the bank and coefficient whose correlation with the current
residual has largest magnitude:

```text
(b*, k*) = argmax_(b,k) |[H D(q,s,b,g) r_s,g]_k|.
```

One code per group stores:

```text
2-bit bank | 17-bit Hadamard index | 1-bit coefficient sign.
```

The 96 group codes therefore occupy exactly 240 bytes. One positive finite
IEEE-binary16 amplitude is shared across the role and stage, making every
stage exactly 242 bytes. The decoder regenerates the chosen atoms and adds
their signed, scaled values to the reconstruction. Four fresh procedural
bases are addressed at every stage through the role/stage seed; no basis
matrix is serialized.

The physical budget permits 756 stages. The encoder evaluates the exact
prefix gain curves and chooses the integer allocation that minimizes the worst
role gap at the final common physical rate: 269 stages for up, 100 for down,
and 387 for gate.

This is a sparse-regression/successive-refinement residual code layered on
VORPAL. It is not a claim that sparse regression codes, Hadamard transforms,
matching pursuit, Rice codes, or residual quantization are individually new.
The research contribution under test is this charged, decoder-complete,
role-balanced composition for frozen neural weights.

## Physical format and provenance

The normative `VJWRAP42` file is exact concatenation:

```text
224-byte wrapper header | original VORPAL base | VJSPRC41 extension | EOF
```

The wrapper header contains lengths, panel geometry, a header CRC32, and six
SHA-256 bindings:

1. embedded VORPAL base;
2. residual extension;
3. selected 400-block manifest;
4. exact-source VORPAL evaluation;
5. normative VORPAL reconstruction; and
6. the strict-PTQ encoder implementation.

The extension contains a versioned header, three role descriptors, three
400-bit role masks, the three Rice streams, 756 fixed-size stage records, and
an extension CRC32. Parsers require exact section lengths, canonical constants,
disjoint 48-block masks, zero padding, and physical EOF.

The extension's fixed non-stage portion is 977 bytes and its stage records are
`756 * 242 = 182,952` bytes, totaling 183,929 bytes. The complete wrapper—not
the shorter extension or nominal symbol count—is the rate denominator.

## Independent verification

The role-extension exact-source verifier is separately implemented and does
not import the encoder. Before opening any source weight it:

- checks all six wrapper bindings, both CRCs, section lengths, hashes, and EOF;
- verifies that the embedded base equals the published VORPAL bundle byte for
  byte;
- parses all role masks, Rice streams, amplitudes, and 756 stage records; and
- applies every transmitted correction to the externally supplied normative
  base reconstruction after verifying its wrapper-bound SHA-256.

That invocation does not rerun the polar decoder for the embedded base. The
base publication separately provides and audits the independent VORPAL decode,
and the wrapper requires the embedded bytes to be exactly that published base.
Decoder completeness is therefore established by composition of the VORPAL
base decoder and this independent residual decoder; the exact-source role
receipt specifically audits the second step and the binding between them.

It then verifies every one of the 144 selected BF16 source hashes and computes
source energy and SSE directly in Float64 with CuPy 14.2.0 on an NVIDIA RTX
A6000. The corrected three-role reconstruction SHA-256 is
`4a4d7d72881f608088e20a382ada4d6a029a8fae3ba1b435d94c5bdb3b3e0f06`.

Separate role auditors replay the actual matching-pursuit choices. The gate
audit, for example, recomputes all `387 * 96` emitted bank/index/sign choices
against all four procedural bases, obtains zero mismatches, and exercises nine
tamper cases. The publication's standard-library source-free verifier checks
the physical wrapper, canonical extension, published manifests and receipts,
rate/gap arithmetic, and fail-closed mutations without requiring Qwen weights,
the 800 MiB reconstruction, CuPy, or a GPU. A source-free pass verifies the
published evidence chain; only the exact-source replay remeasures distortion.

Run the source-free check from the repository root:

```bash
python vorpal/role_joint_sparc4/verify_source_free.py
```

The full exact-source command and immutable input paths are recorded in the
[artifact README](../evaluation/qwen3_vorpal_role_joint_v1/README.md). It
requires the pinned Qwen BF16 blocks and normative VORPAL reconstruction, which
are deliberately omitted from Git.

## Metric interpretation and claim boundary

For each role, the reported quantity is:

```text
D_role       = sum_(i in role) (x_i - x_hat_i)^2 / sum_(i in role) x_i^2
D_G(R)       = 2^(-2R)
gap_role_dB  = 10 log10(D_role / D_G(R)).
```

The same physical wrapper rate `R` is used for all three role comparisons.
The negative values mean those finite Qwen role aggregates have lower measured
relative distortion than the declared i.i.d. unit-variance Gaussian reference
at that rate. The Gaussian curve is not a universal lower bound for a finite,
heterogeneous neural-weight source, so this is not a Shannon-bound violation.

The panel contains one selected expert up, down, and gate block in every layer
0 through 47. It does not contain every expert matrix in the full checkpoint.
Results are energy-weighted role aggregates, not guarantees for each of the 144
individual blocks. The base method, residual method, and stage allocation were
developed using this same frozen panel; this is post-hoc development evidence,
not a disjoint confirmatory result. No perplexity, downstream accuracy,
inference kernel, end-to-end checkpoint, or full-checkpoint physical rate is
established here.

The next defensible milestones are to freeze this format, test an untouched
role/layer panel, encode multiple experts per layer, and measure model-level
quality with a decoder-integrated inference path.
