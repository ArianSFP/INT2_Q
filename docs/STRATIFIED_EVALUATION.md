# Qwen3 broad-coverage evaluation protocol

This protocol extends the preliminary 47-block evidence with a frozen,
weight-blind selection of 400 previously untested PLTE blocks. Its purpose is
to expose tensor-role, layer, expert, and flat-position transfer failures. It
does not turn a sample into a whole-checkpoint result.

## Checkpoint inventory

The 16 published safetensors headers describe 18,867 tensors and 15 normalized
roles in `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

| Architecture path | Roles | Coverage treatment |
|---|---:|---|
| Attention matrices | q, k, v, o | One new PLTE block per role in every layer |
| Expert matrices | gate, up, down | One new expert/block pair per role in every layer |
| Vocabulary matrices | embedding, LM head | 32 new header-stratified PLTE blocks each |
| MoE routers | gate | Existing literal Q4 census of all 48 layers |
| Rank-one tensors | input/post-attention norms, q/k norms, final norm | Exact BF16 census of all 193 tensors |

The rank-one tensors total 210,944 values. None reaches the fixed PLTE block
length of 262,144 values, so padding them would produce a misleading PLTE
measurement. The architecture retains them losslessly in BF16. Their sources
and literal decodes are therefore hashed and compared exactly.

The non-router rank-two inventory contains 116,422 full blocks. The 400 new
blocks are 0.34358% of that population; the prior and new sets together contain
447 unique PLTE blocks, or 0.38395%. In particular, one sampled expert tensor
per layer and role is not a census of the 18,432 expert tensors.

## Frozen selection

Run:

```bash
python tools/build_stratified_manifest.py \
  --output evaluation/qwen3_stratified_v1/manifest.json
```

The builder reads only the pinned tensor headers and the canonical identifiers
in the published evidence manifest. It does not fetch or inspect selected
weight payloads. A SHA-256 seed is derived from:

```text
PLTE-QWEN3-30B-A3B-COVERAGE-V1
NUL
ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
NUL
5d2a29cb60f2e068ac0a49cc33e04f51f515720e
```

The resulting selection-seed SHA-256 is
`7597aba80b3f5365f5d072b0fdd57b0c7ee8e6bd6ed88b0aaa5bc364f29b2768`.

Within each header-defined stratum, a SHA-256-derived starting point is used
and canonical candidates are traversed cyclically. The 47 previously tested
canonical blocks are skipped by identifier. This prevents duplicate work; no
source values or prior distortion values participate in selection.

The manifest must be committed before any selected source is fetched. Failed
quality results are never replaced. Retrying the same canonical block after an
infrastructure interruption is allowed and must retain the same source hash.

## Exact codec boundary

Every PLTE row uses the already-published encoder bytes with SHA-256
`4d76ba53c88710778085917108b7940517ed14565815fc8437ea4919d7df4bf8`:

```text
N                    262144
sigma source         3.0
test-channel D       0.29
eta                  0.5989929996555583
alphabet             64
decision             MAP
coset seed           20260831
container cap        81242 bytes
```

The encoder is CuPy-backed. On the current RunPod, eight isolated encoder
processes match the 7.65-core CPU quota and outperform both one and sixteen
processes. All numerical and codec parameters remain identical across blocks;
changing the coset per source would change the codec and introduce side
information.

The independent decoder SHA-256 is
`7589f4be6e784d8e5a0067303da389b6d982430eb84fda52f668808f322c25d9`.
It receives the serialized container, public JSON parameters, and the frozen
profile, but no encoder probability sequence. Every new block is scheduled for
this clean decode, not merely the encoder's internal round trip.

## Metrics and acceptance

For block `i`, using the literal FP32 scale serialized in its container:

```text
E_i = sum(w_i^2)
S_i = sum((w_i - decoded_i)^2)
D_i = S_i / E_i
R_i = 8 * literal_container_bytes_i / 262144
gap_i = 10 log10(D_i / 2^(-2 R_i))
```

The primary coverage endpoint uses the fixed PLTE slot rate
`R_slot = 2.47930908203125 bpw`, not any smaller realized file length. Its
Gaussian reference is `0.0321593450611333`. A block fails the quality endpoint
when its fixed-slot gap is greater than or equal to 0.10 dB.

Aggregate distortion is energy weighted:

```text
D_aggregate = sum_i S_i / sum_i E_i
```

Mean block-relative MSE and mean dB gap are descriptive only and must not be
substituted for this ratio. Summaries report the maximum, p95, p99, every
failure, and separate role/layer aggregates. The sample has one draw in each
layer/role stratum, so it does not support a within-stratum sampling-variance
estimate or a checkpoint-wide maximum claim.

A hard integrity failure is any of:

- source, checkpoint, encoder, decoder, profile, or container hash mismatch;
- non-finite or zero source energy;
- base stream or final container exceeding 81,242 bytes;
- arithmetic, causal-frequency, frozen-bit, reconstruction-index, tail-record,
  padding, or header round-trip mismatch;
- an independent decoder that does not consume the exact file or reproduce the
  encoder's literal MSE to absolute tolerance `1e-12`.

Hard failures stop the run. A distortion failure is retained and disclosed;
it does not silently remove the block. Only a clearly systemic pilot collapse
may stop the remaining breadth run early.

## Claim boundary

If all 400 rows pass, the valid statement is that all 400 preregistered new
blocks passed, with complete layer-by-role coverage in this panel. It is not a
statement that every Qwen block passes. Routers remain measured Q4 exceptions,
rank-one tensors remain exact BF16 exceptions, checkpoint rate remains a
conditional inventory calculation, and any full-model distortion remains a
sample-based projection until every source block is encoded.
