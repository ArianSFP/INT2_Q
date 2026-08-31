# Qwen3 broad-coverage evaluation protocol

This protocol extends the preliminary 47-block evidence with a frozen,
weight-blind selection of 400 previously untested PLTE blocks. Its purpose is
to expose tensor-role, layer, expert, and flat-position transfer failures. It
does not turn a sample into a whole-checkpoint result.

All 400 blocks have now been evaluated. The original universal Tier-0 endpoint
failed on 15 cap overflows. A post-hoc deterministic reservoir assigns those 15
blocks to Tier 1 and the other 385 to Tier 0; under that amended definition,
all 400 pass clean independent decoding and the charged 0.10 dB quality bound.
The two outcomes are reported separately below.

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

On the documented RunPod layout, the resumable end-to-end command is:

```bash
/root/int2-venv/bin/python tools/run_stratified_evaluation.py \
  --python /root/int2-venv/bin/python \
  --polar-repo /root/PolarLatticeQuantization \
  --fetch-workers 16 --encode-workers 8 --decode-workers 8
```

Raw BF16 ranges and per-job logs remain under the gitignored
`tmp/qwen3_stratified_v1/` workspace. Finalization emits a source-free hash
manifest, the full encoder metadata, every clean-decoder audit, a concatenated
container bundle with an offset table, the rank-one census, and summary JSON
under `evaluation/qwen3_stratified_v1/`.

## Metrics and acceptance

For block `i`, using the literal FP32 scale serialized in its container:

```text
E_i = sum(w_i^2)
S_i = sum((w_i - decoded_i)^2)
D_i = S_i / E_i
R_i = 8 * literal_container_bytes_i / 262144
gap_i = 10 log10(D_i / 2^(-2 R_i))
```

The original coverage endpoint uses the fixed PLTE slot rate
`R_slot = 2.47930908203125 bpw`, not any smaller realized file length. Its
Gaussian reference is `0.0321593450611333`. A block fails the quality endpoint
when its fixed-slot gap is greater than or equal to 0.10 dB.

The amended endpoint charges each block at its assigned tier boundary plus its
four-bit map entry. Its aggregate gap uses the panel's energy-weighted relative
MSE and mean all-in charged rate. A block fails the amended quality endpoint
when its all-in charged pointwise gap is greater than or equal to 0.10 dB.

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
- for the original endpoint, a base stream or final container exceeding 81,242 bytes;
- for the amended endpoint, an incorrect assigned tier or a final container exceeding its assigned `Tk` boundary;
- arithmetic, causal-frequency, frozen-bit, reconstruction-index, tail-record,
  padding, or header round-trip mismatch;
- an independent decoder that does not consume the exact file or reproduce the
  encoder's literal MSE to absolute tolerance `1e-12`.

Hard failures stop successful finalization. A distortion failure is retained
and disclosed; it does not silently remove the block. After the first exact
cap-overflow exception, already scheduled first-pass encodes may continue only
to measure the cap-length distribution; they do not rescue the failed original
endpoint. Only a clearly systemic pilot collapse may stop that diagnostic
breadth pass early.

## Post-hoc rate-reservoir amendment

The broad first pass falsified the universal 81,242-byte-slot assumption: rare
arithmetic base streams exceeded that cap by a small number of bytes. Those
events remain failures of the original endpoint and their logs are retained.
They are not converted into Tier-0 successes.

For engineering evaluation, a separate deterministic checkpoint reservoir is
frozen before any retry:

```text
T0 = 81242 bytes
Tk = T0 + 64 k bytes
k  = max(0, ceil((base_container_bytes - T0) / 64))
```

Only the exact base-container-overflow exception may trigger a retry. Its tier
is computed directly from the base length; quality metrics never choose the
tier. A successful Tier-0 artifact remains byte-for-byte unchanged. A retry
must reproduce the first-pass base length, uses the same weight bytes, coset,
encoder, decoder, and frozen profile, and changes only
`container_cap_bytes`.

A four-bit tier map in canonical non-router block order supports tiers 0–15.
The exact conditional checkpoint rate is

```text
(75,724,918,048 + 4*116,422 + 512*sum(k_i)) / 30,532,122,624 bpw
```

Including the map, no overflow tiers cost `2.4801873315 bpw`; every non-router
at Tier 1 costs `2.4821396381 bpw`; and every non-router at Tier 10 costs
`2.4997103973 bpw`. The strict global condition is
`sum(k_i) <= 1,181,489`.

The panel packer writes the four-bit map and each literal container followed by
verified zero padding to its assigned `Tk` boundary. The decoder wrapper uses
the existing header to infer the literal prefix, checks every padding byte, and
passes that unchanged prefix to the independent decoder. Per-block quality is
charged at `8*Tk/262144`, not at Tier 0 or merely the shorter literal length.

This repair is strict PTQ, but it was designed after observing first-pass cap
failures. The same 400 frozen blocks therefore become a post-hoc engineering
evaluation of the amended codec, not an untouched confirmatory holdout. A new
disjoint selection is required for a confirmatory reservoir claim.

### Frozen outcome

The immutable original Tier-0 outcome is:

| Quantity | Value |
|---|---:|
| Attempted blocks | 400 |
| Tier-0 successes | 385 |
| Recognized cap failures | 15 / 400 |
| Other failures | 0 |
| Maximum base container | 81,278 bytes |
| Maximum cap overflow | 36 bytes |

The 15 original cap failures were distributed as follows; no selected attention
Q, K, or O block overflowed:

| Role | Tier-0 cap failures |
|---|---:|
| Embedding | 3 |
| LM head | 1 |
| Attention V | 3 |
| Expert gate | 3 |
| Expert up | 2 |
| Expert down | 3 |

The original universal fixed-cap endpoint therefore failed. Reservoir retries
do not reclassify those 15 rows as Tier-0 successes. The frozen amended tier
distribution is 385 Tier 0 + 15 Tier 1, with no higher tier used.

The amended 400-block result, including the four-bit map charge, is:

| Metric | Value |
|---|---:|
| Energy-weighted relative MSE | 0.03271539785114697 |
| Mean all-in charged rate | 2.4793975830078123 bpw |
| Aggregate charged Gaussian gap | 0.07498293435240821 dB |
| Pointwise charged gap p95 | 0.08279060024787634 dB |
| Pointwise charged gap p99 | 0.08458994756368801 dB |
| Pointwise charged gap maximum | 0.08667984346279214 dB |
| Pointwise gaps at or above 0.10 dB | 0 |
| Clean independent decodes | 400 / 400 |

Coverage comprises 336 layer/role cells (`48 × 7`), 32 embedding blocks, and
32 LM-head blocks, alongside the complete 48-router Q4 and 193-rank-one exact
BF16 exception censuses. Exact outcomes are published in
[`original_tier0_outcome.json`](../evaluation/qwen3_stratified_v1/original_tier0_outcome.json),
[`reservoir_plan.json`](../evaluation/qwen3_stratified_v1/reservoir_plan.json),
and [`summary.json`](../evaluation/qwen3_stratified_v1/summary.json).

The source-free artifact set is checked with:

```bash
python tools/verify_stratified_evaluation.py
```

That verifier rebuilds the manifest and reservoir plan and checks all 400
container segments, encoder reports, clean-decoder receipts, the tier map,
padded slot image, metric summaries, and router and rank-one censuses.

## Claim boundary

The valid measured statement is that all 400 frozen new blocks pass the amended
all-in charged reservoir endpoint, with complete layer-by-role coverage in this
panel and 400/400 clean independent decodes. The original preregistered
universal Tier-0 endpoint did not pass: it retains 15/400 cap failures. Because
the reservoir was designed after those failures, its result is post-hoc and
non-confirmatory even though the codec remains strict PTQ.

This is not a statement that every Qwen block passes. Routers remain measured
Q4 exceptions, rank-one tensors remain exact BF16 exceptions, checkpoint rate
remains a conditional inventory calculation, and any full-model distortion
remains sample based until every source block is encoded. The result is neither
a whole-checkpoint measurement nor a definitive SOTA or worst-case claim.
