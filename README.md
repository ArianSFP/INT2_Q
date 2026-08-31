# INT2_Q: Polar-Lattice Tail-Escape PTQ

This repository publishes **PLTE**, a strict post-training weight codec developed while searching for sub-0.10 dB finite-length quantization of Qwen3-30B-A3B matrices below 2.5 bits per original checkpoint parameter.

PLTE combines a six-level polar-lattice source code, causal arithmetic coding, fixed-slot sparse exact-BF16 tail escapes, and a literal Q4 exception for MoE routers. It uses frozen weights only: no retraining, QAT, activation calibration, distillation, task loss, or untransmitted learned decoder state.

> [!IMPORTANT]
> This is a **preliminary mixed-precision research candidate**, not a homogeneous INT2 codec and not a whole-checkpoint SOTA claim. Routers use Q4 and rank-one tensors remain BF16. The historical global distortion number is a projection from six non-router blocks plus exact router aggregates; the 400-block result below is a measured panel result, not a whole-checkpoint measurement. The checkpoint rate remains a conditional inventory calculation, not a realized checkpoint file.

## Result status

| Status | What is established |
|---|---|
| **Historical measured evidence** | 47 unique selected frozen Qwen blocks were literally encoded and decoded. Every observed block is below 0.10 dB; the maximum is **0.084674951 dB**. |
| **Historical independent decodes** | A clean decoder that does not import the encoder exactly reproduces both normative exemplars and all six controlled-projection blocks. |
| **Original 400-block endpoint** | **Failed as specified:** 385 Tier-0 successes and 15/400 recognized cap failures, with maximum base length 81,278 bytes (36 bytes over cap) and zero other failures. |
| **Post-hoc reservoir panel** | The amended 385 Tier-0 + 15 Tier-1 panel has 400/400 clean independent decodes, energy-weighted relative MSE **0.0327153979**, mean all-in charged rate **2.4793975830 bpw**, and aggregate charged gap **0.0749829344 dB**. No block reaches 0.10 dB. |
| **Panel coverage** | All 48 layers × 7 PLTE roles, 32 embedding blocks, and 32 LM-head blocks, plus complete 48-router and 193-rank-one exception censuses. This is complete stratum coverage in the panel, not complete checkpoint encoding. |
| **Router-complete** | All 48 router matrices, 12,582,912 weights total, were encoded with the literal all-Q4 format at relative MSE **0.0332189501**. |
| **Historical projection** | Six adjacent expert down-projection blocks plus the exact router aggregate give relative MSE **0.0327395512** and Gaussian gap **0.082851019 dB**. |
| **Historical conditional rate** | Fixed-inventory arithmetic gives **2.480172079 bpw** under the now-falsified assumption that every non-router block fits an 81,242-byte slot. |
| **Not established** | Full-checkpoint distortion, full-checkpoint slot feasibility, a realized packed checkpoint, perplexity, downstream accuracy, or superiority over published methods under a common harness. |

## Preregistered broad-coverage evaluation

A metadata-only selection of **400 new, previously untested blocks** is frozen in
[`evaluation/qwen3_stratified_v1/manifest.json`](evaluation/qwen3_stratified_v1/manifest.json).
The manifest was committed before the selected weight payloads were fetched. It
covers all 48 layers across each of the seven non-router matrix roles
(`48 × 7 = 336` blocks), plus 32 embedding and 32 LM-head blocks. The existing
router artifact is a census of all 48 routers, and the extension separately
audits all 193 rank-one tensors as lossless BF16 exceptions.

The preregistered universal Tier-0 endpoint did **not** pass. Of 400 attempted
blocks, 385 fit the original 81,242-byte cap and 15 produced recognized cap
failures; the largest base container was 81,278 bytes, or 36 bytes over cap.
There were zero other failures. This immutable outcome is published in
[`original_tier0_outcome.json`](evaluation/qwen3_stratified_v1/original_tier0_outcome.json).

After those failures were observed, a deterministic 64-byte tier reservoir was
designed and frozen. The amended panel assigns 385 blocks to Tier 0 and all 15
original cap failures to Tier 1. All 400 literal containers then pass clean
independent decoding. Charged at each assigned slot plus its four-bit map entry,
the measured panel has:

| 400-block amended-panel metric | Value |
|---|---:|
| Energy-weighted relative MSE | 0.03271539785114697 |
| Mean all-in charged rate | 2.4793975830078123 bpw |
| Aggregate charged Gaussian gap | 0.07498293435240821 dB |
| Pointwise charged gap p95 | 0.08279060024787634 dB |
| Pointwise charged gap p99 | 0.08458994756368801 dB |
| Pointwise charged gap maximum | 0.08667984346279214 dB |
| Pointwise gaps at or above 0.10 dB | 0 |
| Clean independent decodes | 400 / 400 |

The source weights remain frozen and there is no retraining, calibration, or
task feedback, so both runs are strict PTQ. The reservoir, however, is a
**post-hoc engineering amendment**, and these same 400 blocks are therefore not
an untouched confirmatory set for it. The exact results and boundary are in
[`summary.json`](evaluation/qwen3_stratified_v1/summary.json) and
[`docs/STRATIFIED_EVALUATION.md`](docs/STRATIFIED_EVALUATION.md). They are not a
whole-checkpoint distortion measurement, a checkpoint-wide worst-case result,
or a definitive SOTA claim.

The shaping gap used here is

```text
gap_dB = 10 log10(D / 2^(-2R))
```

where `D = SSE / source_energy` and `2^(-2R)` is the unit-variance Gaussian rate-distortion reference. It is not task accuracy and not an error amplitude expressed in dB.

## Conditional budget and projection

| Quantity | Value |
|---|---:|
| Total conditional budget | 75,724,918,048 bits |
| Rate | 2.480172079109753 bpw |
| Headroom below 2.5 bpw | 0.019827920890247 bpw |
| Gaussian reference at that rate | 0.0321208936559302 |
| 0.10 dB distortion ceiling | 0.0328690853839087 |
| Projected relative MSE | 0.032739551181266 |
| Projected Gaussian gap | 0.082851019174621 dB |

The exact accounting and its assumptions are in [`plte/agent_root_polar_escape_full_model_ledger.json`](plte/agent_root_polar_escape_full_model_ledger.json). This table is the historical six-block projection and is separate from the new measured panel. The historical and stratified sets contain 447 unique PLTE blocks, or 0.38395% of the 116,422 non-router blocks; 115,975 remain unencoded. The amended panel's 2.4793975830078123 bpw is its mean all-in charged rate, not a realized checkpoint rate.

## Verify the published evidence

The repository includes the exact evidenced implementations, 49 historical compact polar report/stream pairs, eight historical clean-decoder reports, the six-mask profile, the all-router literal artifact, and the source-free 400-block stratified bundle. Raw Qwen weight bytes are deliberately excluded.

```bash
python tools/verify_repository.py
python tools/verify_stratified_evaluation.py
```

The first standard-library verifier checks the historical publication bundle.
The exact stratified-artifact verifier independently rebuilds the frozen
selection and reservoir plan; checks the 400-container bundle, tier map, padded
slot image, encoder reports, and 400 independent-decoder receipts; recomputes
the published source-free metrics; and validates the 193 rank-one and 48-router
censuses. Neither verifier remeasures distortion from the deliberately omitted
raw BF16 sources.

## Repository map

- [`docs/METHOD.md`](docs/METHOD.md): architecture, format, and mathematical motivation.
- [`docs/EVIDENCE.md`](docs/EVIDENCE.md): measured versus projected evidence and router trade-offs.
- [`docs/STRATIFIED_EVALUATION.md`](docs/STRATIFIED_EVALUATION.md): frozen 400-block protocol, original failure, and post-hoc reservoir result.
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md): exact environment and reproduction commands.
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md): provenance, hashes, omissions, and directory layout.
- [`plte/`](plte/): exact implementations and compact evidence bundle.
- [`tools/verify_repository.py`](tools/verify_repository.py): weight-free publication integrity check.
- [`tools/verify_stratified_evaluation.py`](tools/verify_stratified_evaluation.py): exact source-free verifier for the stratified artifacts.

## Research basis and novelty boundary

Polar lattices achieving the Gaussian rate-distortion bound and integrating entropy coding are established in [Liu, Shi, and Ling](https://arxiv.org/abs/1501.05683). A later [quantization-goodness proof](https://arxiv.org/abs/2405.04051) establishes asymptotic normalized-second-moment goodness. The numerical reliability-order reference is pinned to commit `458187b9b03db1768a4b72d617e591f7862f6fca` of [graceBaoXP/PolarLatticeQuantization](https://github.com/graceBaoXP/PolarLatticeQuantization).

The mathematical primitives are published. The research contribution here is the deployment combination for frozen neural weights: literal multilevel MAP source coding, causal-prior range coding, fixed-slot largest-error exact BF16 escapes, checkpoint accounting, a clean decoder, and a role-specific router format. No exhaustive patent or novelty search has been performed.

## Checkpoint provenance

- Model: [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- Immutable revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Upstream license metadata: Apache-2.0
- Parameters inventoried: 30,532,122,624

The source-weight cache is not committed. Fetch only the immutable blocks needed for reproduction and comply with the upstream model license. See [`third_party/NOTICES.md`](third_party/NOTICES.md).
