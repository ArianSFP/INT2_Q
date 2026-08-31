# INT2_Q: Polar-Lattice Tail-Escape PTQ

This repository publishes **PLTE**, a strict post-training weight codec developed while searching for sub-0.10 dB finite-length quantization of Qwen3-30B-A3B matrices below 2.5 bits per original checkpoint parameter.

PLTE combines a six-level polar-lattice source code, causal arithmetic coding, fixed-slot sparse exact-BF16 tail escapes, and a literal Q4 exception for MoE routers. It uses frozen weights only: no retraining, QAT, activation calibration, distillation, task loss, or untransmitted learned decoder state.

> [!IMPORTANT]
> This is a **preliminary mixed-precision research candidate**, not a homogeneous INT2 codec and not a whole-checkpoint SOTA claim. Routers use Q4 and rank-one tensors remain BF16. The global distortion number is a projection from six non-router blocks plus exact router aggregates; the global rate is a conditional fixed-slot budget, not a realized checkpoint file.

## Result status

| Status | What is established |
|---|---|
| **Measured** | 47 unique selected frozen Qwen blocks were literally encoded and decoded. Every observed block is below 0.10 dB; the maximum is **0.084674951 dB**. |
| **Independently decoded** | A clean decoder that does not import the encoder exactly reproduces both normative exemplars and all six controlled-projection blocks. |
| **Router-complete** | All 48 router matrices, 12,582,912 weights total, were encoded with the literal all-Q4 format at relative MSE **0.0332189501**. |
| **Projected** | Six adjacent expert down-projection blocks plus the exact router aggregate give relative MSE **0.0327395512** and Gaussian gap **0.082851019 dB**. |
| **Conditional rate** | Fixed-inventory arithmetic gives **2.480172079 bpw**, provided every non-router block fits an 81,242-byte slot. |
| **Not established** | Full-checkpoint distortion, full-checkpoint slot feasibility, a realized packed checkpoint, perplexity, downstream accuracy, or superiority over published methods under a common harness. |

## Preregistered broad-coverage extension

A metadata-only selection of **400 new, previously untested blocks** is frozen in
[`evaluation/qwen3_stratified_v1/manifest.json`](evaluation/qwen3_stratified_v1/manifest.json).
It covers all 48 layers across each of the seven non-router matrix roles, plus
32 embedding and 32 LM-head strata. The existing router artifact is a census of
all 48 routers, and the extension separately audits all 193 rank-one tensors as
lossless BF16 exceptions. The design, failure rules, metrics, and claim boundary
are specified in [`docs/STRATIFIED_EVALUATION.md`](docs/STRATIFIED_EVALUATION.md).

The manifest was committed before the selected weight payloads were fetched.
Until result artifacts are published, it is a test protocol rather than new
performance evidence.

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

The exact accounting and its assumptions are in [`plte/agent_root_polar_escape_full_model_ledger.json`](plte/agent_root_polar_escape_full_model_ledger.json). The evidence covers 47 of 116,422 non-router blocks (0.0403704%); 116,375 remain unencoded.

## Verify the published evidence

The repository includes the exact evidenced implementations, 49 compact polar report/stream pairs, eight clean-decoder reports, the six-mask profile, and the all-router literal artifact. Raw Qwen weight bytes are deliberately excluded.

```bash
python tools/verify_repository.py
```

This standard-library verifier checks every published report/container hash, required literal audit flag, implementation hash, frozen-mask level, standalone-decoder linkage, router artifact, and ledger invariant. It does **not** remeasure MSE without the original BF16 sources.

## Repository map

- [`docs/METHOD.md`](docs/METHOD.md): architecture, format, and mathematical motivation.
- [`docs/EVIDENCE.md`](docs/EVIDENCE.md): measured versus projected evidence and router trade-offs.
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md): exact environment and reproduction commands.
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md): provenance, hashes, omissions, and directory layout.
- [`plte/`](plte/): exact implementations and compact evidence bundle.
- [`tools/verify_repository.py`](tools/verify_repository.py): weight-free publication integrity check.

## Research basis and novelty boundary

Polar lattices achieving the Gaussian rate-distortion bound and integrating entropy coding are established in [Liu, Shi, and Ling](https://arxiv.org/abs/1501.05683). A later [quantization-goodness proof](https://arxiv.org/abs/2405.04051) establishes asymptotic normalized-second-moment goodness. The numerical reliability-order reference is pinned to commit `458187b9b03db1768a4b72d617e591f7862f6fca` of [graceBaoXP/PolarLatticeQuantization](https://github.com/graceBaoXP/PolarLatticeQuantization).

The mathematical primitives are published. The research contribution here is the deployment combination for frozen neural weights: literal multilevel MAP source coding, causal-prior range coding, fixed-slot largest-error exact BF16 escapes, checkpoint accounting, a clean decoder, and a role-specific router format. No exhaustive patent or novelty search has been performed.

## Checkpoint provenance

- Model: [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- Immutable revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Upstream license metadata: Apache-2.0
- Parameters inventoried: 30,532,122,624

The source-weight cache is not committed. Fetch only the immutable blocks needed for reproduction and comply with the upstream model license. See [`third_party/NOTICES.md`](third_party/NOTICES.md).
