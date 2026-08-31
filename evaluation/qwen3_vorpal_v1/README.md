# VORPAL publication section

> **Status — 2026-08-31:** VORPAL is a provisional name for
> **V**ariance-**O**rdered **R**everse-waterfilled **P**olar **A**daptive
> **L**attice PTQ. The experimental fixed-route-v2 development endpoint has
> passed physical packing, independent source-free decode, exact-source CuPy
> evaluation, and the standard-library publication verifier.

VORPAL is a strict post-training weight codec for frozen Qwen BF16 matrices.
It describes local variance compactly, deterministically pools similarly
scaled 2,048-value groups across tensor and layer boundaries, assigns
polar-lattice profiles by reverse water-filling, and routes a small set of
A64/A128 and sparse-tail alternatives under an exact physical-byte objective.
It uses no training, gradients, activations, prompts, labels, Hessians, task
loss, or weight updates.

## Frozen panel

The development panel contains 400 blocks from `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, each with `2^18 = 262,144`
weights: 104,857,600 frozen BF16 values in total. It covers nine roles:

- all 48 layers for each of seven layerwise roles: attention Q/K/V/O and
  expert MLP gate/up/down (`7 * 48 = 336` blocks);
- 32 embedding blocks; and
- 32 LM-head blocks.

This is broad role/layer coverage, not a complete-checkpoint census. Method
and routing choices were developed after inspecting this panel, so its results
are development evidence rather than a disjoint confirmatory evaluation.

## Completed base diagnostic

The base diagnostic combines the exact clean-decoder score with the fully
charged fixed-route prelude. It is not substituted for the final selected
bundle/decode/evaluation result.

| Quantity | Frozen value |
|---|---:|
| Source values | 104,857,600 |
| Source energy | 65,382.85877511848 |
| SSE | 1,961.1814347120044 |
| Energy-relative distortion, `D` | 0.029995345438433692 |
| PLTE container bytes | 32,547,152 |
| Fixed-route prelude bytes | 35,222 |
| Physical all-in base bytes | 32,582,374 |
| Physical all-in rate | 2.4858378601 bpw |
| Aggregate i.i.d. unit-variance Gaussian-reference gap | -0.2632261169 dB |

The adaptive receipt is complete and has SHA-256
`091537886638d5b024aca11b6d00d16330396f4cad845e0604ffa594d1aeeec3`.
It evaluates all 400 base gaps, triggers the 38 chunks with gap strictly above
`0.10 dB`, and records 342 sparse-tail candidates (nine prefixes per trigger)
plus 38 A128 upgrades. All 38 triggered base chunks are A64.

## Exact publication metric

For a literal final bundle of `B` bytes and `M = 104,857,600` source values:

```text
R       = 8B / M
D       = sum_i (w_i - w_hat_i)^2 / sum_i w_i^2
D_G(R)  = 2^(-2R)
gap_dB  = 10 log10(D / D_G(R)).
```

The target is the signed predicate `R < 2.5` and `gap_dB <= -0.10`. The rate
charges the literal `WFOUTR01` file: header, fixed-route side payload,
compressed mask, every container header and payload, sparse tails, and byte
padding. Nominal profile rate is never used in place of physical bytes.

`D_G` is the rate-distortion curve of an i.i.d. unit-variance Gaussian source,
not a universal lower bound for heterogeneous Qwen weights. A negative gap can
reflect charged exploitation of variance and tail structure; it is not a
Shannon violation.

## Final 400-block development-panel result

| Field | Value |
|---|---:|
| Selected bundle bytes | 32,583,835 |
| Physical all-in bpw | 2.4859493255615234375 bpw |
| Independently decoded chunks | 400/400 |
| Float64 source energy | 65,382.858775118555 |
| Float64 source SSE | 1,950.5459564624546 |
| Energy-relative distortion | 0.02983268081273826 |
| Gaussian reference | 0.031864665997923494 |
| Signed Gaussian-reference gap | **-0.2861708909351923 dB** |
| Target verdict | **PASS** — `R < 2.5` and `gap_dB <= -0.10` |

The 260,670,680 encoded bits leave 0.0140506744384765625 bpw of
rate headroom. The gap is 0.1861708909351923 dB beyond the requested threshold.
The exact selector chose 362 base containers, 18 A64-to-A128 upgrades, and 20
sparse-tail containers carrying 74 tail records. The selected stream contains
321 A64 and 79 A128 chunks.

These values come only from the physically materialized bundle, the source-free
decode receipt, and the separate exact-source evaluator receipt. The final
evaluation used CuPy 14.2.0 on an NVIDIA RTX A6000 and verified all 400 source
hashes, ordinals, and scatter coverage.

This remains a post-hoc development result on the nine declared PLTE matrix
roles. Routers and rank-one exception tensors are covered by the repository's
separate stratified evidence, not by this VORPAL aggregate. It is not a whole
checkpoint, pointwise, task-accuracy, kernel-speed, or definitive SOTA claim.

Further details are in [VORPAL_METHOD.md](../../docs/VORPAL_METHOD.md),
[VORPAL_REPRODUCING.md](../../docs/VORPAL_REPRODUCING.md),
[VORPAL_EVIDENCE.md](../../docs/VORPAL_EVIDENCE.md), and the machine-readable
[ARTIFACT_INVENTORY.json](ARTIFACT_INVENTORY.json).
