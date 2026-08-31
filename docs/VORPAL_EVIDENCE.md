# Evidence, claims, and limitations

## What is complete

- A frozen 400-block source ledger binds the checkpoint revision, canonical
  order, identities, and BF16 source hashes.
- All 400 construction chunks were encoded: 339 base A64 and 61 base A128.
- All 400 base containers passed internal round trips.
- A clean decoder produced exact-source diagnostic scores for all 400 chunks.
- The base aggregate is `SSE = 1961.1814347120044`,
  `energy = 65382.85877511848`, and `D = 0.029995345438433692`.
- Exact base containers occupy 32,547,152 bytes. Adding the fixed-route
  prelude gives 32,582,374 bytes, 2.4858378601 bpw, and a diagnostic signed
  Gaussian-reference gap of -0.2632261169 dB.
- The all-400 adaptive scan is complete: 38 triggers, 342 tail candidates,
  and 38 A128 upgrades, with zero receipt failures.
- The exact fixed-route selector materialized a 32,583,835-byte bundle from
  362 base choices, 18 A128 upgrades, and 20 sparse-tail choices.
- An independent source-free decoder reconstructed all 400 chunks and bound
  the 838,860,800-byte float64 reconstruction by SHA-256.
- A separate CuPy evaluator verified all 400 frozen BF16 source hashes and
  measured `SSE = 1950.5459564624546`, `energy = 65382.858775118555`,
  `D = 0.02983268081273826`, and signed gap `-0.2861708909351923 dB`
  at exactly `2.4859493255615234375 bpw`.

The final values are float64 evaluation results over exact frozen BF16 source
values. Tiny last-digit differences from the selector prediction are reduction
order effects, not source or reconstruction mismatches.

## Publication gate — passed

A final result may be reported only when all of the following are present and
hash-bound:

1. fixed-route selection receipt and its checksum;
2. selected manifest and exactly 400 staged containers;
3. literal side and side receipt;
4. one `selected.wfouter` physical bundle and bundle receipt;
5. source-free independent decode receipt and float64 reconstruction hash;
6. separate exact-source evaluation receipt;
7. recomputed aggregate and strata; and
8. literal passes of `R < 2.5` and signed `gap_dB <= -0.10`.

All eight conditions pass in the published artifact tree. The standard-library
verifier reparses the physical bundle, recomputes the exposed selection
frontier and metric formulae, checks all 400 containers and coverage cells, and
finds no raw or normalized source payloads.

Key immutable identities are:

| Artifact | SHA-256 |
|---|---|
| Selected bundle | `320320eac95720d79633da57a048a75c10b63259de518f92684a6e84fc7026d2` |
| Selection receipt | `41e2da7a5e97a30e28160dd12ca684ecf46cf5fb7618fe222dae32beabd38eb4` |
| Decode receipt | `da1fdd1862407ca6b9173112ab0d7e7099ab132169c43f9d0f29ed45227a83aa` |
| Evaluation receipt | `81d838e1e51f324bff856eb22a5830c8fb9b70c0c12d319965952056e10c1ee6` |
| Reconstruction | `84b034fbbd92c83b2e953554793fee2af968bb19086722a47a75280d7d3483b5` |

## Claim boundary

The defensible claim is limited to the measured physical `(R,D)` pair for this
heterogeneous 400-block Qwen development panel under the stated metric. It is
not yet evidence for:

- a complete Qwen checkpoint;
- every Qwen model, tensor, expert, or block;
- downstream accuracy, perplexity, latency, memory bandwidth, or kernel speed;
- homogeneous INT2 arithmetic or an unqualified “INT2 model” deployment;
- a pointwise gap guarantee for every block;
- comparison with another method evaluated using a different source panel,
  distortion normalization, or byte ledger; or
- confirmatory generalization, because policy choices were made post hoc on
  this development panel.

Embedding and LM-head coverage consists of 32 blocks each, not every block in
those large tensors. The seven layerwise roles have exactly one canonical
sample in each layer 0–47; expert-role samples do not constitute a census of
all experts. Cross-tensor permutation also trades tensor-local random access
for source homogeneity.

The VORPAL panel contains the nine PLTE matrix roles. It contains no router or
rank-one tensor payloads; the repository's separate stratified release covers
all 48 router matrices and all 193 rank-one exceptions. Those exception results
are not folded into this VORPAL `(R,D)` pair.

The aggregate pass is not pointwise. At the common aggregate charged panel
rate, 233/400 diagnostic block gaps are negative and 167/400 are positive; the
maximum is `+5.691621761805742 dB`. Cross-tensor joint allocation is evaluated
by the energy-weighted aggregate declared above.

## Base diagnostic arithmetic

The frozen values reproduce as follows:

```text
M = 400 * 262144 = 104857600
B_base = 32547152 + 35222 = 32582374
R_base = 8 * 32582374 / 104857600
       = 2.485837860107421875
D_base = 1961.1814347120044 / 65382.85877511848
       = 0.029995345438433692...
gap_base = 10 log10(D_base / 2^(-2 R_base))
         = -0.263226116892... dB.
```

The displayed 2.4858378601 bpw and -0.2632261169 dB are rounded views of those
identities. Final receipts should retain enough decimal precision to reproduce
the target comparison without rounding ambiguity.

## Development-panel caveat

The panel was originally frozen for broad role/layer coverage, but VORPAL and
its adaptive trigger/tail policy were developed using observations from these
same payloads. The result is therefore an honest post-hoc development result.
A disjoint panel frozen before further method changes is required for a
confirmatory claim; a whole-checkpoint claim requires physical packing and
evaluation of the entire declared checkpoint representation.
