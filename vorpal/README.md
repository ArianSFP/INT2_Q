# VORPAL implementation

This directory contains the frozen implementation used for the experimental
400-block VORPAL endpoint. VORPAL means **Variance-Ordered Reverse-waterfilled
Polar Adaptive Lattice**. It is strict post-training weight quantization: it
uses frozen checkpoint weights only and performs no retraining, QAT, activation
calibration, gradient optimization, distillation, or task-loss fitting.

The measured release artifact is in
[`evaluation/qwen3_vorpal_v1/`](../evaluation/qwen3_vorpal_v1/). Its physical
bundle contains 32,583,835 bytes for 104,857,600 values, or exactly
2.4859493255615234375 bpw. Independent float64 evaluation over the frozen BF16
sources measured relative MSE 0.02983268081273826 and a -0.2861708909351923 dB
gap to the declared unit-variance Gaussian reference.

This is post-hoc development evidence for the nine declared PLTE matrix roles,
not a disjoint confirmation or whole-checkpoint result. Router and rank-one
exception evidence is published separately and is not included in this `(R,D)`
pair.

## Code map

- `build_continuous_waterfill.py` constructs variance labels, stable group
  order, and continuous reverse-waterfilled profile controls.
- `pack_continuous_side.py` serializes the decoder side information.
- `run_continuous_panel.py` performs the base 400-chunk encode.
- `decode_continuous_chunk.py` and `run_base_clean_scores.py` independently
  score the base containers.
- `adaptive_candidate_audit/` contains the pinned candidate-generation core.
- `adaptive_candidate_audit_v3/` contains the hardened, receipt-producing
  launchers and tests used for the completed scan.
- `select_continuous_adaptive.py` performs the exact Pareto dynamic program
  over the frozen candidate set.
- `outer_v2_fixed_route/` defines, packs, decodes, and evaluates the physical
  fixed-route `WFOUTR01` experiment-v2 format.
- `publication_verifier/` is a standard-library-only, source-free verifier for
  the canonical release layout.
- [`outer_decoder_impl_v1/`](../outer_decoder_impl_v1/) contains the audited v1
  implementation loaded by the fixed-route wrapper. Its location is
  intentional and preserves the wrapper's pinned relative lookup.

The original clean polar decoder and six-level mask remain in [`plte/`](../plte/).
Implementation SHA-256 values are recorded in the artifact inventory and in
the receipts; changing a pinned file creates a different experiment.

## Verification

From the repository root:

```bash
python vorpal/publication_verifier/verify_vorpal_publication.py \
  evaluation/qwen3_vorpal_v1 --compact

python -m unittest -v \
  vorpal/publication_verifier/test_verify_vorpal_publication.py
```

The first command reparses the bundle, recomputes physical rate, checks all 400
containers and role/layer coverage, recomputes the exposed selection frontier,
validates receipts and hashes, and rejects raw or normalized source payloads.
It does not independently remeasure SSE because the source weights are
deliberately not published.

See [`docs/VORPAL_METHOD.md`](../docs/VORPAL_METHOD.md) for the method contract,
[`docs/VORPAL_EVIDENCE.md`](../docs/VORPAL_EVIDENCE.md) for claim boundaries,
and [`docs/VORPAL_REPRODUCING.md`](../docs/VORPAL_REPRODUCING.md) for the full
audited command sequence.
