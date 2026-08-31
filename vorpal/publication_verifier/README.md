# VORPAL continuous-400 source-free verifier

This isolated verifier checks the forthcoming fixed-route VORPAL publication
without model weights, BF16 sources, NumPy, CuPy, or encoder/decoder imports.
It is standard-library-only and fails closed on missing or inconsistent
evidence.

Run:

```text
python verify_vorpal_publication.py /path/to/publication
python -m unittest -v test_verify_vorpal_publication.py
```

## Canonical publication layout

```text
publication/
  construction.manifest.json
  selected.manifest.json
  source_ledger.json
  base.run.receipt.json
  candidate.receipt.json
  selection.receipt.json
  selection.receipt.sha256
  side.bin
  side.receipt.json
  selected.wfouter
  bundle.receipt.json
  decode.receipt.json
  evaluation.json
  containers/
    wf-000.polar.bin
    ...
    wf-399.polar.bin
```

An optional `reconstruction.f32` or `reconstruction.f64` is rehashed when
present. Its absence is reported explicitly but is not a failure because the
physical bundle plus independent decoder is the reproducible source-free
representation.

## Independently checked

- Exact codec-3 WFOUTR01 header, XZ EOF, canonical all-A64 side, LSB-first
  400-bit route, reconstructed literal-side hash, BZ2 EOF, normative embedded
  mask, 400 self-delimiting containers, zero padding, and exact final EOF.
- Physical file bytes and SHA-256 at every available receipt edge; exact
  bundle byte partition and all-in `8*bytes/104857600` rate.
- Frozen strict-PTQ/no-retraining flags and audited implementation hashes.
- Construction/selected manifest identity, with only selected alphabet codes
  permitted to change.
- Candidate-v3 completeness declarations and exact correspondence between
  every exposed candidate and `options_considered`.
- Recomputed integer-byte/Decimal Pareto frontier, frontier hash, optimal
  choice, tie-break, materialized selection map, and staged/bundled bytes.
- Independent decoder source-isolation declaration plus all 400 chunk receipt
  hashes, lengths, padding, route alphabets, and reconstruction binding.
- Exact-source evaluation identity, aggregate SSE/energy/MSE, Gaussian
  reference, signed gap formula, strict `<2.5` rate, and required
  `gap <= -0.10 dB`.
- Exact canonical panel coverage: 400 blocks, nine roles, every layer `0..47`
  exactly once for each of seven layerwise roles, and 32 embedding plus 32 LM
  head blocks.
- Recomputed role, layer, role/layer, and mixed-chunk aggregates.
- Recursive absence of raw BF16 sources, normalized sources, model/checkpoint
  arrays, symlinks, known source hashes, and undeclared block-sized payloads.

## Claim boundary

This is a source-free artifact verifier, not a second exact-source evaluator.
It independently verifies physical coding, provenance bindings, exposed
selection optimality, identities, accounting, and all formulas. Individual
SSE measurements necessarily remain assertions of the exact-source evaluator
because publishing or rereading the frozen BF16 sources would violate the
source-free boundary. See `SCHEMA_BLOCKERS.md` for resolved producer issues and
the remaining portable-evidence and execution-attestation trust boundaries.
