# Publication schema audit notes and remaining trust boundaries

1. **Resolved producer issue: candidate energy equality originally blocked the selector.** The completed
   candidate-v3 receipt has SHA-256
   `091537886638d5b024aca11b6d00d16330396f4cad845e0604ffa594d1aeeec3`.
   For real row 011 it serializes base energy `48.66798296516522` and A128
   independent-decode energy `48.667982965165194`. The selector originally
   parsed both as `Decimal` and required exact equality. This was
   subsequently repaired and audited with bounded ULP and relative tolerances;
   exact decode-to-upgrade and tail identities remain required. The real
   fixed-route selection now passes this verifier's pre-evaluation audit.

2. **Resolved producer issue: the final selection receipt binds both selector
   layers.** `selector_dependency_bindings` records the fixed-route wrapper,
   delegated selector core, and pinned core hashes. The verifier also
   independently recomputes the exposed DP.

3. **Candidate completeness is only partly portable.** Candidate-v3 contains
   private absolute paths to base reports, upgrade reports/decodes, and tail
   containers. The publication can prove that the selection used every option
   serialized in candidate-v3, but it cannot source-free revalidate those
   private files unless a portable candidate evidence bundle or Merkle/index
   of those artifacts is published.

4. **The ledger's `selection_manifest_sha256` has no standard artifact slot.**
   It names the earlier block-selection manifest, not the adaptive
   `selected.manifest.json`. Include that upstream manifest in a future
   publication index if the hash is intended to be independently resolved.

5. **Receipts lack canonical argv/log bindings.** Current independent decoder
   and evaluator receipts bind implementation hashes and artifacts, but do not
   include canonical argv, stdout/stderr hashes, environment lock, or an
   external signature/attestation. They establish deterministic internal
   consistency, not who executed the commands.

6. **The GitHub release omits the 838,860,800-byte reconstruction.** The decode
   and evaluation receipts cross-bind its SHA-256. Supplying the locally backed
   up `reconstruction.f64` lets this verifier rehash the bytes; omitting it
   leaves that edge receipt-bound while the compressed bundle remains fully
   physical and reproducible.

7. **Resolved publication gate: the real final layout passed.** The verifier
   passed the 413-file canonical artifact layout without reconstruction, the
   same 414 files with the materialized reconstruction, and the 417-file
   repository layout with documentation and checksums. It found zero raw or
   normalized sources. Future schema changes should still fail closed and be
   reviewed rather than relaxed automatically.
