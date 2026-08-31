# Exact adaptive selection and staging

`select_continuous_adaptive.py` is the write-once selection boundary between
exploratory candidate generation and a literal 400-container ensemble. It does
not decode or alter PLTE payloads and does not modify the base or candidate
directories.

## Inputs and trust boundary

- The original continuous-waterfill manifest.
- The completed `full_jobs` directory, including `run.receipt.json`, 400 encoder
  reports, and 400 self-delimiting containers.
- A completed canonical V3 `candidate.receipt.json` from the pinned adaptive
  runner (`continuous PLTE all-base adaptive candidate receipt v3`). No V2
  aliases or schema translation are accepted.
- The evaluator's exact aggregate base raw SSE, passed explicitly as a decimal
  string through `--base-total-raw-sse`.
- The existing `pack_continuous_side.py`; its SHA-256 is bound into the output.
- The final `pack_bundle.py` and pinned raw six-level mask. Their bytes and
  SHA-256 hashes are bound too.

Every base report, source binding, internal round-trip flag, container length,
and container SHA-256 is checked. Every candidate report/decode receipt/hash is
checked too. Sparse-tail candidates must preserve the base scale and arithmetic
payload byte-for-byte, have a self-consistent header and length, and contain
zero tail padding. All four hardened tail booleans must be literal JSON `true`.
The candidate trigger set must reproduce the strict `gap > 0.10 dB` predicate
over all 400 base reports, including the A128 region, which prevents silently
omitted difficult chunks. An A64 base row must contain base, explicit A128
alphabet upgrade, and every declared tail prefix. An A128 base row must contain
base plus those tails and a null upgrade; A256 is neither generated nor implied.

Raw-source energy is option-invariant but is reduced by two audited GPU paths:
the base/tail repacker performs one CuPy float64 reduction, while the independent
A128 decoder sums per-group CuPy float64 reductions. Candidate energy fields
must be literal finite positive JSON decimals. Base and every tail remain exact;
an A128 energy is accepted only after its normalized-source SHA-256, manifest,
metadata/report, container, scorer, decoder, and mask bindings all match and
the independently reduced value is both at most four binary64 ULPs and at most
`1e-15` relative from the canonical base value. The selection receipt records
the comparison census and observed maxima. Five ULPs, material relative error,
strings, booleans, integers, Python floats, NaN, and infinities fail closed.

## Exact optimization

Each triggered A64 chunk contributes base, A128 upgrade, and every serialized
sparse-tail option; each triggered A128 chunk contributes base and every tail.
Tails always preserve the base alphabet. Decimal SSE values are parsed directly
from JSON and the DP axis is exact integer physical container-byte delta.

The final state minimizes

```text
(base_total_raw_sse - total_saving)
    * 2**(16 * WFOUTR01_bundle_bytes / panel_values)
```

under a strict `< 2.5 bpw` default budget. Transcendental comparisons use
80-digit `Decimal` arithmetic. The source energy is constant and therefore
cannot affect the selected state; if available, it is used to report the final
raw MSE and Gaussian-reference gap.

The claim-grade mode uses isolated side codec ID 3. It normalizes every profile
alphabet byte in the literal side to A64, compresses that canonical side once
with Python `lzma` `FORMAT_XZ` preset 9, then appends one exact 50-byte,
LSB-first A64/A128 route. The route length is invariant for every DP state, so
ordinary byte/SSE Pareto dominance is globally exact. The mask is compressed
once with BZ2 level 9. The physical prelude is exactly:

```text
168-byte WFOUTR01 header + len(XZ(canonical side)) + 50 + len(BZ2(raw mask))
```

The 47,356-byte literal raw side is never called a physical prelude or directly charged.
After selection, the real side packer must reproduce the modeled raw side
byte-for-byte. The real bundle packer then emits `selected.wfouter`, reparses it
source-free, and must reproduce the modeled XZ hash/length, BZ2 hash/length,
ordered container hashes, physical prelude, bundle length, and exact objective.

The fixed-route wrapper forces `--physical-selection fixed-route`; this is the
mode valid for the final claim. `exact` means exhaustive current-codec XZ
signatures and is retained for small diagnostics. `proxy` is explicitly
non-global and is not valid for the final claim.

### Physical result and dependency boundary

On the real side and mask, canonical XZ is 33,756 bytes, route is 50 bytes,
mask BZ2 is 1,248 bytes, and the retained header is 168 bytes: a 35,222-byte
prelude. This is only 46 bytes above the current literal-side-XZ prelude.

Codec ID 3 remains isolated from audited v1. Its decoder, packer, and evaluator
receipts separately bind the fixed wrapper, shared codec helper, pinned audited
v1 outer decoder, fixed evaluator wrapper, and pinned delegated v1 evaluator.
The v1 module `__file__` is never rewritten. Every dependency binding and an
unaudited outer override have adversarial fail-closed tests.

## Invocation

```bash
/root/int2-venv/bin/python /root/negative_gap_root/continuous_v1/outer_v2_fixed_route/select_continuous_adaptive_fixed_route_v2.py \
  --manifest /root/negative_gap_root/continuous_v1/panel/manifest.json \
  --base-dir /root/negative_gap_root/continuous_v1/full_jobs \
  --candidate-receipt /root/negative_gap_root/continuous_v1/adaptive_candidates_v3_t010/candidate.receipt.json \
  --base-total-raw-sse '<exact evaluator decimal>' \
  --packer /root/negative_gap_root/continuous_v1/pack_continuous_side.py \
  --bundle-packer /root/negative_gap_root/continuous_v1/outer_v2_fixed_route/pack_bundle_fixed_route_v2.py \
  --raw-mask /root/int2/INT2_Q_stratified_run/plte/agent_root_polar_escape_frozen_profiles.bin \
  --python /root/int2-venv/bin/python \
  --output-dir /root/negative_gap_root/continuous_v1/selected_fixed_route_v2
```

The output path must not exist. A successful directory contains:

- `selected.manifest.json`, with only selected alphabet overrides;
- `side.bin` and the packer's exact round-trip receipt;
- `containers/wf-000.polar.bin` through `wf-399.polar.bin`;
- `selected.wfouter` and its source-free bundle pack receipt; this file alone
  supplies the charged byte count;
- `selection.receipt.json`, which binds every input, considered option,
  selected mapping, output hash, physical byte count, DP frontier hash, and
  all-in objective;
- `selection.receipt.sha256`, a detached integrity checksum.

The staging copies are canonical inputs for the independent outer decoder. No
candidate report, encoder JSON, normalized source, or original Qwen source is a
decoder input.

## Synthetic verification

```bash
python -m py_compile select_continuous_adaptive.py test_select_continuous_adaptive.py
python -m unittest -v test_select_continuous_adaptive.py
```

The tests compare both the proxy DP and signature-aware physical XZ rerank
against exhaustive enumeration, prove cross-signature dominated states are
preserved, cover negative byte deltas and the strict rate boundary, validate
self-delimiting containers, arbitrary large tail prefixes (including 240 and
480), literal hardened booleans, V2-schema rejection, and tail padding failures,
and run a synthetic
400-container write-once pack through A64/A128 override, side regeneration,
physical bundle creation, and receipt verification.
