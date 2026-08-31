# Reproduction and finalization

Commands below describe the audited RunPod execution and use the CuPy
environment at `/root/int2-venv/bin/python`. The frozen implementations are
published under `vorpal/`, with the audited v1 decoder under
`outer_decoder_impl_v1/`. Use fresh write-once output paths; never overwrite
the frozen evidence tree in `evaluation/qwen3_vorpal_v1/`.

The producer sections use two explicit roots: a Git checkout for code and
published metadata, and the archived RunPod experiment tree for private source
payloads and frozen receipts. Choose a new `REPRO_ROOT`; the guard below fails
if it already exists.

```bash
export REPO_ROOT=/root/int2/INT2_Q_stratified_run
export RUN_ROOT=/root/negative_gap_root/continuous_v1
export POLAR_REPO=/root/PolarLatticeQuantization
export PY=/root/int2-venv/bin/python
export SOURCE_ROOT="$REPO_ROOT"
export SOURCE_DIR="$SOURCE_ROOT/tmp/qwen3_stratified_v1/sources"
export RESULTS_JSON="$REPO_ROOT/evaluation/qwen3_stratified_v1/results.json"
export REPRO_ROOT=/root/vorpal_reproduction_20260831_01
export PANEL_DIR="$REPRO_ROOT/panel"
export BASE_DIR="$REPRO_ROOT/full_jobs"
export CANDIDATE_DIR="$REPRO_ROOT/adaptive_candidates_v3_t010"
export SELECTION_DIR="$REPRO_ROOT/selected_fixed_route_v2"

test ! -e "$REPRO_ROOT"
mkdir "$REPRO_ROOT"
```

Sections 2–6 are not reproducible from the GitHub checkout alone. They require
the deliberately unpublished BF16 source blocks, the original selection
results, base reports and generated candidate alternatives, the external
PolarLattice repository, and a CuPy-capable GPU environment. The checkout by
itself supports the complete source-free verification in section 7 and can
regenerate a reconstruction from the published bundle when the decoder runtime
dependencies are installed.

## 1. Verify frozen inputs

```bash
cd "$RUN_ROOT"
sha256sum \
  panel/manifest.json \
  panel/source_ledger.json \
  panel/side.bin \
  full_jobs/run.receipt.json \
  base_clean_scores/score.receipt.json \
  adaptive_candidates_v3_t010/candidate.receipt.json
```

Expected hashes are in
`$REPO_ROOT/evaluation/qwen3_vorpal_v1/ARTIFACT_INVENTORY.json`. Also verify
the immutable checkpoint revision and every source hash in
`panel/source_ledger.json` before exact-source scoring.

## 2. Rebuild the construction path when sources are available

```bash
"$PY" -B "$REPO_ROOT/vorpal/build_continuous_waterfill.py" \
  --results "$RESULTS_JSON" \
  --sources "$SOURCE_DIR" \
  --output-dir "$PANEL_DIR" \
  --lambda-variance 1.8252209629460492e-5 \
  --min-rate 1.45 \
  --max-rate 3.05

"$PY" -B "$REPO_ROOT/vorpal/pack_continuous_side.py" \
  --manifest "$PANEL_DIR/manifest.json" \
  --output "$PANEL_DIR/side.bin" \
  --receipt "$PANEL_DIR/side.receipt.json"

"$PY" -B "$REPO_ROOT/vorpal/run_continuous_panel.py" \
  --manifest "$PANEL_DIR/manifest.json" \
  --repo "$REPO_ROOT" \
  --encoder "$REPO_ROOT/plte/agent_root_polar_lattice_gate.py" \
  --polar-repo "$POLAR_REPO" \
  --python "$PY" \
  --output-dir "$BASE_DIR" \
  --workers 8
```

The build and score paths use CuPy for the large group/statistics and exact SSE
accumulations. A reproduced construction should match the frozen hashes before
being treated as equivalent evidence.

## 3. Reproduce the complete adaptive candidate scan

```bash
"$PY" -B \
  "$REPO_ROOT/vorpal/adaptive_candidate_audit_v3/run_adaptive_candidates.py" \
  --manifest "$PANEL_DIR/manifest.json" \
  --base-dir "$BASE_DIR" \
  --base-receipt "$BASE_DIR/run.receipt.json" \
  --output-dir "$CANDIDATE_DIR" \
  --encoder "$REPO_ROOT/plte/agent_root_polar_lattice_gate.py" \
  --repacker "$REPO_ROOT/vorpal/adaptive_candidate_audit_v3/repack_tail_prefixes.py" \
  --scorer "$REPO_ROOT/vorpal/decode_continuous_chunk.py" \
  --decoder "$REPO_ROOT/plte/agent_polar_codec_audit_independent_decoder.py" \
  --raw-mask "$REPO_ROOT/plte/agent_root_polar_escape_frozen_profiles.bin" \
  --repo "$REPO_ROOT" \
  --polar-repo "$POLAR_REPO" \
  --python "$PY" \
  --trigger-gap-db 0.10 \
  --tail-ranking raw-gain \
  --tail-ks 1 3 7 15 30 60 120 240 480 \
  --workers 8
```

Require format `continuous PLTE all-base adaptive candidate receipt v3`,
status `complete`, exactly 400 scanned indices, 38 triggers, 342 tails, 38
upgrades, literal-boolean hardening flags, physical container hashes, and the
frozen candidate receipt hash.

## 4. Run exact fixed-route selection over the frozen candidate set

First verify the reconciled selector core and wrapper hashes from the inventory.
The wrapper must pin the same selector core; a stale transition pair must fail.

```bash
"$PY" -B \
  "$REPO_ROOT/vorpal/outer_v2_fixed_route/select_continuous_adaptive_fixed_route_v2.py" \
  --manifest "$PANEL_DIR/manifest.json" \
  --base-dir "$BASE_DIR" \
  --candidate-receipt "$CANDIDATE_DIR/candidate.receipt.json" \
  --base-total-raw-sse 1961.1814347120044 \
  --total-raw-energy 65382.85877511848 \
  --packer "$REPO_ROOT/vorpal/pack_continuous_side.py" \
  --bundle-packer "$REPO_ROOT/vorpal/outer_v2_fixed_route/pack_bundle_fixed_route_v2.py" \
  --raw-mask "$REPO_ROOT/plte/agent_root_polar_escape_frozen_profiles.bin" \
  --python "$PY" \
  --output-dir "$SELECTION_DIR" \
  --max-bpw 2.5 \
  --physical-selection fixed-route
```

The passed receipt format must be
`continuous PLTE exact adaptive selection receipt fixed-route v2`. Validate its
checksum, input hashes, exact Decimal arithmetic, one selected option per
chunk, 400 staged container hashes, and exact physical bundle reparse.

## 5. Decode without sources or construction manifest

```bash
"$PY" -B \
  "$REPO_ROOT/vorpal/outer_v2_fixed_route/outer_decode_fixed_route_v2.py" \
  --bundle "$SELECTION_DIR/selected.wfouter" \
  --clean-decoder "$REPO_ROOT/plte/agent_polar_codec_audit_independent_decoder.py" \
  --reconstruction "$SELECTION_DIR/reconstruction.f64" \
  --receipt "$SELECTION_DIR/decode.receipt.json" \
  --reconstruction-dtype f64 \
  --workers 8
```

Reject trailing compressed data, nonzero padding, source/manifest access,
wrong implementation hashes, a count other than 400, or bytes after the last
container.

## 6. Score separately against exact frozen sources

```bash
"$PY" -B \
  "$REPO_ROOT/vorpal/outer_v2_fixed_route/evaluate_sources_fixed_route_v2.py" \
  --decode-receipt "$SELECTION_DIR/decode.receipt.json" \
  --reconstruction "$SELECTION_DIR/reconstruction.f64" \
  --source-ledger "$REPO_ROOT/evaluation/qwen3_vorpal_v1/source_ledger.json" \
  --expected-source-ledger-sha256 ceaba045a4ed368901e52ad12716d7c56cdf15e9e55c5f0e7a0dbbc2318eebfc \
  --source-root "$SOURCE_ROOT" \
  --bundle "$SELECTION_DIR/selected.wfouter" \
  --output "$SELECTION_DIR/evaluation.json"
```

Verify the resulting literal files against the frozen hashes in
`$REPO_ROOT/evaluation/qwen3_vorpal_v1/ARTIFACT_INVENTORY.json`. Recompute `B`, `R`,
`SSE`, `E`, `D`, `D_G`, and signed gap. The release passes only if `R < 2.5`
and `gap_dB <= -0.10` both hold without rounding.

## 7. Verify the source-free publication

The verifier uses only the Python standard library and deliberately does not
read model weights, BF16 sources, NumPy, CuPy, or encoder modules:

```bash
"$PY" "$REPO_ROOT/vorpal/publication_verifier/verify_vorpal_publication.py" \
  "$REPO_ROOT/evaluation/qwen3_vorpal_v1" --compact

"$PY" "$REPO_ROOT/vorpal/publication_verifier/test_verify_vorpal_publication.py"
```

The GitHub tree omits the 838,860,800-byte reconstruction. Its SHA-256 remains
cross-bound by the decode and evaluation receipts, and it can be regenerated
from the decoder-self-contained panel bundle. When a local
`reconstruction.f64` is placed in the artifact directory, the same verifier
hashes it directly.
