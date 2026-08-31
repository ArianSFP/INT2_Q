# Published artifacts and provenance

## Evidence chain

The compact publication bundle preserves this chain:

```text
immutable Qwen revision + source-block SHA
    -> exact encoder implementation SHA
    -> JSON report + literal container SHA
    -> clean-decoder report for eight cases
    -> pinned evidence manifest
    -> conditional ledger and projection
```

The broad evaluation adds a second, source-free chain:

```text
reproducible weight-blind 400-block manifest
    -> 593 immutable source receipts (400 PLTE + 193 rank-one)
    -> original Tier-0 outcomes: 385 fits + 15 preserved failures
    -> deterministic post-hoc reservoir plan
    -> 400 encoder reports and literal containers
    -> 400 clean-decoder audits and execution receipts
    -> four-bit tier map + padded mixed-tier slot image
    -> header-derived extraction, hash, padding, and EOF readback audit
    -> role/layer summaries and strict claim boundary
```

Run both `python tools/verify_repository.py` and `python tools/verify_stratified_evaluation.py` to check every link that does not require the omitted source weights.

## `plte/` implementation files

| File | Purpose | SHA-256 |
|---|---|---|
| `agent_root_polar_lattice_gate.py` | CuPy polar/tail encoder and in-process causal audits | `4d76ba53c88710778085917108b7940517ed14565815fc8437ea4919d7df4bf8` |
| `agent_polar_codec_audit_independent_decoder.py` | Clean NumPy/SciPy decoder | `7589f4be6e784d8e5a0067303da389b6d982430eb84fda52f668808f322c25d9` |
| `agent_router_adaptive_q234.py` | Literal router Q2/Q3/Q4 codec | `632bdde7c74b90820a0b3905ee39223d60eb320a0936c399afa3bd78740a3a97` |
| `agent_root_polar_escape_full_model_ledger.py` | Fail-closed inventory, provenance, budget, and projection builder | `dec59bb01872f57f6f57169224474760dd1ec2551a892404d74a72b7663bdc1d` |
| `agent_root_export_polar_profile.py` | Six-mask exporter | See Git object / publication manifest |
| `agent_root_fetch_qwen_block.py` | Immutable safetensors HTTP range fetcher | See Git object / publication manifest |

The exact encoder and decoder files contain some historical introductory comments and machine-local default paths. They remain byte-for-byte unchanged because the evidence manifest pins their hashes. Public documentation, rather than those comments, defines the current claim boundary.

## Polar evidence

- `agent_root_polar_escape_evidence_manifest.json`: 49 report records / 47 canonical source blocks.
- `agent_root_polar_escape_final_*.json` and `.polar.bin`: 47 final report/container pairs.
- `agent_root_polar_escape_normative_*.json` and `.polar.bin`: two normative duplicates used as stable exemplars.
- `agent_root_polar_escape_*_standalone_decode.json`: eight clean-decoder reports.
- `agent_root_polar_escape_frozen_profiles.bin` and manifest: six raw masks.
- `agent_root_polar_escape_full_model_ledger.json`: conditional budget, exact inventory, measured evidence index, and distortion projection.

Why 49 rows but 47 blocks: the normative embedding block and normative expert-up block duplicate two sources in the final set.

## Router evidence

- `agent_router_adaptive_q4_t0045_all48.bin`: literal 6,488,688-byte all-router Q4 stream.
- `agent_router_adaptive_q4_t0045_all48.json`: per-router result rows and aggregate decoder audit.
- `agent_router_adaptive_q4_all48_audit.json`: corrected checkpoint accounting and mixed projection.

## Stratified evaluation bundle

All paths below are under `evaluation/qwen3_stratified_v1/`.

| Artifact | Purpose |
|---|---|
| `manifest.json` | Reproducible, weight-blind selection of 400 new PLTE blocks and complete router/rank-one inventories |
| `reservoir_plan.json` | Frozen post-hoc tier assignments, original outcome accounting, conditional checkpoint rate arithmetic, and claim boundary |
| `source_hashes.json` | Source-free provenance for 400 PLTE blocks and 193 rank-one tensors; no weight payloads |
| `original_tier0_outcome.json` | Immutable failed original endpoint: 385 Tier-0 fits and 15 recognized cap overflows |
| `original_tier0_failure_logs/*.txt` | The 15 exact first-pass overflow logs; relative paths and individual hashes are enumerated by `original_tier0_outcome.json` |
| `results.json` | ID-sorted encoder reports, offsets, literal hashes, reservoir tiers, all-in map charges, and packed readback record |
| `containers.polar.bin` | Concatenation of the 400 unpadded literal PLTE containers |
| `tier_map.bin` | 200-byte, four-bit-per-block panel tier map; low nibble first in `results.json` order |
| `tiered_slots.bin` | 400 literal containers padded with zeros to their assigned `T_k` boundaries |
| `independent_decodes.json` | 400 clean-decoder audits plus decoder/profile/report/container/source/log receipts |
| `rank1_exact_audit.json` | Exact-BF16 census and literal-copy hash audit for all 193 rank-one tensors |
| `summary.json` | Coverage, energy-weighted metrics, all-in charged gaps, original endpoint, reservoir accounting, router/rank-one summaries, and claim boundary |

The packed readback proves that every tier-map nibble selects the recorded slot size, every header-derived literal prefix matches the independently decoded container hash, arithmetic/tail padding bits and slot padding bytes are zero, offsets are contiguous, and both binary files end exactly where expected.

The amended panel passes its all-in `<0.10 dB` endpoint, but the original fixed-cap endpoint does not. The reservoir was designed after observing those failures. It remains strict PTQ, but this bundle is non-confirmatory engineering evidence and not a whole-checkpoint result.

## Metadata without weight redistribution

The repository includes:

- all 16 safetensors header inventories used for checkpoint accounting;
- 88 range-fetch manifests describing immutable block provenance;
- 593 source-hash receipts for the broad panel and rank-one census;
- no raw `.bf16.bin` source-weight cache.

The materialization helper fetches only the bytes required for a requested replay and validates them against the evidence SHA-256 values.

## Deliberately excluded

- Cached Qwen BF16 weight bytes.
- Complete Qwen safetensors shards.
- The external `PolarLatticeQuantization` checkout, because no upstream license file was detected.
- Exploratory dead ends and unrelated workspace experiments.
- RunPod SSH details, workstation backup scripts, and machine-transfer manifests.
- A deployable compressed checkpoint—none exists yet.

Measured non-router PLTE coverage is now 447 of 116,422 full blocks (`0.383948%`): 47 earlier blocks plus 400 new stratified blocks. The remaining 115,975 blocks and their reservoir tiers have not been measured.

## Key generated-artifact hashes

| Artifact | SHA-256 |
|---|---|
| Evidence manifest | `ef3003874885e82ce4d27f2682c22a4a307598561302389139ef06ca42d8538e` |
| Conditional ledger JSON | `044613b336db84d393509d736ec9d5782a039c3e425f960bce8dbcbfa35ebbff` |
| Frozen six-mask file | `11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6` |
| Router container | `31de0cc8f7a7a97b12c72440f680a22bdd0494f8fe474254257e7f9ea1c9dab6` |
| Normative expert stream | `de7309f23cd8cb9e636cb96a77354a891ec8a0d7b0d7b4c2367fca5cd411c149` |
| Normative embedding stream | `4fa0fc6690142bb89b1af10a020c47114879995cb5f7032cddd50ab76c2f1bc6` |

## Stratified-bundle hashes

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 476,787 | `fe1577391d6d61e5b22c4e6c2fcbfd384881594b09177405f90f16cc304bb9a1` |
| `reservoir_plan.json` | 245,460 | `c3ed0cc41d722d81d623febedd6d28e2cd767be918cc906dfecad087e907c50d` |
| `summary.json` | 76,741 | `20efd24cff825089d52acdb253276ab86cd5a7917e82d855f65c9904114a2e2c` |
| `results.json` | 2,635,291 | `02322f9eecaecaa901c90b99e8cd5056fa495fcdfe0824020ca25608d6e1fec1` |
| `independent_decodes.json` | 1,852,445 | `8fa974a274784f9f0627aa160eda9effcc6a48dd21e8f1406376c03c5d562e69` |
| `source_hashes.json` | 304,912 | `5b653416b016755ab900a42ee04b5f09aa1275d1f6e55b21d7b88fa5c3cfba23` |
| `rank1_exact_audit.json` | 100,009 | `4192fe5e17392a5716ae7460f3de2d9ee9437b067aec4a852b95d9e6455f1256` |
| `original_tier0_outcome.json` | 10,294 | `d4efaec5824c496a9561f5786050ccb10150ea947406300b533981fbaab096e3` |
| `tier_map.bin` | 200 | `4498660239da2a19fd4d3aa3cad6de6c7a6e2b4e3d5eb81101d5c311dcbfc20a` |
| `containers.polar.bin` | 32,497,123 | `f69097dbaceb6c80577297a82bd32fad7d0bd702578dd5fd603643dae53055ab` |
| `tiered_slots.bin` | 32,497,760 | `fadc1e7735a8386dc0babb57870c58ac9886c86405e738078122560855ba14db` |

The 15 failure-log hashes are intentionally stored per file inside `original_tier0_outcome.json`; the verifier checks each copied `.txt` file against both that document and the frozen reservoir plan.

Git provides a second content-addressed layer for the complete public bundle.
