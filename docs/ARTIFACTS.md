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

Run `python tools/verify_repository.py` to check every link that does not require the omitted source weights.

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

## Metadata without weight redistribution

The repository includes:

- all 16 safetensors header inventories used for checkpoint accounting;
- 88 range-fetch manifests describing immutable block provenance;
- no raw `.bf16.bin` source-weight cache.

The materialization helper fetches only the bytes required for a requested replay and validates them against the evidence SHA-256 values.

## Deliberately excluded

- Cached Qwen BF16 weight bytes.
- Complete Qwen safetensors shards.
- The external `PolarLatticeQuantization` checkout, because no upstream license file was detected.
- Exploratory dead ends and unrelated workspace experiments.
- RunPod SSH details, workstation backup scripts, and machine-transfer manifests.
- A deployable compressed checkpoint—none exists yet.

## Key generated-artifact hashes

| Artifact | SHA-256 |
|---|---|
| Evidence manifest | `ef3003874885e82ce4d27f2682c22a4a307598561302389139ef06ca42d8538e` |
| Conditional ledger JSON | `044613b336db84d393509d736ec9d5782a039c3e425f960bce8dbcbfa35ebbff` |
| Frozen six-mask file | `11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6` |
| Router container | `31de0cc8f7a7a97b12c72440f680a22bdd0494f8fe474254257e7f9ea1c9dab6` |
| Normative expert stream | `de7309f23cd8cb9e636cb96a77354a891ec8a0d7b0d7b4c2367fca5cd411c149` |
| Normative embedding stream | `4fa0fc6690142bb89b1af10a020c47114879995cb5f7032cddd50ab76c2f1bc6` |

Git provides a second content-addressed layer for the complete public bundle.
