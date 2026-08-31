# Qwen VORPAL role-joint development artifact

This directory contains the exact physical wrapper and verification receipts
for the strict-PTQ result in which the sampled expert-MLP up, down, and gate
role aggregates are each below the declared Gaussian reference.

## Exact endpoint

| Quantity | Value |
|---|---:|
| Frozen panel | 400 blocks, 104,857,600 BF16 values |
| Role coverage corrected by the extension | 48 up + 48 down + 48 gate blocks |
| Physical wrapper | 32,767,988 bytes |
| Physical all-in rate | 2.49999908447265625 bpw |
| Gaussian reference | 0.031250039662224983 |
| Up role gap | **-0.0018119033858966 dB** |
| Down role gap | **-0.0017979875814763 dB** |
| Gate role gap | **-0.0013266518828134 dB** |
| Full-panel corrected gap | **-0.2718238828655344 dB** |

The wrapper SHA-256 is
`86ac8e569c34360cd3d4dbcfade9289b142a21ed0f4c1ebd6c83f0c05defc598`.
It contains the original 32,583,835-byte VORPAL base byte-for-byte, a 224-byte
header, and the 183,929-byte residual extension. The base digest remains
`320320eac95720d79633da57a048a75c10b63259de518f92684a6e84fc7026d2`.

## Files

- `vorpal_joint_sparc4.vjwrap` is the normative all-in physical artifact.
- `joint_sparc4.extension.bin` is the extracted residual section for format
  inspection. Its bytes also occur inside the wrapper and are not charged
  twice.
- `joint_sparc4.receipt.json` is the encoder's strict-PTQ method, allocation,
  source-provenance, accounting, and metric receipt.
- `joint_sparc4.independent-verification.json` records a separately implemented
  no-encoder-import decode and exact Float64 CuPy source replay.
- `joint_sparc4.source-free-verification.json` records the standard-library
  publication check that requires neither weights nor a GPU.
- `joint_sparc4.tamper-tests.json` records fail-closed physical-format tests.
- `audits/` contains independent up- and gate-specific replay receipts and
  repaired-checksum/noncanonical tamper results.
- `ARTIFACT_INVENTORY.json` defines the claim and all immutable dependencies.
- `CHECKSUMS.sha256` binds every file in this release directory except itself.

The implementations are in
[`../../vorpal/role_joint_sparc4/`](../../vorpal/role_joint_sparc4/). The
published base, selected manifest, and exact evaluation are reused without
duplication from [`../qwen3_vorpal_v1/`](../qwen3_vorpal_v1/).

## Source-free verification

From the repository root, run:

```bash
python vorpal/role_joint_sparc4/verify_source_free.py
```

The verifier uses the Python standard library. It checks the complete wrapper,
embedded base identity, header and extension CRCs, all six SHA-256 provenance
bindings, exact EOF and padding, three disjoint 48-block role masks, canonical
Rice streams, all 756 stage records, receipt bindings, physical rate, and all
role/global gap formulae. It also mutates protected fields and requires each
case to fail closed.

This check validates the publication and its evidence chain. It cannot
remeasure distortion because the Qwen source blocks and 800 MiB decoded base
reconstruction are intentionally not committed.

## Full exact-source replay

The normative run used Python 3.12.3, NumPy 2.5.2, CuPy 14.2.0, CUDA 12, and an
NVIDIA RTX A6000. On the preserved RunPod layout the exact command is:

```bash
/root/int2-venv/bin/python \
  /root/vorpal_role_joint_v3_sparc4/verify_joint_sparc4.py \
  --wrapper /root/vorpal_role_joint_v3_sparc4/candidate_v2/vorpal_joint_sparc4.vjwrap \
  --base-bundle /root/negative_gap_root/continuous_v1/selected_fixed_route_v2/selected.wfouter \
  --manifest /root/negative_gap_root/continuous_v1/selected_fixed_route_v2/selected.manifest.json \
  --evaluation /root/negative_gap_root/continuous_v1/final_fixed_route_v2/evaluation.json \
  --reconstruction /root/negative_gap_root/continuous_v1/final_fixed_route_v2/reconstruction.f64 \
  --source-root /root/int2/INT2_Q_stratified_run \
  --experiment-receipt /root/vorpal_role_joint_v3_sparc4/candidate_v2/joint_sparc4.receipt.json \
  --encoder /root/vorpal_role_joint_v3_sparc4/build_joint_sparc4.py \
  --output /root/vorpal_role_joint_v3_sparc4/candidate_v2/joint_sparc4.independent-verification.json
```

The verifier parses the entire extension and applies it to the externally
supplied, wrapper-bound normative base reconstruction before opening any source
weight. It does not rerun the base polar decoder in that invocation; the
published VORPAL base verifier audits that preceding step independently. It
then validates all 144 selected BF16 block hashes and recomputes the three role
SSEs directly. Its final receipt SHA-256 is
`f11ab99fc74ef0438ea9e32fe0db6276ff95052548e0f95060f1f750a4d8fe8b`.

To rebuild the physical wrapper rather than only replay it:

```bash
/root/int2-venv/bin/python \
  /root/vorpal_role_joint_v3_sparc4/build_joint_sparc4.py \
  --manifest /root/negative_gap_root/continuous_v1/selected_fixed_route_v2/selected.manifest.json \
  --evaluation /root/negative_gap_root/continuous_v1/final_fixed_route_v2/evaluation.json \
  --reconstruction /root/negative_gap_root/continuous_v1/final_fixed_route_v2/reconstruction.f64 \
  --source-root /root/int2/INT2_Q_stratified_run \
  --base-bundle /root/negative_gap_root/continuous_v1/selected_fixed_route_v2/selected.wfouter \
  --output-dir /root/vorpal_role_joint_v3_sparc4/candidate_v2_rebuild \
  --max-stages 450
```

The normative encoder SHA-256 is
`a5f36bc1d108280e3e50fa5857652681895a421335bd2944b96692e2a65bea1a`.
The wrapper binds that implementation digest, along with the selected manifest,
evaluation, reconstruction, base, and extension digests.

## Scope

This is strict post-training weight quantization: no retraining, QAT, gradient,
activation calibration, prompt, label, or task objective is used. It is also a
post-hoc development result. The same frozen panel informed the method and
stage allocation.

“Up/down/gate below 0 dB” means three energy-weighted role aggregates at the
common physical wrapper rate. It does not mean every individual block is below
0 dB, and the panel contains one selected expert block per role and layer—not
all experts in the checkpoint. The result is not a whole-checkpoint rate,
perplexity result, task-accuracy result, inference-kernel benchmark, or
definitive SOTA comparison. See
[`../../docs/VORPAL_ROLE_JOINT.md`](../../docs/VORPAL_ROLE_JOINT.md) for the
architecture, formulas, audit chain, and novelty boundary.
