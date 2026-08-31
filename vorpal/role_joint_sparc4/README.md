# Role-joint SPARC4 implementation

This directory freezes the encoder, independent exact-source verifier, and
standard-library publication verifier for the VORPAL role-joint residual
artifact.

- `build_joint_sparc4.py` is the CuPy strict-PTQ encoder. The physical wrapper
  binds its exact SHA-256,
  `a5f36bc1d108280e3e50fa5857652681895a421335bd2944b96692e2a65bea1a`.
- `verify_joint_sparc4.py` is the separate CuPy decoder and exact-source
  evaluator. It does not import the encoder. Its frozen SHA-256 is
  `fe98f117c2aeec1db5a0f66c1eb7bd486ba120fb410ef23981e7bf4b1437d483`.
- `verify_joint_sparc4_source_free.py` is the exact standard-library structural
  and receipt verifier used for the publication receipt. Its SHA-256 is
  `4168cf4e7e335943602f96125c8fed00ae59ad251704c323dcfa7f71276fac13`.
- `verify_source_free.py` supplies repository-relative inputs and temporary
  output paths so the published artifact can be checked with one command.
- `audits/` contains independently written role-specific replay tools.

Run the publication check from the repository root:

```bash
python vorpal/role_joint_sparc4/verify_source_free.py
```

It needs no Qwen weights, reconstruction file, NumPy, CuPy, CUDA, or GPU. The
full exact-source replay and build commands are recorded in the
[artifact README](../../evaluation/qwen3_vorpal_role_joint_v1/README.md).

For the architecture, physical format, equations, exact result, and claim
boundary, see [VORPAL_ROLE_JOINT.md](../../docs/VORPAL_ROLE_JOINT.md).
