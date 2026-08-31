# Isolated fixed-route outer side codec experiment v2

This directory does not modify the audited v1 outer decoder or bundle packer.
It tests side codec ID 3 inside the existing 168-byte `WFOUTR01` header:

```text
XZ(WFPLTE01 with all 400 profile codes canonicalized to A64)
| 50-byte LSB-first A64/A128 route
```

The header's raw-side length/hash bind the reconstructed literal WFPLTE01 side.
Its compressed-side length/hash bind the complete `XZ || route` payload. The
decoder requires exactly one XZ stream and exact XZ EOF, exactly 50 route bytes
(all 400 bits are meaningful), exactly 400 canonical profile offsets, canonical
A64 codes, routed alphabets only in `{A64,A128}`, and the reconstructed literal
side SHA-256 before parsing or decoding any container.

The shared helper pins the audited v1 outer decoder SHA-256
`15417800e16598b1fefe68b96796b5812b8294c0e53fc58a3092db3f6286b8fa`.
The evaluator separately pins delegated v1 `evaluate_sources.py` SHA-256
`1fa3ba98529860d2e900b89d188f5451bf7b2b63becfb7b89469cf07f9b75f52`.
Receipts record the actual fixed wrapper, codec helper, and pinned v1 hashes as
separate fields; the wrapper never rewrites the v1 module's `__file__`.

Files:

- `fixed_route_codec.py`: shared fail-closed wire logic;
- `pack_bundle_fixed_route_v2.py`: isolated source-free pack/reparse path;
- `outer_decode_fixed_route_v2.py`: isolated independent decoder entry;
- `evaluate_sources_fixed_route_v2.py`: evaluator wrapper that reparses codec 3;
- `select_continuous_adaptive_fixed_route_v2.py`: forces globally exact
  fixed-route adaptive selection;
- `test_fixed_route_v2.py`: adversarial codec and dependency-tamper tests.

The selector invocation is the standard adaptive selector CLI, except the
bundle packer points here; its wrapper forces `--physical-selection fixed-route`.
This mode safely uses the ordinary global byte/SSE Pareto DP because the side
payload length is independent of the A64/A128 route.

Measured on the real 400-profile side: canonical XZ is 33,756 bytes, the route
is 50 bytes, BZ2 mask is 1,248 bytes, and the retained header is 168 bytes. The
physical prelude is therefore 35,222 bytes, 46 bytes above codec-1 literal XZ.
