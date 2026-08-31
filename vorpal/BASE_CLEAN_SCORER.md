# Resumable base clean scorer

`run_base_clean_scores.py` obtains exact raw-source SSE for the continuous
waterfill **base** containers by invoking the existing
`decode_continuous_chunk.py`. It is an exploratory selection aid, not the final
bundle certificate. The selected codec must still pass the independent outer
decoder and `evaluate_sources.py` over the physical self-contained bundle.

The scorer is deliberately read-only with respect to the base artifacts. It
writes only atomic score envelopes, logs, and `score.receipt.json` beneath its
disjoint output directory. A resumed score is accepted only if its manifest,
encoder, clean scorer, clean decoder, mask, source-ledger, encoder-report,
normalized-source, and container hashes still match.

## RunPod invocation

```bash
cd /root/int2/INT2_Q_stratified_run
/root/int2-venv/bin/python \
  /root/negative_gap_root/continuous_v1/run_base_clean_scores.py \
  --manifest /root/negative_gap_root/continuous_v1/panel/manifest.json \
  --expected-manifest-sha256 14f54c728017ab61574487828e0e4881e59df1bec77cbbe1789cee7a03f0c727 \
  --run-receipt /root/negative_gap_root/continuous_v1/full_jobs/run.receipt.json \
  --base-dir /root/negative_gap_root/continuous_v1/full_jobs \
  --repo /root/int2/INT2_Q_stratified_run \
  --chunk-decoder /root/negative_gap_root/continuous_v1/decode_continuous_chunk.py \
  --expected-chunk-decoder-sha256 2bbf9790acd59f461598f0ea35fe6940d1b18589cb27141eafe8d0ee96f01613 \
  --decoder /root/int2/INT2_Q_stratified_run/plte/agent_polar_codec_audit_independent_decoder.py \
  --expected-decoder-sha256 7589f4be6e784d8e5a0067303da389b6d982430eb84fda52f668808f322c25d9 \
  --raw-mask /root/int2/INT2_Q_stratified_run/plte/agent_root_polar_escape_frozen_profiles.bin \
  --expected-raw-mask-sha256 11efea4247aadfb8d30369483a9753921f46f93f8cc2c0e94325538b159b29a6 \
  --expected-encoder-sha256 4d76ba53c88710778085917108b7940517ed14565815fc8437ea4919d7df4bf8 \
  --python /root/int2-venv/bin/python \
  --output-dir /root/negative_gap_root/continuous_v1/base_clean_scores \
  --workers 32 \
  --require-final
```

Omit `--require-final` while the base encoder is still producing chunks. That
invocation scores the immutable ready-job snapshot and exits successfully with
`status: "partial"`. Rerunning the same command validates and resumes existing
score envelopes before decoding newly available chunks.

`status: "complete"` requires all of the following:

- the exact canonical 400-block/400-chunk manifest and all 800 source hashes;
- 400 valid uncapped base reports and containers;
- 400 bound clean scores in canonical order;
- stable inputs before and after the run;
- a complete canonical encoder `run.receipt.json` whose hashes, rows, sizes,
  rates, alphabet census, and roundtrip status cross-check the base artifacts.

Raw energy and SSE are aggregated with `math.fsum` over sorted canonical chunk
indices. Partial aggregates are explicitly diagnostic. Full-panel fields remain
JSON `null` until the complete gate passes.

SIGINT or SIGTERM is cooperative: active chunk decoders finish and publish
atomically, queued chunks are cancelled, and a partial receipt is written. For
a background launch, record the Python PID and use `kill -TERM <pid>`; wait for
the process to exit before relaunching with a different worker count.

## Tests

```bash
cd /root/negative_gap_root/continuous_v1
/root/int2-venv/bin/python -m py_compile \
  run_base_clean_scores.py test_run_base_clean_scores.py
/root/int2-venv/bin/python -m unittest -v test_run_base_clean_scores.py
```

The tests cover container/report tampering, forged SSE, deterministic
aggregation, canonical run-receipt order, atomic subprocess publication,
hash-bound resume, refusal to overwrite an invalid score, and preservation of
the base artifacts.
