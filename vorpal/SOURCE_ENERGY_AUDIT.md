# Adaptive V3 source-energy discrepancy audit

Audited input (read-only):

`/root/negative_gap_root/continuous_v1/adaptive_candidates_v3_t010/candidate.receipt.json`

The receipt is complete and contains 38 triggered A64 rows. For each row the
selector compares one base, one A128 upgrade, and nine tail prefixes.

## Exact census

- Total base-to-option energy comparisons: 380.
- Exact comparisons: 348.
- Non-exact comparisons: 32, all A128 upgrades.
- A128 upgrades: 38 total; 6 exact and 32 non-exact.
- Tails: 342 total; all 342 exactly equal the base energy.
- Maximum absolute discrepancy: `1.2e-13` (chunk 332).
- Maximum relative discrepancy:
  `5.745119905955487263109594632e-16` (chunk 332).
- Maximum binary64 distance: 4 ULP.

Binary64 ULP census:

- 0 ULP: chunks `13, 87, 105, 155, 283, 284`.
- 1 ULP: chunks `98, 131, 150, 182, 201, 249, 260, 268, 292, 307`.
- 2 ULP: chunks `31, 72, 74, 95, 103, 106, 124, 180, 206, 224, 250, 315, 335`.
- 3 ULP: chunks `41, 50, 286, 299, 313, 326, 336`.
- 4 ULP: chunks `11, 332`.

Every non-exact row:

| Chunk | Base energy | A128 energy | Absolute | Relative | ULP |
|---:|---:|---:|---:|---:|---:|
| 11 | 48.66798296516522 | 48.667982965165194 | 2.6e-14 | 5.342321258435933e-16 | 4 |
| 31 | 85.4387699392164 | 85.43876993921637 | 3e-14 | 3.511286506271434e-16 | 2 |
| 41 | 97.91787512097861 | 97.91787512097856 | 5e-14 | 5.106319958253225e-16 | 3 |
| 50 | 105.78189622916645 | 105.7818962291664 | 5e-14 | 4.726706722261789e-16 | 3 |
| 72 | 118.47932492026601 | 118.47932492026598 | 3e-14 | 2.532087351121332e-16 | 2 |
| 74 | 118.83037932925059 | 118.83037932925056 | 3e-14 | 2.524606937160166e-16 | 2 |
| 95 | 126.26504003927255 | 126.26504003927252 | 3e-14 | 2.375954578612498e-16 | 2 |
| 98 | 127.59037695830021 | 127.5903769583002 | 1e-14 | 7.837581672219885e-17 | 1 |
| 103 | 129.35346252000852 | 129.35346252000846 | 6e-14 | 4.638453337939767e-16 | 2 |
| 106 | 129.37769954202236 | 129.3776995420223 | 6e-14 | 4.637584391467076e-16 | 2 |
| 124 | 135.0100141297628 | 135.01001412976274 | 6e-14 | 4.444114785613749e-16 | 2 |
| 131 | 135.88680705905725 | 135.88680705905722 | 3e-14 | 2.207719840452342e-16 | 1 |
| 150 | 140.13075043104286 | 140.13075043104283 | 3e-14 | 2.140857728066099e-16 | 1 |
| 180 | 148.33861710485115 | 148.3386171048511 | 5e-14 | 3.370666450574915e-16 | 2 |
| 182 | 148.92251314336974 | 148.9225131433697 | 4e-14 | 2.685960581493240e-16 | 1 |
| 201 | 152.99386419167345 | 152.99386419167348 | 3e-14 | 1.960862950844581e-16 | 1 |
| 206 | 155.29155568589806 | 155.291555685898 | 6e-14 | 3.863700104940643e-16 | 2 |
| 224 | 160.538545862148 | 160.53854586214794 | 6e-14 | 3.737420173938855e-16 | 2 |
| 249 | 168.47445814667444 | 168.4744581466744 | 4e-14 | 2.374247137520149e-16 | 1 |
| 250 | 169.3840007927712 | 169.38400079277113 | 7e-14 | 4.132621715886840e-16 | 2 |
| 260 | 172.3314595329255 | 172.33145953292546 | 4e-14 | 2.321108409829120e-16 | 1 |
| 268 | 176.21980928154926 | 176.21980928154923 | 3e-14 | 1.702419275239852e-16 | 1 |
| 286 | 186.45227251040734 | 186.45227251040743 | 9e-14 | 4.826972543066020e-16 | 3 |
| 292 | 187.53821375807507 | 187.5382137580751 | 3e-14 | 1.599673975710364e-16 | 1 |
| 299 | 191.54437236784898 | 191.5443723678489 | 8e-14 | 4.176577939150570e-16 | 3 |
| 307 | 194.44543302867595 | 194.44543302867598 | 3e-14 | 1.542849298783774e-16 | 1 |
| 313 | 200.4854681168253 | 200.4854681168254 | 1e-13 | 4.987892685654840e-16 | 3 |
| 315 | 200.45055945585736 | 200.45055945585742 | 6e-14 | 2.993256799226495e-16 | 2 |
| 326 | 206.8319805245491 | 206.83198052454918 | 8e-14 | 3.867873807382736e-16 | 3 |
| 332 | 208.87292513356596 | 208.87292513356584 | 1.2e-13 | 5.745119905955487e-16 | 4 |
| 335 | 212.37004373103292 | 212.37004373103287 | 5e-14 | 2.354381019166955e-16 | 2 |
| 336 | 213.10204297066434 | 213.10204297066426 | 8e-14 | 3.754070063561653e-16 | 3 |

## Provenance audit

There were 2,778 independent checks and zero failures. These include:

- 38 canonical normalized-source file hashes;
- 534 original raw-source file sizes and 534 SHA-256 hashes;
- exact equality of all 38 tail-report and A128-decode raw-source lists;
- 76 encoder-report source hashes and 76 source paths;
- 76 physical container hashes and 76 report-to-container hashes;
- all 266 A128 decode bindings for manifest, metadata, container, scorer,
  decoder, raw mask, and normalized source;
- all 38 decode-to-row energy and SSE bindings;
- all 38 tail manifest, normalized-source, base report, base container, decoder,
  mask, base energy, and base SSE bindings;
- 342 tail-container hashes and 342 exact tail-to-base energy comparisons.

The discrepancy is explained by reduction order, not different sources. The
base/tail repacker concatenates the raw coordinates and performs one CuPy
float64 sum. The independent clean decoder performs one CuPy float64 sum per
group and adds the resulting Python floats. These mathematically equivalent
orders differ by at most four binary64 ULPs on the frozen panel.

## Fail-closed selector policy

The base/tail single reduction is canonical. The A128 independent value is
accepted only after all exact source and implementation bindings pass and only
when both conditions hold:

```text
binary64 ULP distance <= 4
relative Decimal error <= 1e-15
```

All energy fields must be literal finite positive JSON decimals. Tail energies
and the decode-receipt-to-candidate A128 energy remain exact comparisons.
Regression tests accept exactly four ULPs and reject five, and reject strings,
integers, booleans, Python floats, NaN, and positive/negative infinity.
