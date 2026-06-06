# HNSW.swift Performance Report

Date: 2026-06-06
Host target reported by Swift Testing: `arm64e-apple-macos14.0`
Build/test command: `RUN_PERF=1 swift test -c release --filter Performance`

Harness: one serialized `Performance` suite (no cross-suite parallelism), warmup per benchmark, **median of 3 runs**
(override with `HNSW_PERF_ITERATIONS`), batch native APIs pinned to **`HNSW_PERF_THREADS=4`** (12 logical processors
reported). Values below are medians; the test log prints min/max per metric.

## Summary

The largest improvement came from adding Apple Silicon NEON distance kernels to the vendored hnswlib distance
spaces. Subsequent work reduced Swift/C++ crossing and result-handling overhead. **Batch insert/search APIs** cut
bulk L2 ingest time for 5,000 × 64-d vectors from ~221ms (per-point `addPoint`) to ~69ms (`addPoints` at 4 threads)—
about **3.2×** on this harness. **Batch delete** at 4 threads did not beat a per-label loop here (~0.77ms vs ~0.44ms
for 5,000 soft-deletes); tombstoning is cheap and parallel coordination dominates at this scale.

| Step                               | `l2_insert_5000` | `l2_insert_5000_batch` | `l2_delete_5000` | `l2_delete_5000_batch` | `l2_search_p95` | `l2_batch_search_250` | `metadata_filtered_search_p95` | `cosine_search_p95` | `allowlist_filtered_search_p95` |
| ---------------------------------- | ---------------: | ---------------------: | ---------------: | ---------------------: | --------------: | --------------------: | -----------------------------: | ------------------: | ------------------------------: |
| Baseline                           |        336.573ms |                    n/a |              n/a |                    n/a |         0.050ms |                   n/a |                        0.491ms |             0.043ms |                             n/a |
| NEON L2/IP kernels                 |        198.572ms |                    n/a |              n/a |                    n/a |         0.025ms |                   n/a |                        0.331ms |             0.026ms |                             n/a |
| Direct result fill/count return    |        199.875ms |                    n/a |              n/a |                    n/a |         0.026ms |                   n/a |                        0.329ms |             0.025ms |                             n/a |
| Reusable buffers + normalized APIs |        197.841ms |                    n/a |              n/a |                    n/a |         0.026ms |                   n/a |                        0.327ms |             0.026ms |                             n/a |
| Native dense allowlist filter      |        203.691ms |                    n/a |              n/a |                    n/a |         0.026ms |                   n/a |                        0.412ms |             0.026ms |                         0.273ms |
| Dense native metadata storage      |        213.287ms |                    n/a |              n/a |                    n/a |         0.026ms |                   n/a |                        0.323ms |             0.026ms |                         0.274ms |
| Batch insert/search APIs           |        221.119ms |               42.293ms |              n/a |                    n/a |         0.033ms |               1.040ms |                        0.365ms |             0.029ms |                         0.292ms |
| **Current (stable harness)**       |        221.356ms |               68.769ms |          0.440ms |                0.767ms |         0.029ms |               1.894ms |                        0.401ms |             0.029ms |                         0.310ms |

### Remeasurement detail (2026-06-06, median of 3 runs, `batch_threads=4`)

| Metric | Median | Min | Max |
| ------ | -----: | --: | --: |
| `l2_insert_5000` | 221.356ms | 220.452ms | 224.453ms |
| `l2_insert_5000_batch` | 68.769ms | 68.635ms | 70.827ms |
| `l2_delete_5000` | 0.440ms | 0.324ms | 0.463ms |
| `l2_delete_5000_batch` | 0.767ms | 0.604ms | 0.875ms |
| `l2_search_p95` | 0.029ms | 0.026ms | 0.031ms |
| `l2_batch_search_250` | 1.894ms | 1.750ms | 2.061ms |
| `metadata_filtered_search_p95` | 0.401ms | 0.390ms | 0.412ms |
| `cosine_search_p95` | 0.029ms | 0.028ms | 0.032ms |
| `allowlist_filtered_search_p95` | 0.310ms | 0.298ms | 0.323ms |

An earlier single-run outlier the same day (276ms insert, 0.587ms allowlist) came from **`Performance budgets` and
`Native filter performance` suites starting concurrently**; that split is removed. Batch insert is slower here than
the earlier 42ms row because this harness pins **`HNSW_PERF_THREADS=4`** instead of default hardware concurrency.

Notes:

- **`l2_insert_5000`** uses a loop of ``addPoint``. **`l2_insert_5000_batch`** uses one ``addPoints`` call per timed
  run (fresh index each iteration) with `numThreads=4`.
- **`l2_delete_5000`** / **`l2_delete_5000_batch`** time only the delete phase; index build is outside the timed
  window (fresh 5k index per iteration).
- **`l2_search_p95`** is the median of per-run p95 over 250 ``searchKnn`` calls. **`l2_batch_search_250`** is median
  wall time for one ``searchKnnBatch`` over 250 queries.
- Override harness: `HNSW_PERF_ITERATIONS=5`, `HNSW_PERF_THREADS=8`, etc.
- A full release test run with `swift test -c release` passed **49** tests; performance budgets are gated behind
  `RUN_PERF=1` (single `Performance` suite in `Tests/HNSWTests/PerformanceBudgetTests.swift`).

## Completed work

These optimizations are present in the current source. File references point at where each lives.

1. **Apple Silicon NEON distance kernels.** `space_l2.h` and `space_ip.h` now provide `*SIMD16ExtNEON` /
   `*SIMD4ExtNEON` kernels, dispatched via hnswlib's manual-vectorization path. `hnswlib.h` defines `USE_NEON` when
   `__ARM_NEON` is present, so `arm64` builds use NEON instead of the scalar fallback. This was the single biggest
   win (insert `336.573ms` -> `198.572ms`, L2 search p95 `0.050ms` -> `0.025ms`).
2. **Direct result fill, real count return.** The C wrapper search functions
   (`hnswlib_search_knn`, `hnswlib_search_knn_with_label_filter`, `hnswlib_search_knn_with_allowlist` in
   `Sources/CHNSWLib/hnswlib_wrapper.cpp`) drain hnswlib's priority queue backward straight into the caller's
   `ids`/`distances` buffers and return the actual result count, with no intermediate vector.
3. **Reusable-buffer and normalized-vector APIs** in `Sources/HNSW/HNSWIndex.swift`:
   - `searchKnn(_:maxResults:ids:distances:)`
   - `searchKnnNormalized(_:maxResults:)`
   - `searchKnnNormalized(_:maxResults:ids:distances:)`
   - `addNormalizedPoint(_:id:metadata:)`

   The normalized overloads let callers with pre-normalized cosine embeddings skip the wrapper's normalization copy
   on both insert and query.
4. **Native filtered search.** `HNSWLabelAllowlist` plus the `DenseAllowListFilterFunctor` C++ functor evaluate a
   dense allowlist entirely in C++ during graph search, and a `labelFilter: (Int32) -> Bool` overload passes only the
   integer label across the boundary. The string-metadata filter overload still exists for convenience, but native
   alternatives are now available for hot paths.
5. **Dense native metadata storage.** Wrapper metadata moved from `std::unordered_map<int, std::string>` to dense
   `std::vector<std::string>` + `std::vector<uint8_t>` keyed by bounded external label IDs.
6. **Batch insert/search APIs.** `hnswlib_add_points` and `hnswlib_search_knn_batch` in `Sources/CHNSWLib/` with
   optional multithreading (`hnswlib_set_num_threads`), exposed on ``HNSWIndex`` as ``addPoints``, ``searchKnnBatch``,
   and normalized variants.
7. **Batch delete APIs.** `hnswlib_mark_deleted_batch` and `hnswlib_unmark_deleted_batch`, exposed as
   ``HNSWIndex/markDeleted(ids:numThreads:)`` and ``HNSWIndex/unmarkDeleted(ids:numThreads:)``.
8. **Stable perf harness.** Single serialized `Performance` suite, warmup + median-of-N reporting, pinned batch thread
   count, delete benchmarks time only the delete phase. See `Tests/HNSWTests/PerformanceBudgetTests.swift`.

## Remaining work

- **Concurrent read throughput.** Read traffic still serializes through `HNSWContainer`. A read-only or reader-writer
  container would be a separate concurrency-focused change; per-query `ef` is already required on search calls.
- **Batch APIs with metadata / filtered batch search** for workloads that still need per-label strings or predicate
  evaluation during graph walks.
