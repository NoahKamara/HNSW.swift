# HNSW.swift Performance Report

Date: 2026-06-04
Host target reported by Swift Testing: `arm64e-apple-macos14.0`
Build/test command: `RUN_PERF=1 swift test -c release --filter Performance`

## Summary

The largest improvement came from adding Apple Silicon NEON distance kernels to the vendored hnswlib distance spaces.
The remaining changes reduce overhead around search result handling and filtered-search paths.

| Step                               | `l2_insert_5000` | `l2_search_p95` | `metadata_filtered_search_p95` | `cosine_search_p95` | `allowlist_filtered_search_p95` |
| ---------------------------------- | ---------------: | --------------: | -----------------------------: | ------------------: | ------------------------------: |
| Baseline                           |        336.573ms |         0.050ms |                        0.491ms |             0.043ms |                             n/a |
| NEON L2/IP kernels                 |        198.572ms |         0.025ms |                        0.331ms |             0.026ms |                             n/a |
| Direct result fill/count return    |        199.875ms |         0.026ms |                        0.329ms |             0.025ms |                             n/a |
| Reusable buffers + normalized APIs |        197.841ms |         0.026ms |                        0.327ms |             0.026ms |                             n/a |
| Native dense allowlist filter      |        203.691ms |         0.026ms |                        0.412ms |             0.026ms |                         0.273ms |
| Dense native metadata storage      |        213.287ms |         0.026ms |                        0.323ms |             0.026ms |                         0.274ms |

Notes:

- The native allowlist and metadata-filter suites ran in the same process but separate Swift Testing suites; small p95
  differences between adjacent runs should be treated as noise.
- A final full release test run with `swift test -c release` passed 41 tests. Because of the existing local edit that
  leaves `metadataFilteredSearchP95Budget` enabled without `RUN_PERF=1`, that full run also printed
  `metadata_filtered_search_p95=0.318ms`.

## Changes Measured

1. Added NEON SIMD kernels for ARM targets in `space_l2.h` and `space_ip.h`, enabled through hnswlib's manual
   vectorization dispatch.
2. Changed the C wrapper search functions to return the actual result count and fill Swift result buffers directly
   from hnswlib's priority queue without an intermediate vector.
3. Added reusable-buffer search APIs and normalized-vector APIs:
   - `searchKnn(_:maxResults:ids:distances:)`
   - `searchKnnNormalized(_:maxResults:)`
   - `searchKnnNormalized(_:maxResults:ids:distances:)`
   - `addNormalizedPoint(_:id:metadata:)`
4. Added `HNSWLabelAllowlist` and native allowlist search overloads so filtered search can avoid Swift callbacks.
5. Replaced wrapper metadata storage from `std::unordered_map<int, std::string>` with dense vectors keyed by bounded
   external label IDs.

## Remaining Work

- Batch insert/search APIs are still the next useful API-level optimization for workloads dominated by Swift/C++
  crossing overhead.
- Concurrent read throughput still depends on caller-side synchronization or `HNSWContainer`; a read-only wrapper or
  reader-writer container would be a separate concurrency-focused change.
