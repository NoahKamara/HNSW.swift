# Filtered search

Use predicates during graph exploration so approximate search respects label or metadata constraints.

## Overview

HNSW supports **filtered** approximate search: candidates are checked against your predicate as the search explores the graph, rather than taking the top‑k unfiltered neighbors and discarding some afterward.

### Label filter

``HNSWIndex/searchKnn(_:maxResults:labelFilter:)`` invokes your closure with each candidate’s **label id** (the same `id` passed to ``HNSWIndex/addPoint(_:id:metadata:)``). Return `true` to allow the label in the result set.

This overload is marked `@_disfavoredOverload` so a trailing closure without an argument label prefers the metadata overload when both could apply. If you mean label filtering, call the parameter as `labelFilter: { ... }`.

### Metadata filter

``HNSWIndex/searchKnn(_:maxResults:filter:)`` passes the stored metadata string, or `nil` if no metadata exists. You decide whether `nil` matches.

Return type is an array of tuples `(id: Int, distance: Float)` rather than ``HNSWSearchResult`` for this overload.

### Selective predicates and `ef`

Highly selective filters (few passing labels) may return fewer than `k` neighbors unless the search explores enough candidates. Increase ``HNSWIndex/setEf(_:)`` so the underlying search can discover enough matches when they exist.

### Thread safety

Filtered and unfiltered search on the same ``HNSWIndex`` instance are **not** thread-safe. Use one index per serial context, external locking, or ``HNSWContainer`` to serialize calls.
