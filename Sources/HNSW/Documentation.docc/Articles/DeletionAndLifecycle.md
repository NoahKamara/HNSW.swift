# Deletion, capacity, and lifecycle

Soft-delete labels, resize capacity, follow label rules, and reset an actor-wrapped index.

## Overview

### Soft delete

``HNSWIndex/markDeleted(_:)`` marks a label so it is omitted from search results without necessarily reclaiming all underlying graph storage immediately. ``HNSWIndex/unmarkDeleted(_:)`` restores a previously deleted label.

Errors from these operations surface as ``HNSWError/generalError(message:)`` when the native layer reports failure.

### Resizing capacity

``HNSWIndex/resizeIndex(to:)`` grows (or adjusts) the maximum number of elements. The new capacity must be at least the current ``HNSWIndex/elementCount``. The implementation verifies that element count and reported capacity stay consistent after the native resize.

### Label rules

Label ids must be **non-negative**. Negative values produce ``HNSWError/invalidLabel(id:)`` on operations that validate labels in Swift before calling into hnswlib.

### Querying label state

``HNSWIndex/labelExists(id:)`` returns `true` if the external label is still registered in the index, **including** points that are only soft-deleted (``markDeleted(_:)`` does not remove the label from the native lookup map).

``HNSWIndex/isLabelActive(id:)`` returns `true` only when the label exists **and** is not soft-deleted—i.e. it can show up in an ordinary unfiltered ``searchKnn(_:maxResults:ef:)``.

Together with ``HNSWIndex/maxElements`` (same notion as “max element capacity”), these APIs support reconciliation: scan candidate labels or slots, drop rows that are not active in HNSW, and soft-delete HNSW labels that no longer exist in your authoritative store, then persist.

```swift
// Example: after loading HNSW and SQLite, reconcile one label id
let id: Int32 = 42
if try index.isLabelActive(id: id) {
    // ensure SQLite has a row for this embedding
} else if try index.labelExists(id: id) {
    // present but soft-deleted — optional cleanup in your app store
} else {
    // not in the index
}
```

### When to recreate an index

``HNSWContainer/reset()`` drops the wrapped index and creates a new one with the same configuration parameters. Use this when you want a clean graph without reloading from disk.
