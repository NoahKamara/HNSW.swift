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

### When to recreate an index

``HNSWContainer/reset()`` drops the wrapped index and creates a new one with the same configuration parameters. Use this when you want a clean graph without reloading from disk.
