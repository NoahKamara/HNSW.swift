# Concurrency with `HNSWContainer`

Share one index across concurrent tasks safely using the package’s `actor` wrapper.

## Overview

``HNSWIndex`` is a class and is **not** `Sendable`. The underlying native index is not documented as safe for concurrent use without synchronization.

### Actor wrapper

``HNSWContainer`` is a `public actor` that owns a single ``HNSWIndex``. Call its `perform` methods to run a closure with exclusive access to the index on the actor’s executor.

Use this pattern when multiple concurrent tasks need to share one logical index.

### Example

```swift
import HNSW

let container = HNSWContainer(dimension: 128, maxElements: 50_000)

await container.perform { index in
    try index.addPoint(vector, id: 1)
}

let neighbors = await container.perform { index in
    try index.searchKnn(query, maxResults: 10)
}
```

### `reset`

``HNSWContainer/reset()`` replaces the inner index with a fresh instance using the same dimension, capacity, `M`, `efConstruction`, and space. It does not delete files on disk.

### When not to use the actor

If the index is already confined to a single thread or serial queue, calling ``HNSWIndex`` directly avoids actor hopping. Choose based on your app’s concurrency model.
