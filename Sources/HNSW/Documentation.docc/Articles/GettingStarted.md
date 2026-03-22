# Getting started

Add the **HNSW** package, create an index, insert vectors, and run your first k-nearest neighbor query.

## Overview

### Add the package

In your `Package.swift`, add a package dependency pointing at this repository and depend on the **HNSW** product.

### Minimal example

```swift
import HNSW

let index = HNSWIndex(dimension: 3, maxElements: 10_000)

try index.addPoint([0, 0, 1], id: 0)
try index.addPoint([0, 1, 0], id: 1)
try index.addPoint([1, 0, 0], id: 2)

index.setEf(32)
let neighbors = try index.searchKnn([0, 0, 1], maxResults: 2)

for neighbor in neighbors {
    print(neighbor.id, neighbor.distance)
}
```

### Results

``HNSWIndex/searchKnn(_:maxResults:)`` returns ``HNSWSearchResult`` values ordered by **increasing distance** (best match first). The `distance` interpretation depends on ``HNSWSpaceType`` (L2 vs cosine); see <doc:IndexConfiguration>.

### Next steps

- Tune build and query parameters: <doc:IndexConfiguration>
- Learn search and `ef` in depth: <doc:AddingAndSearching>
- Use metadata or JSON payloads: <doc:MetadataAndJSON>
