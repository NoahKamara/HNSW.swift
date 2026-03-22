# ``HNSW``

Approximate nearest neighbor search in Swift using an HNSW (Hierarchical Navigable Small World) index backed by [hnswlib](https://github.com/nmslib/hnswlib).

## Overview

The ``HNSW`` module exposes ``HNSWIndex``, a thin Swift wrapper around a native HNSW index. You store high-dimensional vectors under integer **labels**, run **k**-nearest neighbor queries, optionally attach **metadata** per label, and save or load indexes from disk.

Search is **approximate**: results are usually very close to the true nearest neighbors, with quality controlled by index build parameters and query-time settings.

For concurrent use from multiple tasks, prefer ``HNSWContainer``, an actor that serializes access to an ``HNSWIndex``. The index type itself is not `Sendable`; do not share one instance across concurrency domains without external synchronization.

## Topics

### Guides

- <doc:GettingStarted>
- <doc:IndexConfiguration>
- <doc:AddingAndSearching>
- <doc:MetadataAndJSON>
- <doc:FilteredSearch>
- <doc:DeletionAndLifecycle>
- <doc:SavingAndLoadingIndexes>
- <doc:ConcurrencyWithContainer>


### Essentials

- ``HNSWIndex``
- ``HNSWContainer``
- ``HNSWSearchResult``
- ``HNSWSpaceType``

### Indexing

- ``HNSWIndex/addPoint(_:id:metadata:)``
- ``HNSWIndex/addPoint(_:id:jsonMetadata:encoder:)``
- ``HNSWIndex/resizeIndex(to:)``

### Search

- ``HNSWIndex/searchKnn(_:maxResults:)``
- ``HNSWIndex/searchKnn(_:maxResults:labelFilter:)``
- ``HNSWIndex/searchKnn(_:maxResults:filter:)``
- ``HNSWIndex/setEf(_:)``

### Metadata

- ``HNSWIndex/getMetadata(for:)``
- ``HNSWIndex/setMetadata(_:for:)``
- ``HNSWIndex/removeMetadata(for:)``
- ``HNSWIndex/getJSONMetadata(for:as:decoder:)``
- ``HNSWIndex/setJSONMetadata(_:for:encoder:)``

### Maintenance

- ``HNSWIndex/markDeleted(_:)``
- ``HNSWIndex/unmarkDeleted(_:)``

### Saving and loading

- ``HNSWIndex/saveIndex(to:)``
- ``HNSWIndex/loadIndex(from:maxElements:)``

### Errors

- ``HNSWError``
