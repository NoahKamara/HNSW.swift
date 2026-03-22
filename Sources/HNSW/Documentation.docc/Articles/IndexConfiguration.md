# Index configuration

Choose vector dimension, capacity, graph connectivity, build-time breadth, and distance space when you create an ``HNSWIndex``.

## Overview

### Constructor parameters

| Parameter | Role |
|-----------|------|
| `dimension` | Length of every vector passed to ``HNSWIndex/addPoint(_:id:metadata:)`` and search. |
| `maxElements` | Maximum number of labeled points the index can hold before ``HNSWIndex/resizeIndex(to:)``. |
| `M` | Maximum number of outgoing graph edges per node (default `16`). Higher values often improve recall at the cost of memory and build time. |
| `efConstruction` | Candidate list size while inserting points (default `200`). Larger values usually yield a better graph and higher recall, with slower inserts. |
| `space` | ``HNSWSpaceType/l2`` (Euclidean) or ``HNSWSpaceType/cosine`` (cosine similarity via normalized vectors). |

### Space types

- **L2**: Distance is Euclidean. Vectors are stored and compared as provided (no automatic normalization).
- **Cosine**: The wrapper **normalizes** vectors on insert and normalizes query vectors before search so distances align with cosine-related behavior in the underlying library.

Pick one space at creation time. If you ``HNSWIndex/loadIndex(from:maxElements:)``, the loaded index must match the space type of the instance you load into; otherwise ``HNSWError/spaceMismatch(expected:actual:)`` is thrown.

### Query-time accuracy: `ef`

Construction parameters affect index quality. At query time, ``HNSWIndex/setEf(_:)`` controls how many candidates are explored during a search. Higher `ef` generally improves recall (especially for hard queries or filtered search) at the cost of latency. See <doc:AddingAndSearching> and <doc:FilteredSearch>.

### Capacity and growth

`maxElements` is fixed at initialization unless you call ``HNSWIndex/resizeIndex(to:)``. Label IDs must fit within the index’s element capacity rules enforced by the native layer; see ``HNSWError/idExceedsMaxElements(maxElements:attemptedId:)``.
