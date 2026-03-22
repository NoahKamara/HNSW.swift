# Adding points and searching

Insert vectors under labels, query approximate k-nearest neighbors, and tune query-time behavior.

## Overview

### Inserting vectors

Use ``HNSWIndex/addPoint(_:id:metadata:)`` (or the JSON convenience overload) to insert a vector under a non-negative **label** `id`. The vector length must equal ``HNSWIndex/dimension``.

If you add the same label twice, ``HNSWError/pointAlreadyExists(id:)`` is thrown.

### k-nearest neighbors

``HNSWIndex/searchKnn(_:maxResults:)`` returns up to `maxResults` neighbors, ordered by **increasing distance** (closest first). Fewer than `k` results are returned when the index contains fewer points or when the graph cannot fill the result buffer.

### Query-time `ef`

Before searching, set ``HNSWIndex/setEf(_:)`` to a value appropriate for your latency and recall targets. There is no single correct value: start from a modest setting, measure recall or user-visible quality on your data, then increase `ef` if results are too noisy or unstable.

### Cosine space

If ``HNSWIndex/space`` is ``HNSWSpaceType/cosine``, query vectors are normalized inside the wrapper the same way as stored vectors.

### Filtered search

To restrict candidates by label or metadata **during** the graph search (not “search then filter”), use the overloads documented in <doc:FilteredSearch>.
