# Saving and loading indexes

Persist an index to disk and load it back with space-type checks and capacity parameters.

## Overview

### Save

``HNSWIndex/saveIndex(to:)`` writes the index to a file path. On failure, ``HNSWError/generalError(message:)`` is thrown.

### Load

``HNSWIndex/loadIndex(from:maxElements:)`` reads an index from disk into an existing ``HNSWIndex`` instance. You must pass a `maxElements` value consistent with how you intend to use the index (native layer enforces capacity).

After load, the wrapper compares the index’s reported ``HNSWIndex/space`` with the space type **before** load. If they differ, ``HNSWError/spaceMismatch(expected:actual:)`` is thrown so you do not silently query with the wrong metric.

### Paths

Use filesystem paths appropriate for your platform. Ensure the process has read/write permission and that parent directories exist for writes.

### Versioning and compatibility

Indexes are binary artifacts from hnswlib. Compatibility across library versions or different build configurations is not guaranteed by this Swift package—treat saved files as opaque to your app and version them with your embedding model or schema.
