# Saving and loading indexes

Persist an index to disk and load it back with package metadata checks and capacity parameters.

## Overview

### Save

``HNSWIndex/saveIndex(to:)`` writes the native hnswlib binary snapshot to a file path and writes package sidecars next to it. The required wrapper metadata sidecar records the package format version, space type, and dimension; the label metadata sidecar records any metadata strings. On failure, ``HNSWError/generalError(message:)`` is thrown.

### Load

``HNSWIndex/loadIndex(from:maxElements:)`` reads an index from disk into an existing ``HNSWIndex`` instance. You must pass a `maxElements` value consistent with how you intend to use the index (native layer enforces capacity).

Before loading the native graph, the wrapper reads the required metadata sidecar and verifies that its saved ``HNSWIndex/space`` and dimension match the receiving instance. Space mismatches throw ``HNSWError/spaceMismatch(expected:actual:)`` and dimension mismatches throw ``HNSWError/vectorMismatch(expected:actual:)``. Missing or invalid wrapper metadata is a load failure.

### Paths

Use filesystem paths appropriate for your platform. Ensure the process has read/write permission and that parent directories exist for writes.

### Versioning and compatibility

Indexes are binary artifacts from hnswlib plus package sidecars. Compatibility across library versions or different build configurations is not guaranteed by this Swift package—treat saved files as opaque to your app and version them with your embedding model or schema.
