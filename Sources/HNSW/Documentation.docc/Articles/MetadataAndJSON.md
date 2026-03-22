# Metadata and JSON

Store per-label string metadata, encode `Codable` payloads as JSON, and use metadata in filtered search.

## Overview

### String metadata

Each label can carry an optional metadata string:

- ``HNSWIndex/addPoint(_:id:metadata:)`` stores metadata alongside the vector.
- ``HNSWIndex/getMetadata(for:)``, ``HNSWIndex/setMetadata(_:for:)``, and ``HNSWIndex/removeMetadata(for:)`` read or update that string.

Metadata is stored in the index’s side table; keep strings reasonably small if you care about memory.

### JSON helpers

The ``HNSWIndex`` extensions ``HNSWIndex/getJSONMetadata(for:as:decoder:)`` and ``HNSWIndex/setJSONMetadata(_:for:encoder:)`` encode and decode `Codable` values to UTF-8 JSON strings using the same metadata channel.

``HNSWIndex/addPoint(_:id:jsonMetadata:encoder:)`` is a convenience that encodes metadata once at insert time.

If encoded data is not valid UTF-8 (unexpected for standard `JSONEncoder` output), ``HNSWError/jsonEncodedMetadataNotUTF8`` can be thrown when setting JSON metadata through these helpers.

### Using metadata in search

``HNSWIndex/searchKnn(_:maxResults:filter:)`` evaluates your predicate against the stored metadata string (or `nil` when none exists). The predicate decides whether `nil` is allowed; there is no implicit exclusion. This matches post-filtering semantics you would apply after calling ``HNSWIndex/getMetadata(for:)``.

See <doc:FilteredSearch> for performance notes and thread-safety guidance.
