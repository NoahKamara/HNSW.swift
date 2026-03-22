//
//  HNSWIndex.swift
//
//  Copyright © 2024 Noah Kamara.
//

import CHNSWLib

private final class LabelFilterBox {
    let predicate: (Int32) -> Bool

    init(_ predicate: @escaping (Int32) -> Bool) {
        self.predicate = predicate
    }
}

private func hnswLabelFilterTrampoline(
    _ userData: UnsafeMutableRawPointer?,
    _ labelId: Int32
) -> Bool {
    let box = Unmanaged<LabelFilterBox>.fromOpaque(userData!).takeUnretainedValue()
    return box.predicate(labelId)
}

private final class MetadataStringFilterBox {
    let index: UnsafeMutableRawPointer
    let predicate: (String?) -> Bool

    init(index: UnsafeMutableRawPointer, predicate: @escaping (String?) -> Bool) {
        self.index = index
        self.predicate = predicate
    }
}

private func hnswMetadataStringFilterTrampoline(
    _ userData: UnsafeMutableRawPointer?,
    _ labelId: Int32
) -> Bool {
    let box = Unmanaged<MetadataStringFilterBox>.fromOpaque(userData!).takeUnretainedValue()
    let metadata: String?
    if let metaPtr = hnswlib_get_metadata(box.index, labelId) {
        metadata = String(cString: metaPtr)
    } else {
        metadata = nil
    }
    return box.predicate(metadata)
}

/// A Swift wrapper around an HNSW (Hierarchical Navigable Small World) index backed by hnswlib.
///
/// Store fixed-length vectors under non-negative integer labels, query *k* approximate nearest
/// neighbors, attach optional metadata, and persist to disk. Search quality and latency depend on
/// construction parameters (``HNSWIndex/M``, ``HNSWIndex/efConstruction``) and query-time
/// ``HNSWIndex/setEf(_:)``.
///
/// ## Thread safety
///
/// Do not call methods on the same instance concurrently unless you provide external synchronization.
/// For Swift concurrency, use ``HNSWContainer`` and its `perform` methods.
///
/// ## See Also
///
/// - <doc:GettingStarted>
/// - <doc:FilteredSearch>
public final class HNSWIndex {
    private let index: UnsafeMutableRawPointer

    deinit {
        hnswlib_free_index(index)
    }

    /// Creates an empty index with the given vector dimension, capacity, graph connectivity, and distance space.
    /// - Parameters:
    ///   - dimension: Length of every vector for ``HNSWIndex/addPoint(_:id:metadata:)`` and search; must match your embedding size.
    ///   - maxElements: Upper bound on stored labels before ``resizeIndex(to:)`` is required.
    ///   - M: Maximum outgoing edges per node (default `16`); affects recall, memory, and build cost.
    ///   - efConstruction: Build-time candidate list size (default `200`); larger values usually improve graph quality at slower inserts.
    ///   - space: ``HNSWSpaceType/l2`` or ``HNSWSpaceType/cosine``; cosine applies normalization in this wrapper.
    public init(
        dimension: Int,
        maxElements: Int,
        M: Int = 16,
        efConstruction: Int = 200,
        space: HNSWSpaceType = .l2
    ) {
        self.index = hnswlib_create_index(
            Int32(dimension),
            Int32(maxElements),
            Int32(M),
            Int32(efConstruction),
            space.cValue
        )
    }

    /// The metric space selected at initialization or reported after load.
    public var space: HNSWSpaceType {
        HNSWSpaceType(cValue: hnswlib_get_space_type(self.index))
    }

    /// Vector length required for every insert and query.
    public var dimension: Int {
        Int(hnswlib_get_dim(self.index))
    }

    /// Maximum graph degree parameter `M` from initialization.
    public var M: Int {
        Int(hnswlib_get_M(self.index))
    }

    /// Build-time `efConstruction` parameter from initialization.
    public var efConstruction: Int {
        Int(hnswlib_get_ef_construction(self.index))
    }

    /// Current maximum number of storable labels (may change after ``resizeIndex(to:)``).
    public var maxElements: Int {
        Int(hnswlib_get_max_elements(self.index))
    }

    /// Number of labels currently present in the index.
    public var elementCount: Int {
        Int(hnswlib_get_current_count(self.index))
    }

    /// Rejects negative labels before calling into hnswlib (native code uses unsigned labels and only checks `id >= max_elements`).
    private func requireValidLabelID(_ id: Int32) throws(HNSWError) {
        guard id >= 0 else {
            throw HNSWError.invalidLabel(id: Int(id))
        }
    }

    /// Performs unfiltered approximate k-nearest neighbor search for `query`.
    /// - Parameters:
    ///   - query: Query vector whose length must equal ``dimension``.
    ///   - maxResults: Maximum neighbors to return (`k`); may be fewer if the index is sparse.
    /// - Returns: ``HNSWSearchResult`` values ordered by increasing ``HNSWSearchResult/distance`` (best match first).
    /// - Throws: ``HNSWError/vectorMismatch(expected:actual:)`` when `query.count` ≠ ``dimension``.
    public func searchKnn(
        _ query: [Float],
        maxResults: Int
    ) throws(HNSWError) -> [HNSWSearchResult] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        // Normalize query vector if using cosine similarity space
        let normalizedQuery = self.space == .cosine ? normalize(query) : query

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        normalizedQuery.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn(self.index, ptr.baseAddress, &ids, &distances, Int32(maxResults))
        }

        let missing = ids
            .reversed()
            .prefix(while: { $0 == -1 })
            .count

        return zip(ids, distances)
            .prefix(maxResults - missing)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }
    
    /// Searches for up to `k` nearest neighbors among labels that pass `filter`, using hnswlib’s
    /// filtered graph search (not “take `k` unfiltered then drop”).
    ///
    /// The predicate receives each candidate’s external label id (the same id passed to ``HNSWIndex/addPoint(_:id:metadata:)``).
    /// Use this to filter against your own payload store or allowlists without storing strings in the index.
    /// Named `labelFilter` (not `filter`) so it does not clash with ``searchKnn(_:maxResults:filter:)``’s metadata predicate.
    /// Marked disfavored so a trailing closure without a label (e.g. `{ _ in true }`) resolves to the metadata overload when both could match.
    ///
    /// For highly selective filters, increase the query-time `ef` parameter via ``setEf(_:)`` so
    /// the search explores enough candidates to fill `k` results when that many matches exist.
    ///
    /// This method is not thread-safe: do not call it concurrently on the same index instance.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find (`k`)
    ///   - labelFilter: Return `true` to allow the label in results.
    /// - Returns: Neighbors that pass the filter, ordered by increasing distance (best match first); fewer than `k` when fewer than `k` matches exist.
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    @_disfavoredOverload
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        labelFilter: @escaping (Int32) -> Bool
    ) throws(HNSWError) -> [HNSWSearchResult] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? normalize(query) : query

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        let box = LabelFilterBox(labelFilter)
        let userData = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<LabelFilterBox>.fromOpaque(userData).release() }

        normalizedQuery.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn_with_label_filter(
                self.index,
                ptr.baseAddress,
                &ids,
                &distances,
                Int32(maxResults),
                userData,
                hnswLabelFilterTrampoline
            )
        }

        let missing = ids
            .reversed()
            .prefix(while: { $0 == -1 })
            .count

        return zip(ids, distances)
            .prefix(maxResults - missing)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }

    /// Searches for up to `k` nearest neighbors among labels that pass `filter`, using hnswlib’s
    /// filtered graph search (not “take `k` unfiltered then drop”).
    ///
    /// For highly selective filters, increase the query-time `ef` parameter via ``setEf(_:)`` so
    /// the search explores enough candidates to fill `k` results when that many matches exist.
    ///
    /// This method is not thread-safe: do not call it concurrently on the same index instance.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find (`k`)
    ///   - filter: Receives the stored metadata string, or `nil` when none is stored; `nil` is not
    ///     implicitly excluded—the predicate decides, same as filtering an unfiltered search using
    ///     ``getMetadata(for:)``. Return `true` to allow the label.
    /// - Returns: IDs and distances of neighbors that pass the filter, ordered by increasing distance (best match first); fewer than `k` when fewer than `k` matches exist.
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        filter: @escaping (String?) -> Bool
    ) throws(HNSWError) -> [(id: Int, distance: Float)] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? normalize(query) : query

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        let box = MetadataStringFilterBox(index: self.index, predicate: filter)
        let userData = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<MetadataStringFilterBox>.fromOpaque(userData).release() }

        normalizedQuery.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn_with_label_filter(
                self.index,
                ptr.baseAddress,
                &ids,
                &distances,
                Int32(maxResults),
                userData,
                hnswMetadataStringFilterTrampoline
            )
        }

        let missing = ids
            .reversed()
            .prefix(while: { $0 == -1 })
            .count

        return zip(ids, distances)
            .prefix(maxResults - missing)
            .map { (id: Int($0), distance: $1) }
    }

    /// Inserts `vector` under label `id`, optionally storing a metadata string for filtered search and helpers.
    /// - Parameters:
    ///   - vector: Values whose length must equal ``dimension``; cosine space normalizes a copy before storage.
    ///   - id: Non-negative external label; must be unique and within capacity rules enforced by the native index.
    ///   - metadata: Optional opaque string, or `nil` for no metadata.
    /// - Throws: ``HNSWError/vectorMismatch(expected:actual:)``, ``HNSWError/invalidLabel(id:)``, ``HNSWError/pointAlreadyExists(id:)``, ``HNSWError/idExceedsMaxElements(maxElements:attemptedId:)``, or other ``HNSWError`` cases from the native layer.
    public func addPoint(_ vector: [Float], id: Int32, metadata: String? = nil) throws {
        guard vector.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: vector.count)
        }
        try self.requireValidLabelID(id)

        // Normalize vector if using cosine similarity space
        let normalizedVector = self.space == .cosine ? normalize(vector) : vector

        try normalizedVector.withUnsafeBufferPointer { ptr in
            let result: Int32
            if let metadata = metadata {
                result = hnswlib_add_point_with_metadata(self.index, ptr.baseAddress, id, metadata)
            } else {
                result = hnswlib_add_point(self.index, ptr.baseAddress, id)
            }
            
            guard result == 0 else {
                switch result {
                case -1:
                    throw HNSWError.indexNotInitialized
                case -2:
                    throw HNSWError.idExceedsMaxElements(maxElements: Int(self.maxElements), attemptedId: Int(id))
                case -3:
                    throw HNSWError.pointAlreadyExists(id: Int(id))
                case -4:
                    throw HNSWError.generalError(message: "Failed to add point")
                default:
                    throw HNSWError.generalError(message: "Unknown error")
                }
            }
        }
    }

    /// Returns the stored metadata string for `id`, or `nil` when none was set.
    /// - Parameter id: Non-negative label.
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative.
    public func getMetadata(for id: Int32) throws(HNSWError) -> String? {
        try self.requireValidLabelID(id)
        guard let metadata = hnswlib_get_metadata(self.index, id) else {
            return nil
        }
        return String(cString: metadata)
    }

    /// Associates metadata with an existing label, replacing any previous string; pass `nil` to clear.
    /// - Parameters:
    ///   - metadata: New metadata value, or `nil` to remove the string association.
    ///   - id: Non-negative label.
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative.
    public func setMetadata(_ metadata: String?, for id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        hnswlib_set_metadata(self.index, id, metadata)
    }

    /// Deletes stored metadata for `id` if present.
    /// - Parameter id: Non-negative label.
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative.
    public func removeMetadata(for id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        hnswlib_remove_metadata(self.index, id)
    }

    /// Soft-deletes `id` so it no longer appears in search results.
    /// - Parameter id: Non-negative label.
    /// - Throws: ``HNSWError/generalError(message:)`` when the native call fails (for example if the label is not deletable in the current state).
    public func markDeleted(_ id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        let result = hnswlib_mark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to mark element as deleted")
        }
    }

    /// Restores a previously deleted label to normal search visibility.
    /// - Parameter id: Non-negative label.
    /// - Throws: ``HNSWError/generalError(message:)`` when the native call fails.
    public func unmarkDeleted(_ id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        let result = hnswlib_unmark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to unmark element")
        }
    }

    // MARK: Settings

    /// Updates the native maximum element capacity while preserving the current ``elementCount``.
    /// - Parameter newSize: New capacity; must be ≥ current ``elementCount`` and non-negative.
    /// - Throws: ``HNSWError/generalError(message:)`` if the native resize fails or postconditions are violated.
    public func resizeIndex(to newSize: Int32) throws(HNSWError) {
        precondition(newSize >= 0, "New size must be non-negative")
        precondition(newSize >= Int32(elementCount), "New size must be greater than or equal to current element count")
        
        // Store current state for verification
        let currentCount = elementCount
        
        let result = hnswlib_resize_index(index, newSize)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to resize index (native code: \(result))")
        }
        
        // Verify the resize operation maintained the correct state
        guard elementCount == currentCount else {
            throw HNSWError.generalError(message: "Element count changed during resize")
        }
        
        guard maxElements == Int(newSize) else {
            throw HNSWError.generalError(message: "Max elements not updated correctly after resize")
        }
    }

    /// Sets query-time `ef`, controlling how many candidates are explored per search (higher → usually better recall, slower).
    /// - Parameter ef: Native `ef` value; tune alongside your data and filters.
    public func setEf(_ ef: Int32) {
        hnswlib_set_ef(self.index, ef)
    }

    // MARK: Persistence

    /// Writes a binary index snapshot to `path`.
    /// - Parameter path: Filesystem path writable by the process.
    /// - Throws: ``HNSWError/generalError(message:)`` on I/O or native serialization failure.
    public func saveIndex(to path: String) throws {
        let result = hnswlib_save_index(self.index, path)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to save index")
        }
    }

    /// Replaces the receiver’s native index with contents loaded from `path`.
    /// - Parameters:
    ///   - path: Filesystem path to a file previously written by ``saveIndex(to:)`` or a compatible hnswlib build.
    ///   - maxElements: Capacity bound passed through to the native loader; must suit your workload.
    /// - Throws: ``HNSWError/generalError(message:)`` on load failure, or ``HNSWError/spaceMismatch(expected:actual:)`` if the file’s space disagrees with this instance’s ``space`` before load.
    public func loadIndex(from path: String, maxElements: Int) throws(HNSWError) {
        // Store space information before loading
        let spaceType = self.space
        
        // Try loading with original maxElements
        let result = hnswlib_load_index(self.index, path, Int32(maxElements))
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to load index")
        }
        
        // Verify space after loading
        let loadedSpace = self.space
        guard loadedSpace == spaceType else {
            throw HNSWError.spaceMismatch(expected: spaceType, actual: loadedSpace)
        }
    }

    /// Normalizes a vector to unit length.
    /// - Parameter vector: The vector to normalize
    /// - Returns: The normalized vector
    private func normalize(_ vector: [Float]) -> [Float] {
        let magnitude = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }
}

