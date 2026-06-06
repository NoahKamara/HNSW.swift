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
    let metadata: String? = if let metaPtr = hnswlib_get_metadata(box.index, labelId) {
        String(cString: metaPtr)
    } else {
        nil
    }
    return box.predicate(metadata)
}

private let hnswLoadNativeFailure: Int32 = -1
private let hnswLoadMissingWrapperMetadata: Int32 = -2
private let hnswLoadInvalidWrapperMetadata: Int32 = -3
private let hnswLoadSpaceMismatch: Int32 = -4
private let hnswLoadDimensionMismatch: Int32 = -5

/// A Swift wrapper around an HNSW (Hierarchical Navigable Small World) index backed by hnswlib.
///
/// Store fixed-length vectors under non-negative integer labels, query *k* approximate nearest
/// neighbors, attach optional metadata, and persist to disk. Search quality and latency depend on
/// construction parameters (``HNSWIndex/M``, ``HNSWIndex/efConstruction``) and per-search `ef`.
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

    /// Whether the index can reuse soft-deleted slots when inserting with `replaceDeleted: true`.
    ///
    /// Set at initialization (it must be known before the native graph is built). When `true`, calling
    /// ``markDeleted(_:)`` makes a slot eligible for reuse, and a later ``addPoint(_:id:metadata:replaceDeleted:)``
    /// with `replaceDeleted: true` rewrites that slot in place instead of growing the index.
    public let allowReplaceDeleted: Bool

    deinit {
        hnswlib_free_index(index)
    }

    /// Creates an empty index with the given vector dimension, capacity, graph connectivity, and distance space.
    /// - Parameters:
    ///   - dimension: Length of every vector for ``HNSWIndex/addPoint(_:id:metadata:)`` and search; must match your
    /// embedding size.
    ///   - maxElements: Upper bound on stored labels before ``resizeIndex(to:)`` is required.
    ///   - M: Maximum outgoing edges per node (default `16`); affects recall, memory, and build cost.
    ///   - efConstruction: Build-time candidate list size (default `200`); larger values usually improve graph quality
    /// at slower inserts.
    ///   - space: ``HNSWSpaceType/l2`` or ``HNSWSpaceType/cosine``; cosine applies normalization in this wrapper.
    ///   - allowReplaceDeleted: When `true`, soft-deleted slots become eligible for reuse so high-churn workloads can
    /// insert with `replaceDeleted: true` instead of allocating new slots (see
    /// ``addPoint(_:id:metadata:replaceDeleted:)``).
    public init(
        dimension: Int,
        // Number of edges per node in the index graph.
        // Larger the value - more accurate the search, more space required.
        maxElements: Int,
        M: Int = 16,
        efConstruction: Int = 200,
        space: HNSWSpaceType = .l2,
        allowReplaceDeleted: Bool = false
    ) {
        self.allowReplaceDeleted = allowReplaceDeleted
        self.index = hnswlib_create_index(
            Int32(dimension),
            Int32(maxElements),
            Int32(M),
            Int32(efConstruction),
            space.cValue,
            allowReplaceDeleted
        )
    }

    /// The metric space selected at initialization.
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

    /// Rejects negative labels before calling into hnswlib (native code uses unsigned labels and only checks `id >=
    /// max_elements`).
    private func requireValidLabelID(_ id: Int32) throws(HNSWError) {
        guard id >= 0 else {
            throw HNSWError.invalidLabel(id: Int(id))
        }
    }

    /// Performs unfiltered approximate k-nearest neighbor search for `query`.
    /// - Parameters:
    ///   - query: Query vector whose length must equal ``dimension``.
    ///   - maxResults: Maximum neighbors to return (`k`); may be fewer if the index is sparse.
    ///   - ef: Per-query candidate list size (higher → usually better recall, slower).
    /// - Returns: ``HNSWSearchResult`` values ordered by increasing ``HNSWSearchResult/distance`` (best match first).
    /// - Throws: ``HNSWError/vectorMismatch(expected:actual:)`` when `query.count` ≠ ``dimension``.
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int
    ) throws(HNSWError) -> [HNSWSearchResult] {
        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)
        let resultCount = try self.searchKnn(query, maxResults: maxResults, ef: ef, ids: &ids, distances: &distances)

        return zip(ids, distances)
            .prefix(resultCount)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }

    /// Performs unfiltered k-nearest neighbor search and writes results into caller-provided buffers.
    ///
    /// Use this overload on high-QPS paths to reuse `ids` and `distances` arrays across searches. The arrays are grown
    /// to `maxResults` when needed, but otherwise retain their storage. Only the first returned-count entries are
    /// valid.
    /// - Parameter ef: Per-query candidate list size (higher → usually better recall, slower).
    /// - Returns: Number of valid entries written to `ids` and `distances`.
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        ids: inout [Int32],
        distances: inout [Float]
    ) throws(HNSWError) -> Int {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? self.normalize(query) : query
        return normalizedQuery.withUnsafeBufferPointer { queryPtr in
            self.searchKnn(queryPtr.baseAddress, maxResults: maxResults, ef: ef, ids: &ids, distances: &distances)
        }
    }

    /// Performs unfiltered search for a query that is already normalized for cosine indexes.
    ///
    /// For ``HNSWSpaceType/cosine``, `query` must already have unit length. This skips the wrapper’s normalization
    /// copy.
    /// For ``HNSWSpaceType/l2``, this is equivalent to ``searchKnn(_:maxResults:ef:)``.
    public func searchKnnNormalized(
        _ query: [Float],
        maxResults: Int,
        ef: Int
    ) throws(HNSWError) -> [HNSWSearchResult] {
        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)
        let resultCount = try self.searchKnnNormalized(
            query,
            maxResults: maxResults,
            ef: ef,
            ids: &ids,
            distances: &distances
        )

        return zip(ids, distances)
            .prefix(resultCount)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }

    /// Performs unfiltered search for an already-normalized query and writes results into reusable buffers.
    ///
    /// For ``HNSWSpaceType/cosine``, `query` must already have unit length. Only the first returned-count entries are
    /// valid.
    /// - Returns: Number of valid entries written to `ids` and `distances`.
    public func searchKnnNormalized(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        ids: inout [Int32],
        distances: inout [Float]
    ) throws(HNSWError) -> Int {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        return query.withUnsafeBufferPointer { queryPtr in
            self.searchKnn(queryPtr.baseAddress, maxResults: maxResults, ef: ef, ids: &ids, distances: &distances)
        }
    }

    /// Searches for up to `k` nearest neighbors among labels enabled in a dense native allowlist.
    ///
    /// Unlike Swift closure filters, this filter is evaluated entirely in C++ during graph search. Use
    /// ``HNSWLabelAllowlist`` when labels are bounded and you can maintain/reuse the allowlist bytes outside the query
    /// loop.
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        allowlist: HNSWLabelAllowlist
    ) throws(HNSWError) -> [HNSWSearchResult] {
        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)
        let resultCount = try self.searchKnn(
            query,
            maxResults: maxResults,
            ef: ef,
            allowlist: allowlist,
            ids: &ids,
            distances: &distances
        )

        return zip(ids, distances)
            .prefix(resultCount)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }

    /// Native allowlist filtered search that writes results into caller-provided buffers.
    ///
    /// The arrays are grown to `maxResults` when needed. Only the first returned-count entries are valid.
    /// - Returns: Number of valid entries written to `ids` and `distances`.
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        allowlist: HNSWLabelAllowlist,
        ids: inout [Int32],
        distances: inout [Float]
    ) throws(HNSWError) -> Int {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? self.normalize(query) : query
        return normalizedQuery.withUnsafeBufferPointer { queryPtr in
            allowlist.storage.withUnsafeBufferPointer { allowlistPtr in
                self.searchKnn(
                    queryPtr.baseAddress,
                    maxResults: maxResults,
                    ef: ef,
                    allowlist: allowlistPtr.baseAddress,
                    allowlistCount: allowlist.storage.count,
                    ids: &ids,
                    distances: &distances
                )
            }
        }
    }

    /// Searches for up to `k` nearest neighbors among labels that pass `filter`, using hnswlib’s
    /// filtered graph search (not “take `k` unfiltered then drop”).
    ///
    /// The predicate receives each candidate’s external label id (the same id passed to
    /// ``HNSWIndex/addPoint(_:id:metadata:)``).
    /// Use this to filter against your own payload store or allowlists without storing strings in the index.
    /// Named `labelFilter` (not `filter`) so it does not clash with ``searchKnn(_:maxResults:filter:)``’s metadata
    /// predicate.
    /// Marked disfavored so a trailing closure without a label (e.g. `{ _ in true }`) resolves to the metadata overload
    /// when both could match.
    ///
    /// For highly selective filters, increase `ef` so the search explores enough candidates to fill `k` results when
    /// that many matches exist.
    ///
    /// This method is not thread-safe: do not call it concurrently on the same index instance.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find (`k`)
    ///   - ef: Per-query candidate list size (higher → usually better recall, slower).
    ///   - labelFilter: Return `true` to allow the label in results.
    /// - Returns: Neighbors that pass the filter, ordered by increasing distance (best match first); fewer than `k`
    /// when fewer than `k` matches exist.
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    @_disfavoredOverload
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        labelFilter: @escaping (Int32) -> Bool
    ) throws(HNSWError) -> [HNSWSearchResult] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? self.normalize(query) : query

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        let box = LabelFilterBox(labelFilter)
        let userData = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<LabelFilterBox>.fromOpaque(userData).release() }

        let resultCount = normalizedQuery.withUnsafeBufferPointer { queryPtr in
            ids.withUnsafeMutableBufferPointer { idsPtr in
                distances.withUnsafeMutableBufferPointer { distancesPtr in
                    Int(hnswlib_search_knn_with_label_filter(
                        self.index,
                        queryPtr.baseAddress,
                        idsPtr.baseAddress,
                        distancesPtr.baseAddress,
                        Int32(maxResults),
                        Int32(ef),
                        userData,
                        hnswLabelFilterTrampoline
                    ))
                }
            }
        }

        return zip(ids, distances)
            .prefix(resultCount)
            .map { HNSWSearchResult(id: $0, distance: $1) }
    }

    /// Searches for up to `k` nearest neighbors among labels that pass `filter`, using hnswlib’s
    /// filtered graph search (not “take `k` unfiltered then drop”).
    ///
    /// For highly selective filters, increase `ef` so the search explores enough candidates to fill `k` results when
    /// that many matches exist.
    ///
    /// This method is not thread-safe: do not call it concurrently on the same index instance.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find (`k`)
    ///   - ef: Per-query candidate list size (higher → usually better recall, slower).
    ///   - filter: Receives the stored metadata string, or `nil` when none is stored; `nil` is not
    ///     implicitly excluded—the predicate decides, same as filtering an unfiltered search using
    ///     ``getMetadata(for:)``. Return `true` to allow the label.
    /// - Returns: IDs and distances of neighbors that pass the filter, ordered by increasing distance (best match
    /// first); fewer than `k` when fewer than `k` matches exist.
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        ef: Int,
        filter: @escaping (String?) -> Bool
    ) throws(HNSWError) -> [(id: Int, distance: Float)] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        let normalizedQuery = self.space == .cosine ? self.normalize(query) : query

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        let box = MetadataStringFilterBox(index: self.index, predicate: filter)
        let userData = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<MetadataStringFilterBox>.fromOpaque(userData).release() }

        let resultCount = normalizedQuery.withUnsafeBufferPointer { queryPtr in
            ids.withUnsafeMutableBufferPointer { idsPtr in
                distances.withUnsafeMutableBufferPointer { distancesPtr in
                    Int(hnswlib_search_knn_with_label_filter(
                        self.index,
                        queryPtr.baseAddress,
                        idsPtr.baseAddress,
                        distancesPtr.baseAddress,
                        Int32(maxResults),
                        Int32(ef),
                        userData,
                        hnswMetadataStringFilterTrampoline
                    ))
                }
            }
        }

        return zip(ids, distances)
            .prefix(resultCount)
            .map { (id: Int($0), distance: $1) }
    }

    /// Inserts `vector` under label `id`, optionally storing a metadata string for filtered search and helpers.
    /// - Parameters:
    ///   - vector: Values whose length must equal ``dimension``; cosine space normalizes a copy before storage.
    ///   - id: Non-negative external label; must be unique and within capacity rules enforced by the native index.
    ///   - metadata: Optional opaque string, or `nil` for no metadata.
    ///   - replaceDeleted: When `true`, reuse a previously ``markDeleted(_:)`` slot instead of growing the index. This
    /// lets a saturated index (``elementCount`` at ``maxElements``) accept new points without ``resizeIndex(to:)``.
    /// Requires the index to have been created with `allowReplaceDeleted: true`. The reused slot belonged to some other
    /// previously soft-deleted label chosen by the native layer; as elsewhere in this wrapper, metadata is keyed by
    /// external label and is not auto-cleared, so manage stale entries via ``removeMetadata(for:)`` if needed.
    /// - Throws: ``HNSWError/vectorMismatch(expected:actual:)``, ``HNSWError/invalidLabel(id:)``,
    /// ``HNSWError/replaceDeletedNotEnabled``,
    /// ``HNSWError/pointAlreadyExists(id:)``, ``HNSWError/idExceedsMaxElements(maxElements:attemptedId:)``, or other
    /// ``HNSWError`` cases from the native layer.
    public func addPoint(_ vector: [Float], id: Int32, metadata: String? = nil, replaceDeleted: Bool = false) throws {
        guard vector.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: vector.count)
        }
        try self.requireValidLabelID(id)
        try self.requireReplaceDeletedAllowed(replaceDeleted)

        let normalizedVector = self.space == .cosine ? self.normalize(vector) : vector
        try self.addPointNative(normalizedVector, id: id, metadata: metadata, replaceDeleted: replaceDeleted)
    }

    /// Inserts a vector that is already normalized for cosine indexes.
    ///
    /// For ``HNSWSpaceType/cosine``, `vector` must already have unit length. This skips the wrapper’s normalization
    /// copy on insert. For ``HNSWSpaceType/l2``, this is equivalent to ``addPoint(_:id:metadata:replaceDeleted:)``.
    /// See ``addPoint(_:id:metadata:replaceDeleted:)`` for the `replaceDeleted` semantics.
    public func addNormalizedPoint(
        _ vector: [Float],
        id: Int32,
        metadata: String? = nil,
        replaceDeleted: Bool = false
    ) throws {
        guard vector.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: vector.count)
        }
        try self.requireValidLabelID(id)
        try self.requireReplaceDeletedAllowed(replaceDeleted)

        try self.addPointNative(vector, id: id, metadata: metadata, replaceDeleted: replaceDeleted)
    }

    /// Default thread count for ``addPoints`` and ``searchKnnBatch`` when `numThreads` is `nil` or non-positive.
    ///
    /// `-1` means use hardware concurrency. This value is stored in the native wrapper.
    public var numThreads: Int {
        get { Int(hnswlib_get_num_threads(self.index)) }
        set { hnswlib_set_num_threads(self.index, Int32(newValue)) }
    }

    /// Inserts many vectors in one native call, optionally using multiple threads.
    ///
    /// Vectors are passed as a row-major flat buffer (`vectors.count` must equal `ids.count * dimension`).
    /// For ``HNSWSpaceType/cosine``, vectors are normalized in native code unless you use
    /// ``addNormalizedPoints(vectors:ids:replaceDeleted:numThreads:)``.
    ///
    /// Metadata is not supported on this path; use ``addPoint(_:id:metadata:)`` when you need per-label strings.
    public func addPoints(
        vectors: [Float],
        ids: [Int32],
        replaceDeleted: Bool = false,
        numThreads: Int? = nil
    ) throws {
        try self.addPoints(
            vectors: vectors,
            ids: ids,
            replaceDeleted: replaceDeleted,
            vectorsAreNormalized: false,
            numThreads: numThreads
        )
    }

    /// Inserts many pre-normalized vectors for cosine indexes in one native call.
    public func addNormalizedPoints(
        vectors: [Float],
        ids: [Int32],
        replaceDeleted: Bool = false,
        numThreads: Int? = nil
    ) throws {
        try self.addPoints(
            vectors: vectors,
            ids: ids,
            replaceDeleted: replaceDeleted,
            vectorsAreNormalized: true,
            numThreads: numThreads
        )
    }

    /// Inserts many vectors from a nested array in one native call.
    public func addPoints(
        _ vectors: [[Float]],
        ids: [Int32],
        replaceDeleted: Bool = false,
        numThreads: Int? = nil
    ) throws {
        guard vectors.count == ids.count else {
            throw HNSWError.generalError(message: "Vector count (\(vectors.count)) must match id count (\(ids.count))")
        }
        var flat = [Float]()
        flat.reserveCapacity(vectors.count * self.dimension)
        for vector in vectors {
            guard vector.count == self.dimension else {
                throw HNSWError.vectorMismatch(expected: self.dimension, actual: vector.count)
            }
            flat.append(contentsOf: vector)
        }
        try self.addPoints(
            vectors: flat,
            ids: ids,
            replaceDeleted: replaceDeleted,
            numThreads: numThreads
        )
    }

    /// Batch k-nearest neighbor search over many queries in one native call.
    ///
    /// Returns one result array per query, each ordered by increasing distance. For cosine indexes, queries are
    /// normalized in native code unless you use ``searchKnnBatchNormalized(queries:maxResults:ef:numThreads:)``.
    public func searchKnnBatch(
        queries: [Float],
        queryCount: Int,
        maxResults: Int,
        ef: Int,
        numThreads: Int? = nil
    ) throws(HNSWError) -> [[HNSWSearchResult]] {
        try self.searchKnnBatch(
            queries: queries,
            queryCount: queryCount,
            maxResults: maxResults,
            ef: ef,
            queriesAreNormalized: false,
            numThreads: numThreads
        )
    }

    /// Batch search for queries that are already normalized for cosine indexes.
    public func searchKnnBatchNormalized(
        queries: [Float],
        queryCount: Int,
        maxResults: Int,
        ef: Int,
        numThreads: Int? = nil
    ) throws(HNSWError) -> [[HNSWSearchResult]] {
        try self.searchKnnBatch(
            queries: queries,
            queryCount: queryCount,
            maxResults: maxResults,
            ef: ef,
            queriesAreNormalized: true,
            numThreads: numThreads
        )
    }

    /// Batch k-NN search from nested query vectors.
    public func searchKnnBatch(
        _ queries: [[Float]],
        maxResults: Int,
        ef: Int,
        numThreads: Int? = nil
    ) throws(HNSWError) -> [[HNSWSearchResult]] {
        guard !queries.isEmpty else { return [] }
        var flat = [Float]()
        flat.reserveCapacity(queries.count * self.dimension)
        for query in queries {
            guard query.count == self.dimension else {
                throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
            }
            flat.append(contentsOf: query)
        }
        return try self.searchKnnBatch(
            queries: flat,
            queryCount: queries.count,
            maxResults: maxResults,
            ef: ef,
            numThreads: numThreads
        )
    }

    /// Fails fast when `replaceDeleted` is requested on an index that was not created with `allowReplaceDeleted: true`,
    /// surfacing a clear Swift error instead of the native runtime exception.
    private func requireReplaceDeletedAllowed(_ replaceDeleted: Bool) throws(HNSWError) {
        guard !replaceDeleted || self.allowReplaceDeleted else {
            throw HNSWError.replaceDeletedNotEnabled
        }
    }

    private func addPointNative(
        _ vector: [Float],
        id: Int32,
        metadata: String?,
        replaceDeleted: Bool
    ) throws {
        try vector.withUnsafeBufferPointer { ptr in
            let result: Int32 = if let metadata {
                hnswlib_add_point_with_metadata(self.index, ptr.baseAddress, id, metadata, replaceDeleted)
            } else {
                hnswlib_add_point(self.index, ptr.baseAddress, id, replaceDeleted)
            }
            try Self.throwIfAddFailed(result, id: id, maxElements: self.maxElements)
        }
    }

    private func addPoints(
        vectors: [Float],
        ids: [Int32],
        replaceDeleted: Bool,
        vectorsAreNormalized: Bool,
        numThreads: Int?
    ) throws {
        guard vectors.count == ids.count * self.dimension else {
            throw HNSWError.generalError(
                message: "Expected \(ids.count * self.dimension) vector elements, got \(vectors.count)"
            )
        }
        for id in ids {
            try self.requireValidLabelID(id)
        }
        try self.requireReplaceDeletedAllowed(replaceDeleted)

        let threadArg = Int32(numThreads ?? 0)
        let result = vectors.withUnsafeBufferPointer { vectorsPtr in
            ids.withUnsafeBufferPointer { idsPtr in
                hnswlib_add_points(
                    self.index,
                    vectorsPtr.baseAddress,
                    idsPtr.baseAddress,
                    Int32(ids.count),
                    replaceDeleted,
                    vectorsAreNormalized,
                    threadArg
                )
            }
        }
        try Self.throwIfAddFailed(result, id: nil, maxElements: self.maxElements)
    }

    private func searchKnnBatch(
        queries: [Float],
        queryCount: Int,
        maxResults: Int,
        ef: Int,
        queriesAreNormalized: Bool,
        numThreads: Int?
    ) throws(HNSWError) -> [[HNSWSearchResult]] {
        guard queries.count == queryCount * self.dimension else {
            throw HNSWError.generalError(
                message: "Expected \(queryCount * self.dimension) query elements, got \(queries.count)"
            )
        }
        guard queryCount > 0 else { return [] }

        var ids = [Int32](repeating: -1, count: queryCount * maxResults)
        var distances = [Float](repeating: 0, count: queryCount * maxResults)
        let threadArg = Int32(numThreads ?? 0)

        let status = queries.withUnsafeBufferPointer { queriesPtr in
            ids.withUnsafeMutableBufferPointer { idsPtr in
                distances.withUnsafeMutableBufferPointer { distancesPtr in
                    hnswlib_search_knn_batch(
                        self.index,
                        queriesPtr.baseAddress,
                        Int32(queryCount),
                        idsPtr.baseAddress,
                        distancesPtr.baseAddress,
                        Int32(maxResults),
                        Int32(ef),
                        queriesAreNormalized,
                        threadArg
                    )
                }
            }
        }

        guard status == 0 else {
            switch status {
            case -1:
                throw HNSWError.indexNotInitialized
            case -4:
                throw HNSWError.generalError(message: "Batch search failed")
            default:
                throw HNSWError.generalError(message: "Unknown batch search error")
            }
        }

        var output: [[HNSWSearchResult]] = []
        output.reserveCapacity(queryCount)
        for row in 0..<queryCount {
            let base = row * maxResults
            var rowResults: [HNSWSearchResult] = []
            rowResults.reserveCapacity(maxResults)
            for offset in 0..<maxResults {
                let label = ids[base + offset]
                if label < 0 { break }
                rowResults.append(HNSWSearchResult(id: label, distance: distances[base + offset]))
            }
            output.append(rowResults)
        }
        return output
    }

    private static func throwIfAddFailed(_ result: Int32, id: Int32?, maxElements: Int) throws {
        guard result == 0 else {
            switch result {
            case -1:
                throw HNSWError.indexNotInitialized
            case -2:
                let attempted = id.map(Int.init) ?? -1
                throw HNSWError.idExceedsMaxElements(maxElements: maxElements, attemptedId: attempted)
            case -3:
                let existing = id.map(Int.init) ?? -1
                throw HNSWError.pointAlreadyExists(id: existing)
            case -4:
                throw HNSWError.generalError(message: "Failed to add point")
            default:
                throw HNSWError.generalError(message: "Unknown error")
            }
        }
    }

    private func searchKnn(
        _ query: UnsafePointer<Float>?,
        maxResults: Int,
        ef: Int,
        ids: inout [Int32],
        distances: inout [Float]
    ) -> Int {
        if ids.count < maxResults {
            ids = [Int32](repeating: -1, count: maxResults)
        }
        if distances.count < maxResults {
            distances = [Float](repeating: 0, count: maxResults)
        }

        return ids.withUnsafeMutableBufferPointer { idsPtr in
            distances.withUnsafeMutableBufferPointer { distancesPtr in
                Int(hnswlib_search_knn(
                    self.index,
                    query,
                    idsPtr.baseAddress,
                    distancesPtr.baseAddress,
                    Int32(maxResults),
                    Int32(ef)
                ))
            }
        }
    }

    private func searchKnn(
        _ query: UnsafePointer<Float>?,
        maxResults: Int,
        ef: Int,
        allowlist: UnsafePointer<UInt8>?,
        allowlistCount: Int,
        ids: inout [Int32],
        distances: inout [Float]
    ) -> Int {
        if ids.count < maxResults {
            ids = [Int32](repeating: -1, count: maxResults)
        }
        if distances.count < maxResults {
            distances = [Float](repeating: 0, count: maxResults)
        }

        return ids.withUnsafeMutableBufferPointer { idsPtr in
            distances.withUnsafeMutableBufferPointer { distancesPtr in
                Int(hnswlib_search_knn_with_allowlist(
                    self.index,
                    query,
                    idsPtr.baseAddress,
                    distancesPtr.baseAddress,
                    Int32(maxResults),
                    Int32(ef),
                    allowlist,
                    Int32(allowlistCount)
                ))
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
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative, or
    ///   ``HNSWError/labelNotFound(id:)`` when no point exists for `id`.
    public func setMetadata(_ metadata: String?, for id: Int32) throws(HNSWError) {
        guard try self.labelExists(id: id) else {
            throw HNSWError.labelNotFound(id: Int(id))
        }
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
    /// - Throws: ``HNSWError/generalError(message:)`` when the native call fails (for example if the label is not
    /// deletable in the current state).
    public func markDeleted(_ id: Int32) throws(HNSWError) {
        try self.markDeleted(ids: [id])
    }

    /// Soft-deletes many labels in one native call, optionally using multiple threads.
    ///
    /// Fails on the first label that cannot be deleted (missing, already deleted, etc.). Configure default parallelism
    /// via ``numThreads`` or pass `numThreads` per call.
    public func markDeleted(ids: [Int32], numThreads: Int? = nil) throws(HNSWError) {
        guard !ids.isEmpty else { return }
        for id in ids {
            try self.requireValidLabelID(id)
        }

        let threadArg = Int32(numThreads ?? 0)
        let result = ids.withUnsafeBufferPointer { idsPtr in
            hnswlib_mark_deleted_batch(self.index, idsPtr.baseAddress, Int32(ids.count), threadArg)
        }
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to mark elements as deleted")
        }
    }

    /// Restores a previously deleted label to normal search visibility.
    /// - Parameter id: Non-negative label.
    /// - Throws: ``HNSWError/generalError(message:)`` when the native call fails.
    public func unmarkDeleted(_ id: Int32) throws(HNSWError) {
        try self.unmarkDeleted(ids: [id])
    }

    /// Restores many soft-deleted labels in one native call, optionally using multiple threads.
    ///
    /// Not safe when ``allowReplaceDeleted`` is enabled and deleted slots may have been reused via insert APIs.
    public func unmarkDeleted(ids: [Int32], numThreads: Int? = nil) throws(HNSWError) {
        guard !ids.isEmpty else { return }
        for id in ids {
            try self.requireValidLabelID(id)
        }

        let threadArg = Int32(numThreads ?? 0)
        let result = ids.withUnsafeBufferPointer { idsPtr in
            hnswlib_unmark_deleted_batch(self.index, idsPtr.baseAddress, Int32(ids.count), threadArg)
        }
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to unmark elements")
        }
    }

    // MARK: Label queries

    /// Returns whether the given external label is present in the index, including labels that are only soft-deleted.
    ///
    /// Use this when you need to know if a label still has a slot in the graph (for example, to reconcile an external
    /// store with ``maxElements`` capacity). Soft-deleted labels remain in the lookup map until replaced or the index
    /// is
    /// rebuilt; for “would this label appear in an unfiltered search?” use ``isLabelActive(id:)`` instead.
    /// - Parameter id: Non-negative label (same integer as ``addPoint(_:id:metadata:)`` / ``markDeleted(_:)``).
    /// - Returns: `true` if the label exists in the native `label_lookup_`, `false` if it was never added or was fully
    ///   removed.
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative; ``HNSWError/generalError(message:)`` on
    ///   unexpected native failure.
    public func labelExists(id: Int32) throws(HNSWError) -> Bool {
        try self.requireValidLabelID(id)
        let result = hnswlib_label_exists(self.index, id)
        guard result >= 0 else {
            throw HNSWError.generalError(message: "Failed to check label existence")
        }
        return result == 1
    }

    /// Returns whether the label is present and **not** soft-deleted, so it can appear in a normal unfiltered search.
    ///
    /// Equivalent to: label in the lookup map and its internal node is not marked deleted.
    /// - Parameter id: Non-negative label (same integer as ``addPoint(_:id:metadata:)`` / ``markDeleted(_:)``).
    /// - Returns: `false` if the label is absent or soft-deleted; `true` if it is active.
    /// - Throws: ``HNSWError/invalidLabel(id:)`` when `id` is negative; ``HNSWError/generalError(message:)`` on
    ///   unexpected native failure.
    public func isLabelActive(id: Int32) throws(HNSWError) -> Bool {
        try self.requireValidLabelID(id)
        let result = hnswlib_label_is_active(self.index, id)
        guard result >= 0 else {
            throw HNSWError.generalError(message: "Failed to check whether label is active")
        }
        return result == 1
    }

    // MARK: Settings

    /// Updates the native maximum element capacity while preserving the current ``elementCount``.
    /// - Parameter newSize: New capacity; must be ≥ current ``elementCount`` and non-negative.
    /// - Throws: ``HNSWError/generalError(message:)`` if the native resize fails or postconditions are violated.
    public func resizeIndex(to newSize: Int32) throws(HNSWError) {
        precondition(newSize >= 0, "New size must be non-negative")
        precondition(
            newSize >= Int32(self.elementCount),
            "New size must be greater than or equal to current element count"
        )

        // Store current state for verification
        let currentCount = self.elementCount

        let result = hnswlib_resize_index(index, newSize)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to resize index (native code: \(result))")
        }

        // Verify the resize operation maintained the correct state
        guard self.elementCount == currentCount else {
            throw HNSWError.generalError(message: "Element count changed during resize")
        }

        guard self.maxElements == Int(newSize) else {
            throw HNSWError.generalError(message: "Max elements not updated correctly after resize")
        }
    }

    // MARK: Persistence

    /// Writes a binary index snapshot to `path`, plus required package metadata sidecars.
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
    ///   - path: Filesystem path to a file previously written by ``saveIndex(to:)``.
    ///   - maxElements: Capacity bound passed through to the native loader; must suit your workload.
    /// - Throws: ``HNSWError/spaceMismatch(expected:actual:)`` or ``HNSWError/vectorMismatch(expected:actual:)`` when
    /// the package metadata sidecar disagrees with this instance, and ``HNSWError/generalError(message:)`` on missing
    /// or invalid metadata or native load failure.
    public func loadIndex(from path: String, maxElements: Int) throws(HNSWError) {
        let result = hnswlib_load_index(self.index, path, Int32(maxElements))
        guard result == 0 else {
            switch result {
            case hnswLoadNativeFailure:
                throw HNSWError.generalError(message: "Failed to load index")
            case hnswLoadMissingWrapperMetadata:
                throw HNSWError.generalError(message: "Missing index wrapper metadata")
            case hnswLoadInvalidWrapperMetadata:
                throw HNSWError.generalError(message: "Invalid index wrapper metadata")
            case hnswLoadSpaceMismatch:
                throw HNSWError.spaceMismatch(
                    expected: self.space,
                    actual: HNSWSpaceType(cValue: hnswlib_get_last_loaded_space_type(self.index))
                )
            case hnswLoadDimensionMismatch:
                throw HNSWError.vectorMismatch(
                    expected: self.dimension,
                    actual: Int(hnswlib_get_last_loaded_dim(self.index))
                )
            default:
                throw HNSWError.generalError(message: "Unknown load error (native code: \(result))")
            }
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
