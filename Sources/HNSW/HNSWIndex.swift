//
//  HNSWIndex.swift
//
//  Copyright © 2024 Noah Kamara.
//

import CHNSWLib
import Foundation

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

/// A Swift wrapper for the HNSW (Hierarchical Navigable Small World) index.
/// HNSW is an efficient approximate nearest neighbor search algorithm that uses a hierarchical
/// graph structure.
public final class HNSWIndex {
    private let index: UnsafeMutableRawPointer

    deinit {
        hnswlib_free_index(index)
    }

    /// Creates a new HNSW index with the specified parameters.
    /// - Parameters:
    ///   - dimension: The dimensionality of the vectors to be indexed
    ///   - maxElements: The maximum number of elements that can be stored in the index
    ///   - M: The maximum number of outgoing connections in the graph (default: 16)
    ///   - efConstruction: The construction time/accuracy trade-off parameter (default: 200)
    ///   - space: The space type to use for distance calculations (default: .l2)
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

    /// The space type of the index (currently supports "l2" and "cosine")
    public var space: HNSWSpaceType {
        HNSWSpaceType(cValue: hnswlib_get_space_type(self.index))
    }

    /// The dimensionality of the space
    public var dimension: Int {
        Int(hnswlib_get_dim(self.index))
    }

    /// The maximum number of outgoing connections in the graph
    public var M: Int {
        Int(hnswlib_get_M(self.index))
    }

    /// The construction time/accuracy trade-off parameter
    public var efConstruction: Int {
        Int(hnswlib_get_ef_construction(self.index))
    }

    /// The current capacity of the index
    public var maxElements: Int {
        Int(hnswlib_get_max_elements(self.index))
    }

    /// The current number of elements in the index
    public var elementCount: Int {
        Int(hnswlib_get_current_count(self.index))
    }

    /// Rejects negative labels before calling into hnswlib (native code uses unsigned labels and only checks `id >= max_elements`).
    private func requireValidLabelID(_ id: Int32) throws(HNSWError) {
        guard id >= 0 else {
            throw HNSWError.invalidLabel(id: Int(id))
        }
    }

    /// Searches for k nearest neighbors of a query vector.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find
    /// - Returns: The k nearest neighbors as ``HNSWSearchResult`` values ordered by increasing distance (best match first).
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
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
    /// The predicate receives each candidate’s external label id (the same id passed to ``addPoint``).
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

    /// Adds a vector to the index with the specified ID and metadata.
    /// - Parameters:
    ///   - vector: The vector to add (array of floats)
    ///   - id: Non-negative integer ID to associate with the vector (hnswlib stores labels as unsigned).
    ///   - metadata: Optional metadata string to associate with the vector
    /// - Throws: An error if the vector dimension doesn't match the index dimension or if the add operation fails
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

    /// Gets the metadata associated with a vector ID.
    /// - Parameter id: Non-negative label ID
    /// - Returns: The metadata string, or nil if no metadata exists
    /// - Throws: ``HNSWError/invalidLabel(id:)`` if `id` is negative
    public func getMetadata(for id: Int32) throws(HNSWError) -> String? {
        try self.requireValidLabelID(id)
        guard let metadata = hnswlib_get_metadata(self.index, id) else {
            return nil
        }
        return String(cString: metadata)
    }

    /// Sets or updates the metadata for a vector ID.
    /// - Parameters:
    ///   - metadata: The metadata string to associate with the vector
    ///   - id: Non-negative label ID
    /// - Throws: ``HNSWError/invalidLabel(id:)`` if `id` is negative
    public func setMetadata(_ metadata: String?, for id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        hnswlib_set_metadata(self.index, id, metadata)
    }

    /// Removes the metadata associated with a vector ID.
    /// - Parameter id: Non-negative label ID
    /// - Throws: ``HNSWError/invalidLabel(id:)`` if `id` is negative
    public func removeMetadata(for id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        hnswlib_remove_metadata(self.index, id)
    }

    /// Marks an element as deleted, so it will be omitted from search results.
    /// - Parameter id: Non-negative label ID
    /// - Throws: An error if the element is already deleted
    public func markDeleted(_ id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        let result = hnswlib_mark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to mark element as deleted")
        }
    }

    /// Unmarks an element as deleted, so it will be included in search results.
    /// - Parameter id: Non-negative label ID
    /// - Throws: An error if the element is not deleted
    public func unmarkDeleted(_ id: Int32) throws(HNSWError) {
        try self.requireValidLabelID(id)
        let result = hnswlib_unmark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to unmark element")
        }
    }

    // MARK: Settings

    /// Changes the maximum capacity of the index.
    /// - Parameter newSize: The new maximum capacity
    /// - Throws: An error if the resize operation fails
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

    /// Sets the query time accuracy/speed trade-off parameter.
    /// - Parameter ef: The ef parameter value
    public func setEf(_ ef: Int32) {
        hnswlib_set_ef(self.index, ef)
    }

    // MARK: Persistence

    /// Saves the index to a file.
    /// - Parameter path: The path where to save the index
    /// - Throws: An error if the file cannot be written
    public func saveIndex(to path: String) throws {
        let result = hnswlib_save_index(self.index, path)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to save index")
        }
    }

    /// Loads an index from a file.
    /// - Parameters:
    ///   - path: The path to the index file
    ///   - maxElements: The maximum number of elements that can be stored in the index
    /// - Throws: An error if the file cannot be read
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
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }
}

private func utf8String(from data: Data) throws -> String {
    guard let string = String(data: data, encoding: .utf8) else {
        throw HNSWError.jsonEncodedMetadataNotUTF8
    }
    return string
}

extension HNSWIndex {
    /// Gets the decoded JSON metadata associated with a vector ID.
    /// - Parameter id: The ID of the vector
    /// - Returns: The metadata string, or nil if no metadata exists
    public func getJSONMetadata<T: Decodable>(
        for id: Int32,
        as type: T.Type,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> T? {
        guard let rawMetadata = try self.getMetadata(for: id)?.data(using: .utf8) else {
            return nil
        }
        
        return try decoder.decode(T.self, from: rawMetadata)
    }


    /// Gets the decoded JSON metadata associated with a vector ID.
    /// - Parameter id: The ID of the vector
    /// - Returns: The metadata string, or nil if no metadata exists
    public func setJSONMetadata<T: Encodable>(
        _ metadata: T,
        for id: Int32,
        encoder: JSONEncoder = JSONEncoder()
    ) throws {
        let rawMetadata = try encoder.encode(metadata)
        try self.setMetadata(try utf8String(from: rawMetadata), for: id)
    }

    
    /// Adds a vector to the index with the specified ID and metadata.
    /// - Parameters:
    ///   - vector: The vector to add (array of floats)
    ///   - id: The integer ID to associate with the vector
    ///   - metadata: Optional metadata string to associate with the vector
    /// - Throws: An error if the vector dimension doesn't match the index dimension or if the add operation fails
    public func addPoint<T: Encodable>(
        _ vector: [Float],
        id: Int32,
        jsonMetadata: T,
        encoder: JSONEncoder = JSONEncoder()
    ) throws {
        let rawMetadata = try encoder.encode(jsonMetadata)
        try self.addPoint(vector, id: id, metadata: try utf8String(from: rawMetadata))
    }
}
