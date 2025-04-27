//
//  HNSWIndex.swift
//
//  Copyright © 2024 Noah Kamara.
//

import CHNSWLib
import Foundation

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
    public init(
        dimension: Int,
        maxElements: Int,
        M: Int = 16,
        efConstruction: Int = 200
    ) {
        self.index = hnswlib_create_index(
            Int32(dimension),
            Int32(maxElements),
            Int32(M),
            Int32(efConstruction)
        )
    }

    /// The name of the space (currently only "l2" is supported)
    public var space: String {
        String(cString: hnswlib_get_space(self.index))
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

    /// Searches for k nearest neighbors of a query vector.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find
    ///   - filter: Optional closure that takes a metadata string and returns true if the vector should be included in results
    /// - Returns: An array of tuples containing the IDs and distances of the k nearest neighbors
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    public func searchKnn(
        _ query: [Float],
        maxResults: Int
    ) throws(HNSWError) -> [(id: Int, distance: Float)] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        query.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn(self.index, ptr.baseAddress, &ids, &distances, Int32(maxResults))
        }

        let missing = ids
            .reversed()
            .prefix(while: { $0 == -1 })
            .count

        return zip(ids, distances)
            .prefix(maxResults - missing)
            .map { (Int($0), $1) }
    }

    /// Searches for k nearest neighbors of a query vector with metadata filtering.
    /// Note: This method is not thread-safe. Do not use it concurrently from multiple threads.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find
    ///   - filter: A closure that takes a metadata string and returns true if the vector should be included in results
    /// - Returns: An array of tuples containing the IDs and distances of the k nearest neighbors
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    public func searchKnn(
        _ query: [Float],
        maxResults: Int,
        filter: @escaping (String?) -> Bool
    ) throws(HNSWError) -> [(id: Int, distance: Float)] {
        guard query.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: query.count)
        }

        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)

        // Get all results first
        query.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn(self.index, ptr.baseAddress, &ids, &distances, Int32(maxResults))
        }

        // Filter results based on metadata
        var filteredResults: [(id: Int, distance: Float)] = []
        for (id, distance) in zip(ids, distances) {
            guard id != -1 else { continue }
            let metadata = getMetadata(for: id)
            if filter(metadata) {
                filteredResults.append((Int(id), distance))
            }
        }

        return filteredResults
    }

    /// Adds a vector to the index with the specified ID and metadata.
    /// - Parameters:
    ///   - vector: The vector to add (array of floats)
    ///   - id: The integer ID to associate with the vector
    ///   - metadata: Optional metadata string to associate with the vector
    /// - Throws: An error if the vector dimension doesn't match the index dimension or if the add operation fails
    public func addPoint(_ vector: [Float], id: Int32, metadata: String? = nil) throws {
        guard vector.count == self.dimension else {
            throw HNSWError.vectorMismatch(expected: self.dimension, actual: vector.count)
        }

        try vector.withUnsafeBufferPointer { ptr in
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
    /// - Parameter id: The ID of the vector
    /// - Returns: The metadata string, or nil if no metadata exists
    public func getMetadata(for id: Int32) -> String? {
        guard let metadata = hnswlib_get_metadata(self.index, id) else {
            return nil
        }
        return String(cString: metadata)
    }

    /// Sets or updates the metadata for a vector ID.
    /// - Parameters:
    ///   - metadata: The metadata string to associate with the vector
    ///   - id: The ID of the vector
    public func setMetadata(_ metadata: String?, for id: Int32) {
        hnswlib_set_metadata(self.index, id, metadata)
    }

    /// Removes the metadata associated with a vector ID.
    /// - Parameter id: The ID of the vector
    public func removeMetadata(for id: Int32) {
        hnswlib_remove_metadata(self.index, id)
    }

    /// Marks an element as deleted, so it will be omitted from search results.
    /// - Parameter id: The ID of the element to mark as deleted
    /// - Throws: An error if the element is already deleted
    public func markDeleted(_ id: Int32) throws(HNSWError) {
        let result = hnswlib_mark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to mark element as deleted")
        }
    }

    /// Unmarks an element as deleted, so it will be included in search results.
    /// - Parameter id: The ID of the element to unmark
    /// - Throws: An error if the element is not deleted
    public func unmarkDeleted(_ id: Int32) throws(HNSWError) {
        let result = hnswlib_unmark_deleted(index, id)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to unmark element")
        }
    }

    // MARK: Settings

    /// Changes the maximum capacity of the index.
    /// - Parameter newSize: The new maximum capacity
    /// - Throws: An error if the resize operation fails
    public func resizeIndex(to newSize: Int32) throws {
        precondition(newSize >= 0, "New size must be non-negative")
        precondition(newSize >= Int32(elementCount), "New size must be greater than or equal to current element count")
        
        // Store current state for verification
        let currentCount = elementCount
        
        let result = hnswlib_resize_index(index, newSize)
        guard result == 0 else {
            throw NSError(
                domain: "HNSWIndex",
                code: Int(result),
                userInfo: [NSLocalizedDescriptionKey: "Failed to resize index"]
            )
        }
        
        // Verify the resize operation maintained the correct state
        guard elementCount == currentCount else {
            throw NSError(
                domain: "HNSWIndex",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Element count changed during resize"]
            )
        }
        
        guard maxElements == Int(newSize) else {
            throw NSError(
                domain: "HNSWIndex",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Max elements not updated correctly"]
            )
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
        let result = hnswlib_save_index(index, path)
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to save index")
        }
    }

    /// Loads an index from a file.
    /// - Parameters:
    ///   - path: The path to the index file
    ///   - maxElements: The maximum number of elements that can be stored in the index
    /// - Throws: An error if the file cannot be read
    public func loadIndex(from path: String, maxElements: Int) throws {
        let result = hnswlib_load_index(index, path, Int32(maxElements))
        guard result == 0 else {
            throw HNSWError.generalError(message: "Failed to load index")
        }
    }
}
