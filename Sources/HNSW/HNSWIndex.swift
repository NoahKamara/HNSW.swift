import Foundation
import CHNSWLib

/// A Swift wrapper for the HNSW (Hierarchical Navigable Small World) index.
/// HNSW is an efficient approximate nearest neighbor search algorithm that uses a hierarchical graph structure.
public class HNSWIndex {
    private var index: UnsafeMutableRawPointer?
    private let dimension: Int
    
    deinit {
        if let index = index {
            hnswlib_free_index(index)
        }
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
        self.dimension = dimension
        self.index = hnswlib_create_index(Int32(dimension), Int32(maxElements), Int32(M), Int32(efConstruction))
    }
    
    struct VectorMismatch: Error, LocalizedError {
        let expected: Int
        let actual: Int
        
        var errorDescription: String? {
            "VectorMismatch: expected \(expected) got \(actual)"
        }
    }
    /// Adds a vector to the index with the specified ID.
    /// - Parameters:
    ///   - vector: The vector to add (array of floats)
    ///   - id: The integer ID to associate with the vector
    /// - Throws: An error if the vector dimension doesn't match the index dimension
    public func addPoint(_ vector: [Float], id: Int) throws {
        guard vector.count == dimension else {
            throw VectorMismatch(expected: dimension, actual: vector.count)
        }
        
        vector.withUnsafeBufferPointer { ptr in
            hnswlib_add_point(index, ptr.baseAddress, Int32(id))
        }
    }
    
    /// Searches for k nearest neighbors of a query vector.
    /// - Parameters:
    ///   - query: The query vector (array of floats)
    ///   - maxResults: The maximum number of nearest neighbors to find
    /// - Returns: An array of tuples containing the IDs and distances of the k nearest neighbors
    /// - Throws: An error if the query vector dimension doesn't match the index dimension
    public func searchKnn(_ query: [Float], maxResults: Int) throws -> [(id: Int, distance: Float)] {
        guard query.count == dimension else {
            throw VectorMismatch(expected: dimension, actual: query.count)
        }
        
        var ids = [Int32](repeating: -1, count: maxResults)
        var distances = [Float](repeating: 0, count: maxResults)
        
        query.withUnsafeBufferPointer { ptr in
            hnswlib_search_knn(index, ptr.baseAddress, &ids, &distances, Int32(maxResults))
        }
        
        let missing = ids
            .reversed()
            .prefix(while: { $0 == -1 })
            .count
        
        return zip(ids, distances)
            .prefix(maxResults - missing)
            .map { (Int($0), $1) }
    }
}


