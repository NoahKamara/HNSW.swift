//
//  HNSWError.swift
//  HNSW
//
//  Created by Noah Kamara on 27.04.2025.
//

public enum HNSWError: Error, Equatable {
    case indexNotInitialized
    /// Label IDs must be non-negative: hnswlib stores labels as unsigned integers; negative values are rejected here so callers get a clear error instead of misclassification from the native `id >= max_elements` check alone.
    case invalidLabel(id: Int)
    case idExceedsMaxElements(maxElements: Int, attemptedId: Int)
    case pointAlreadyExists(id: Int)
    case vectorMismatch(expected: Int, actual: Int)
    case spaceMismatch(expected: HNSWSpaceType, actual: HNSWSpaceType)
    /// `JSONEncoder` output could not be read back as UTF-8 (should not occur for standard encoders).
    case jsonEncodedMetadataNotUTF8
    case generalError(message: String)
    
    public var errorDescription: String {
        switch self {
        case .indexNotInitialized:
            return "Index is not initialized"
        case .invalidLabel(let id):
            return "Label ID must be non-negative (hnswlib uses unsigned labels); got \(id)"
        case .idExceedsMaxElements(let max, let id):
            return "ID \(id) exceeds maximum elements (\(max))"
        case .pointAlreadyExists(let id):
            return "Point with ID \(id) already exists"
        case .vectorMismatch(let expected, let actual):
            return "Vector dimension mismatch: expected \(expected), got \(actual)"
        case .spaceMismatch(let expected, let actual):
            return "Space type mismatch after load: expected \(expected), index reports \(actual)"
        case .jsonEncodedMetadataNotUTF8:
            return "Encoded JSON metadata is not valid UTF-8"
        case .generalError(let message):
            return message
        }
    }
}

#if canImport(Foundation)
import Foundation
extension HNSWError: LocalizedError {}
#endif
