//
//  HNSWError.swift
//  HNSW
//
//  Created by Noah Kamara on 27.04.2025.
//

/// Errors thrown by ``HNSWIndex`` and related APIs when inputs are invalid or the native index reports failure.
public enum HNSWError: Error, Equatable {
    /// The native index handle is not in a usable state for the requested operation.
    case indexNotInitialized

    /// The label id is negative; hnswlib stores labels as unsigned values, so Swift rejects negatives up front.
    case invalidLabel(id: Int)

    /// The label id is too large for the index’s current maximum element capacity.
    case idExceedsMaxElements(maxElements: Int, attemptedId: Int)

    /// A point with this label was already inserted.
    case pointAlreadyExists(id: Int)

    /// A vector’s length does not match ``HNSWIndex/dimension``.
    case vectorMismatch(expected: Int, actual: Int)

    /// After ``HNSWIndex/loadIndex(from:maxElements:)``, the file’s space type disagrees with the receiver’s ``HNSWIndex/space``.
    case spaceMismatch(expected: HNSWSpaceType, actual: HNSWSpaceType)

    /// Encoded JSON metadata could not be interpreted as UTF-8 (unexpected for standard `JSONEncoder` output).
    case jsonEncodedMetadataNotUTF8

    /// A catch-all for native failures and invariant violations with a short message.
    case generalError(message: String)

    /// Human-readable text for diagnostics; also used when bridging to `LocalizedError`.
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
