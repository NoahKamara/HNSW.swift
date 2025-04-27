//
//  SwiftUIView.swift
//  HNSW
//
//  Created by Noah Kamara on 27.04.2025.
//

public enum HNSWError: Error {
    case indexNotInitialized
    case idExceedsMaxElements(maxElements: Int, attemptedId: Int)
    case pointAlreadyExists(id: Int)
    case vectorMismatch(expected: Int, actual: Int)
    case generalError(message: String)
    
    public var errorDescription: String {
        switch self {
        case .indexNotInitialized:
            return "Index is not initialized"
        case .idExceedsMaxElements(let max, let id):
            return "ID \(id) exceeds maximum elements (\(max))"
        case .pointAlreadyExists(let id):
            return "Point with ID \(id) already exists"
        case .vectorMismatch(let expected, let actual):
            return "Vector dimension mismatch: expected \(expected), got \(actual)"
        case .generalError(let message):
            return message
        }
    }
}

#if canImport(Foundation)
import Foundation
extension HNSWError: LocalizedError {}
#endif
