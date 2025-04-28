//
//  HNSWContainer.swift
//
//  Copyright © 2024 Noah Kamara.
//

/// A Sendable wrapper around a HNSWIndex
public actor HNSWContainer {
    private let index: HNSWIndex

    public init(index: HNSWIndex) {
        self.index = index
    }
    
    /// Perform an action on the ``HNSWIndex`` and return its result, ensuring async and safe execution within the actor.
    public func perform<R: Sendable>(
        _ action: @Sendable (HNSWIndex) throws -> R
    ) async throws -> R {
        try await withCheckedThrowingContinuation { continuation in
            do {
                // Perform the action synchronously on the actor's state
                let result = try action(self.index)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
