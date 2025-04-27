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

    /// Perform an action on the ``HNSWIndex`` and return it's result.
    public func perform<R>(
        _ action: (HNSWIndex) throws -> R
    ) async rethrows -> R {
        try action(self.index)
    }
    
    @_disfavoredOverload // SR-15150 Async overloading in protocol implementation fails
    /// Perform an action on the ``HNSWIndex`` and return it's result.
    public func perform<R>(_ action: (HNSWIndex) throws -> R) throws -> R {
        try action(self.index)
    }
}
