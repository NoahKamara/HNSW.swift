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

    /// Perform an action on the ``HNSWIndex``.
    public func perform<R>(
        _ action: (HNSWIndex) throws -> R
    ) async rethrows -> R {
        try action(self.index)
    }
}
