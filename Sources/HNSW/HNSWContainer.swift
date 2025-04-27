//
//  SwiftUIView.swift
//  HNSW
//
//  Created by Noah Kamara on 27.04.2025.
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
        try action(index)
    }
}
