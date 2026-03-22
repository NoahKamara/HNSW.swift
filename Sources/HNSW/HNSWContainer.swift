//
//  HNSWContainer.swift
//
//  Copyright © 2024 Noah Kamara.
//

/// A Sendable wrapper around a HNSWIndex
public actor HNSWContainer {
    private var index: HNSWIndex

    public init(index: HNSWIndex) {
        self.index = index
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
        self.index = .init(
            dimension: dimension,
            maxElements: maxElements,
            M: M,
            efConstruction: efConstruction,
            space: space
        )
    }
    
    /// Perform an action on the ``HNSWIndex`` and return its result on the actor executor.
    public func perform<R: Sendable>(
        _ action: @Sendable (HNSWIndex) throws -> R
    ) throws -> R {
        try action(self.index)
    }

    /// Perform an action on the ``HNSWIndex`` and return its result on the actor executor.
    public func perform<R: Sendable>(
        _ action: @Sendable (HNSWIndex) -> R
    ) -> R {
        action(self.index)
    }
    
    /// Drops the current index and initializes a new one with the same configuration
    public func reset() {
        self.index = .init(
            dimension: self.index.dimension,
            maxElements: self.index.maxElements,
            M: self.index.M,
            efConstruction: self.index.efConstruction,
            space: self.index.space
        )
    }
}
