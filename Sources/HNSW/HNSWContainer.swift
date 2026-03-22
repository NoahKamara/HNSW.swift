//
//  HNSWContainer.swift
//
//  Copyright © 2024 Noah Kamara.
//

/// An `actor` that owns an ``HNSWIndex`` and serializes all access to it across concurrent tasks.
///
/// ``HNSWIndex`` is not `Sendable` and its native backing store is not safe for unsynchronized
/// concurrent use. Use the actor’s `perform` methods to run work against the index on this actor’s
/// executor. See also <doc:ConcurrencyWithContainer>.
public actor HNSWContainer {
    private var index: HNSWIndex

    /// Wraps an existing index; the container becomes the sole concurrency domain that should touch it.
    public init(index: HNSWIndex) {
        self.index = index
    }

    /// Creates a new index with the same parameters as
    /// ``HNSWIndex/init(dimension:maxElements:M:efConstruction:space:)``.
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

    /// Runs `action` with exclusive access to the wrapped ``HNSWIndex``; rethrows any error from `action`.
    /// - Parameter action: Synchronous work that must not escape the `HNSWIndex` outside the closure.
    /// - Returns: Whatever `action` returns.
    public func perform<R: Sendable>(
        _ action: @Sendable (HNSWIndex) throws -> R
    ) throws -> R {
        try action(self.index)
    }

    /// Runs `action` with exclusive access to the wrapped ``HNSWIndex``.
    /// - Parameter action: Synchronous work that must not escape the `HNSWIndex` outside the closure.
    /// - Returns: Whatever `action` returns.
    public func perform<R: Sendable>(
        _ action: @Sendable (HNSWIndex) -> R
    ) -> R {
        action(self.index)
    }

    /// Replaces the inner index with a fresh ``HNSWIndex`` using the same dimension, capacity, `M`, `efConstruction`,
    /// and space.
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
