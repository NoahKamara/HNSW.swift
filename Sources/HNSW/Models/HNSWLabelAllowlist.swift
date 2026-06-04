//
//  HNSWLabelAllowlist.swift
//
//  Copyright © 2024 Noah Kamara.
//

/// Dense label allowlist for native filtered search.
///
/// Use this when labels are bounded by ``HNSWIndex/maxElements`` and filtered search is on a hot path. The native
/// wrapper evaluates the allowlist in C++ without calling back into Swift for each candidate.
public struct HNSWLabelAllowlist: Sendable {
    var storage: [UInt8]

    /// Creates an empty allowlist with one byte per possible label.
    public init(maxElements: Int) {
        precondition(maxElements >= 0, "maxElements must be non-negative")
        self.storage = [UInt8](repeating: 0, count: maxElements)
    }

    /// Creates an allowlist and enables the provided labels.
    public init(maxElements: Int, allowing labels: some Sequence<Int32>) {
        self.init(maxElements: maxElements)
        for label in labels {
            self.allow(label)
        }
    }

    /// Number of labels representable by this allowlist.
    public var capacity: Int {
        self.storage.count
    }

    /// Enables a non-negative label within ``capacity``.
    public mutating func allow(_ label: Int32) {
        precondition(label >= 0 && Int(label) < self.storage.count, "label must be within allowlist capacity")
        self.storage[Int(label)] = 1
    }

    /// Disables a non-negative label within ``capacity``.
    public mutating func deny(_ label: Int32) {
        precondition(label >= 0 && Int(label) < self.storage.count, "label must be within allowlist capacity")
        self.storage[Int(label)] = 0
    }

    /// Returns whether a label is currently enabled.
    public func contains(_ label: Int32) -> Bool {
        label >= 0 && Int(label) < self.storage.count && self.storage[Int(label)] != 0
    }
}
