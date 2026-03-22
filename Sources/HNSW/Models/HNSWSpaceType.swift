//
//  HNSWSpaceType.swift
//
//  Copyright © 2024 Noah Kamara.
//

import CHNSWLib

/// Distance metric used when building and querying an ``HNSWIndex``.
///
/// Choose a space at index creation; ``HNSWIndex/loadIndex(from:maxElements:)`` verifies the loaded
/// index reports the same space or throws ``HNSWError/spaceMismatch(expected:actual:)``.
public enum HNSWSpaceType: Sendable, Equatable, CustomStringConvertible {
    /// Squared Euclidean (L2) distance between vectors as stored.
    case l2

    /// Cosine-related similarity using vectors normalized by this package on insert and query.
    case cosine

    /// A short human-readable name for logging or UI.
    public var description: String {
        switch self {
        case .l2: "L2"
        case .cosine: "Cosine"
        }
    }

    var cValue: CHNSWLib.HNSWSpaceType {
        switch self {
        case .l2:
            HNSW_SPACE_L2
        case .cosine:
            HNSW_SPACE_COSINE
        }
    }

    init(cValue: CHNSWLib.HNSWSpaceType) {
        switch cValue {
        case HNSW_SPACE_L2:
            self = .l2
        case HNSW_SPACE_COSINE:
            self = .cosine
        default:
            self = .l2
        }
    }
}
