//
//  File.swift
//  HNSW
//
//  Created by Noah Kamara on 29.04.2025.
//

/// One neighbor returned by unfiltered k-NN search on ``HNSWIndex``.
///
/// Results are ordered by increasing ``distance`` (best match first). The meaning of `distance`
/// depends on ``HNSWSpaceType``: L2 Euclidean length for ``HNSWSpaceType/l2``, and library-defined
/// values for ``HNSWSpaceType/cosine`` after the wrapper normalizes vectors.
public struct HNSWSearchResult: Sendable {
    /// External label type used by ``HNSWIndex/addPoint(_:id:metadata:)`` and search APIs.
    public typealias ID = Int32

    /// The neighbor’s label id (non-negative when produced by this package’s validation path).
    public let id: ID

    /// Distance from the query vector under the index’s space; lower is closer for typical spaces.
    public let distance: Float
}

