//
//  File.swift
//  HNSW
//
//  Created by Noah Kamara on 29.04.2025.
//

public struct HNSWSearchResult: Sendable {
    public typealias ID = Int32
    public let id: ID
    public let distance: Float
}

