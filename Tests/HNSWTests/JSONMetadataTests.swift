//
//  JSONMetadataTests.swift
//
//  Copyright © 2024 Noah Kamara.
//

import HNSW
import HNSWFoundationCompat
import Testing

@Suite("JSON metadata")
struct JSONMetadataTests {
    struct Payload: Codable, Equatable {
        var score: Int
        var label: String
    }

    @Test
    func jsonMetadataRoundTrip() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 4)
        try index.addPoint([1, 0], id: 0)
        let original = Payload(score: 7, label: "x")
        try index.setJSONMetadata(original, for: 0)
        let decoded: Payload? = try index.getJSONMetadata(for: 0, as: Payload.self)
        #expect(decoded == original)
    }
}
