//
//  RegressionAndSemanticsTests.swift
//
//  Covers behaviors that are easy to regress: cosine/query handling, result ordering,
//  persistence + metadata, post-filter k deficiency, and edge cases.
//

import Foundation
@testable import HNSW
import Testing

private func assertSameSearchOrder(
    _ a: [HNSWSearchResult],
    _ b: [(id: Int, distance: Float)],
    accuracy: Float = 1e-4
) {
    #expect(a.count == b.count)
    for (u, f) in zip(a, b) {
        #expect(Int(u.id) == f.id)
        #expect(abs(u.distance - f.distance) < accuracy)
    }
}

@Suite("Search semantics")
struct SearchSemanticsTests {
    /// The C bridge copies from hnswlib's max-heap so distances are non-increasing (farthest first, nearest last).
    @Test
    func knnResultsAreOrderedFarthestToNearest() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 20, M: 8, efConstruction: 64)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([1, 0], id: 1)
        try index.addPoint([2, 0], id: 2)
        try index.addPoint([3, 0], id: 3)
        index.setEf(32)
        let results = try index.searchKnn([0.5, 0], maxResults: 4)
        #expect(results.count == 4)
        let distances = results.map(\.distance)
        for i in distances.indices.dropLast() {
            #expect(distances[i] >= distances[i + 1] - 1e-5)
        }
    }

    @Test
    func searchOnEmptyIndexReturnsNoResults() throws {
        let index = HNSWIndex(dimension: 3, maxElements: 4)
        let plain = try index.searchKnn([0, 0, 1], maxResults: 5)
        #expect(plain.isEmpty)
        let filtered = try index.searchKnn([0, 0, 1], maxResults: 5) { _ in true }
        #expect(filtered.isEmpty)
    }

    @Test
    func l2UnfilteredMatchesFilteredWhenFilterPassesAll() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, space: .l2)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([10, 0], id: 1)
        let query = [Float]([1, 0])
        let u = try index.searchKnn(query, maxResults: 2)
        let f = try index.searchKnn(query, maxResults: 2) { _ in true }
        assertSameSearchOrder(u, f)
    }
}

@Suite("Post-filter search limits")
struct PostFilterSearchTests {
    /// Post-filtering a fixed k-NN list cannot return k matches if fewer than k graph neighbors pass the filter.
    @Test
    func selectiveMetadataFilterMayReturnFewerThanMaxResults() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 64, M: 8, efConstruction: 80)
        for i in 0 ..< 32 {
            let x = Float(i)
            try index.addPoint([x, 0], id: Int32(i), metadata: "{\"tag\":\(i == 5 ? 1 : 0)}")
        }
        index.setEf(64)
        let query = [Float]([5, 0])
        let results = try index.searchKnn(query, maxResults: 10) { meta in
            meta?.contains("\"tag\":1") == true
        }
        #expect(results.count == 1)
        #expect(results.first?.id == 5)
    }
}

@Suite("Persistence")
struct PersistenceTests {
    @Test
    func saveLoadRoundtripPreservesVectorsAndMetadata() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let dim = 3
            let index = HNSWIndex(dimension: dim, maxElements: 10)
            try index.addPoint([1, 0, 0], id: 0, metadata: "alpha")
            try index.addPoint([0, 1, 0], id: 1, metadata: "beta")
            try index.saveIndex(to: url.path)

            let loaded = HNSWIndex(dimension: dim, maxElements: 10)
            try loaded.loadIndex(from: url.path, maxElements: 10)
            #expect(loaded.elementCount == 2)
            #expect(loaded.getMetadata(for: 0) == "alpha")
            #expect(loaded.getMetadata(for: 1) == "beta")

            let q = [Float]([0.9, 0.1, 0])
            let hits = try loaded.searchKnn(q, maxResults: 2)
            let ids = Set(hits.map(\.id))
            #expect(ids == Set<Int32>([0, 1]))
        }
    }

    /// Regression: loading a different snapshot must not leave metadata keys for labels that no longer exist.
    @Test
    func loadIndexClearsStaleMetadataFromPreviousSnapshot() throws {
        try HNSWTestPaths.withTemporaryIndexBase { fullURL in
            let dim = 3
            let partialBase = fullURL.deletingLastPathComponent()
                .appendingPathComponent("hnsw-partial-\(UUID().uuidString)")
            let partialURL = partialBase.appendingPathExtension("bin")
            defer {
                try? FileManager.default.removeItem(at: partialURL)
                try? FileManager.default.removeItem(at: URL(fileURLWithPath: partialURL.path + ".metadata"))
            }

            let full = HNSWIndex(dimension: dim, maxElements: 10)
            try full.addPoint([1, 0, 0], id: 0, metadata: "from-full-0")
            try full.addPoint([0, 1, 0], id: 1, metadata: "from-full-1")
            try full.saveIndex(to: fullURL.path)
            try full.loadIndex(from: fullURL.path, maxElements: 10)

            let partialOnly = HNSWIndex(dimension: dim, maxElements: 10)
            try partialOnly.addPoint([0, 0, 1], id: 0, metadata: "from-partial-0")
            try partialOnly.saveIndex(to: partialURL.path)

            try full.loadIndex(from: partialURL.path, maxElements: 10)
            #expect(full.elementCount == 1)
            #expect(full.getMetadata(for: 0) == "from-partial-0")
            #expect(full.getMetadata(for: 1) == nil)
        }
    }

    @Test
    func cosineIndexSaveLoadAndSearch() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let dim = 2
            let index = HNSWIndex(dimension: dim, maxElements: 8, space: .cosine)
            try index.addPoint([3, 4], id: 0)
            try index.addPoint([0, 1], id: 1)
            try index.saveIndex(to: url.path)

            let loaded = HNSWIndex(dimension: dim, maxElements: 8, space: .cosine)
            try loaded.loadIndex(from: url.path, maxElements: 8)
            let q = [Float]([30, 40])
            let hits = try loaded.searchKnn(q, maxResults: 1)
            #expect(hits.first?.id == 0)
        }
    }
}

@Suite("Add / errors")
struct AddAndErrorTests {
    @Test
    func duplicatePointIdThrows() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 4)
        let v = [Float]([1, 0])
        try index.addPoint(v, id: 0)
        #expect(throws: HNSWError.self) {
            try index.addPoint(v, id: 0)
        }
    }

    @Test
    func idOutOfRangeThrows() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 2)
        #expect(throws: HNSWError.self) {
            try index.addPoint([0, 1], id: 2)
        }
    }
}

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

@Suite("HNSWContainer")
struct HNSWContainerTests {
    @Test
    func performRunsOnSameLogicalIndex() async throws {
        let container = HNSWContainer(dimension: 2, maxElements: 8)
        try await container.perform { idx in
            try idx.addPoint([1, 0], id: 0)
        }
        let count = try await container.perform { $0.elementCount }
        #expect(count == 1)
    }

    @Test
    func resetReplacesUnderlyingIndex() async throws {
        let container = HNSWContainer(dimension: 2, maxElements: 4)
        try await container.perform { try $0.addPoint([1, 0], id: 0) }
        await container.reset()
        let count = try await container.perform { $0.elementCount }
        #expect(count == 0)
        let maxEl = try await container.perform { $0.maxElements }
        #expect(maxEl == 4)
    }
}
