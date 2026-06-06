//
//  RegressionAndSemanticsTests.swift
//
//  Copyright © 2024 Noah Kamara.
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
    /// Results are ordered by increasing distance (nearest neighbor first).
    @Test
    func knnResultsAreOrderedNearestToFarthest() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 20, M: 8, efConstruction: 64)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([1, 0], id: 1)
        try index.addPoint([2, 0], id: 2)
        try index.addPoint([3, 0], id: 3)
        let results = try index.searchKnn([0.5, 0], maxResults: 4, ef: 32)
        #expect(results.count == 4)
        let distances = results.map(\.distance)
        for i in distances.indices.dropLast() {
            #expect(distances[i] <= distances[i + 1] + 1e-5)
        }
    }

    @Test
    func searchOnEmptyIndexReturnsNoResults() throws {
        let index = HNSWIndex(dimension: 3, maxElements: 4)
        let plain = try index.searchKnn([0, 0, 1], maxResults: 5, ef: 32)
        #expect(plain.isEmpty)
        let filtered = try index.searchKnn([0, 0, 1], maxResults: 5, ef: 32) { _ in true }
        #expect(filtered.isEmpty)
    }

    @Test
    func l2UnfilteredMatchesFilteredWhenFilterPassesAll() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, space: .l2)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([10, 0], id: 1)
        let query = [Float]([1, 0])
        let u = try index.searchKnn(query, maxResults: 2, ef: 32)
        let f = try index.searchKnn(query, maxResults: 2, ef: 32) { _ in true }
        assertSameSearchOrder(u, f)
    }

    @Test
    func reusableBuffersMatchAllocatedSearchResults() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, space: .l2)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([1, 0], id: 1)
        try index.addPoint([3, 0], id: 2)

        let allocated = try index.searchKnn([0.2, 0], maxResults: 2, ef: 32)
        var ids: [Int32] = []
        var distances: [Float] = []
        let count = try index.searchKnn([0.2, 0], maxResults: 2, ef: 32, ids: &ids, distances: &distances)

        #expect(count == allocated.count)
        #expect(Array(ids.prefix(count)) == allocated.map(\.id))
        for (a, b) in zip(distances.prefix(count), allocated.map(\.distance)) {
            #expect(abs(a - b) < 1e-5)
        }
    }

    @Test
    func normalizedCosineAPIsMatchWrapperNormalizedSearch() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, space: .cosine)
        try index.addNormalizedPoint([1, 0], id: 0)
        try index.addNormalizedPoint([0, 1], id: 1)

        let normalized = try index.searchKnnNormalized([1, 0], maxResults: 2, ef: 32)
        let wrapperNormalized = try index.searchKnn([3, 0], maxResults: 2, ef: 32)

        #expect(normalized.count == wrapperNormalized.count)
        #expect(normalized.map(\.id) == wrapperNormalized.map(\.id))
    }

    @Test
    func nativeAllowlistSearchMatchesLabelFilter() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 16, M: 8, efConstruction: 40)
        for i in 0..<10 {
            try index.addPoint([Float(i), 0], id: Int32(i))
        }
        let allowed = Set<Int32>([2, 4, 6, 8])
        let allowlist = HNSWLabelAllowlist(maxElements: index.maxElements, allowing: allowed)
        let query: [Float] = [5, 0]
        let native = try index.searchKnn(query, maxResults: 3, ef: 32, allowlist: allowlist)
        let callback = try index.searchKnn(query, maxResults: 3, ef: 32, labelFilter: allowed.contains)

        #expect(native.map(\.id) == callback.map(\.id))
        for result in native {
            #expect(allowlist.contains(result.id))
        }
    }

    /// Labels without stored metadata receive `nil` in the filter closure; they are not skipped
    /// before the predicate runs (same idea as post-filtering an unfiltered search with ``getMetadata(for:)``).
    @Test
    func filteredSearchIncludesNoMetadataLabelsWhenPredicateAcceptsNil() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, M: 8, efConstruction: 40)
        try index.addPoint([0, 0], id: 0)
        try index.addPoint([1, 0], id: 1, metadata: "tagged")
        let hits = try index.searchKnn([0, 0], maxResults: 2, ef: 32) { $0 == nil }
        #expect(hits.count == 1)
        #expect(hits[0].id == 0)
    }
}

@Suite("Filtered search limits")
struct FilteredSearchLimitsTests {
    /// Fewer than `maxResults` hits when fewer than that many labels satisfy the filter (only one point is tagged).
    @Test
    func selectiveMetadataFilterMayReturnFewerThanMaxResults() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 64, M: 8, efConstruction: 80)
        for i in 0..<32 {
            let x = Float(i)
            try index.addPoint([x, 0], id: Int32(i), metadata: "{\"tag\":\(i == 5 ? 1 : 0)}")
        }
        let query = [Float]([5, 0])
        let results = try index.searchKnn(query, maxResults: 10, ef: 64) { meta in
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
            #expect(try loaded.getMetadata(for: 0) == "alpha")
            #expect(try loaded.getMetadata(for: 1) == "beta")

            let q = [Float]([0.9, 0.1, 0])
            let hits = try loaded.searchKnn(q, maxResults: 2, ef: 32)
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
                try? FileManager.default.removeItem(at: URL(fileURLWithPath: partialURL.path + ".index"))
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
            #expect(try full.getMetadata(for: 0) == "from-partial-0")
            #expect(try full.getMetadata(for: 1) == nil)
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
            let hits = try loaded.searchKnn(q, maxResults: 1, ef: 32)
            #expect(hits.first?.id == 0)
        }
    }

    @Test
    func loadIndexThrowsWhenSavedCosineSpaceLoadedIntoL2Wrapper() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let index = HNSWIndex(dimension: 2, maxElements: 8, space: .cosine)
            try index.addPoint([3, 4], id: 0)
            try index.saveIndex(to: url.path)

            let loaded = HNSWIndex(dimension: 2, maxElements: 8, space: .l2)
            #expect(throws: HNSWError.spaceMismatch(expected: .l2, actual: .cosine)) {
                try loaded.loadIndex(from: url.path, maxElements: 8)
            }
        }
    }

    @Test
    func loadIndexThrowsWhenSavedL2SpaceLoadedIntoCosineWrapper() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let index = HNSWIndex(dimension: 2, maxElements: 8, space: .l2)
            try index.addPoint([10, 0], id: 0)
            try index.saveIndex(to: url.path)

            let loaded = HNSWIndex(dimension: 2, maxElements: 8, space: .cosine)
            #expect(throws: HNSWError.spaceMismatch(expected: .cosine, actual: .l2)) {
                try loaded.loadIndex(from: url.path, maxElements: 8)
            }
        }
    }

    @Test
    func loadIndexThrowsWhenSavedDimensionDiffersFromWrapper() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let index = HNSWIndex(dimension: 2, maxElements: 8)
            try index.addPoint([1, 0], id: 0)
            try index.saveIndex(to: url.path)

            let loaded = HNSWIndex(dimension: 3, maxElements: 8)
            #expect(throws: HNSWError.vectorMismatch(expected: 3, actual: 2)) {
                try loaded.loadIndex(from: url.path, maxElements: 8)
            }
        }
    }

    @Test
    func loadIndexThrowsWhenWrapperMetadataSidecarIsMissing() throws {
        try HNSWTestPaths.withTemporaryIndexBase { url in
            let index = HNSWIndex(dimension: 2, maxElements: 8)
            try index.addPoint([1, 0], id: 0)
            try index.saveIndex(to: url.path)
            try FileManager.default.removeItem(at: URL(fileURLWithPath: url.path + ".index"))

            let loaded = HNSWIndex(dimension: 2, maxElements: 8)
            #expect(throws: HNSWError.generalError(message: "Missing index wrapper metadata")) {
                try loaded.loadIndex(from: url.path, maxElements: 8)
            }
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

    @Test
    func negativeLabelIdThrowsInvalidLabel() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 10)
        #expect(throws: HNSWError.invalidLabel(id: -1)) {
            try index.addPoint([0, 1], id: -1)
        }
    }

    @Test
    func setMetadataForMissingLabelThrowsWithoutCreatingMetadata() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 4)

        #expect(throws: HNSWError.labelNotFound(id: 1)) {
            try index.setMetadata("phantom", for: 1)
        }

        #expect(try index.labelExists(id: 1) == false)
        #expect(try index.getMetadata(for: 1) == nil)
    }

    @Test
    func batchAddMatchesSequentialAdd() throws {
        let vectors: [[Float]] = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
        let ids: [Int32] = [0, 1, 2, 3]

        let sequential = HNSWIndex(dimension: 2, maxElements: vectors.count, M: 8, efConstruction: 40)
        for (vector, id) in zip(vectors, ids) {
            try sequential.addPoint(vector, id: id)
        }

        let batch = HNSWIndex(dimension: 2, maxElements: vectors.count, M: 8, efConstruction: 40)
        try batch.addPoints(vectors, ids: ids)

        #expect(batch.elementCount == sequential.elementCount)
        for id in ids {
            let batchExists = try batch.labelExists(id: id)
            let sequentialExists = try sequential.labelExists(id: id)
            #expect(batchExists == sequentialExists)
        }

        let query: [Float] = [0.25, 0.25]
        let sequentialHits = try sequential.searchKnn(query, maxResults: 2, ef: 32)
        let batchHits = try batch.searchKnn(query, maxResults: 2, ef: 32)
        #expect(batchHits.map(\.id) == sequentialHits.map(\.id))
    }

    @Test
    func batchDeleteMatchesSequentialDelete() throws {
        let vectors: [[Float]] = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
        let ids: [Int32] = [0, 1, 2, 3]
        let query: [Float] = [0.25, 0.25]

        let sequential = HNSWIndex(dimension: 2, maxElements: vectors.count, M: 8, efConstruction: 40)
        try sequential.addPoints(vectors, ids: ids)
        for id in ids {
            try sequential.markDeleted(id)
        }
        #expect(try sequential.searchKnn(query, maxResults: 4, ef: 32).isEmpty)

        let batch = HNSWIndex(dimension: 2, maxElements: vectors.count, M: 8, efConstruction: 40)
        try batch.addPoints(vectors, ids: ids)
        try batch.markDeleted(ids: ids)
        #expect(try batch.searchKnn(query, maxResults: 4, ef: 32).isEmpty)

        for id in ids {
            #expect(try batch.labelExists(id: id) == true)
            #expect(try batch.isLabelActive(id: id) == false)
        }

        try batch.unmarkDeleted(ids: ids)
        let restored = try batch.searchKnn(query, maxResults: 1, ef: 32)
        #expect(restored.count == 1)
        #expect(try batch.isLabelActive(id: restored[0].id) == true)
    }

    @Test
    func batchSearchMatchesSingleSearch() throws {
        let index = HNSWIndex(dimension: 2, maxElements: 8, M: 8, efConstruction: 40)
        let stored: [[Float]] = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
        try index.addPoints(stored, ids: [0, 1, 2, 3])

        let queries: [[Float]] = [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
        let batch = try index.searchKnnBatch(queries, maxResults: 2, ef: 32)
        #expect(batch.count == queries.count)

        for (query, row) in zip(queries, batch) {
            let single = try index.searchKnn(query, maxResults: 2, ef: 32)
            #expect(row.map(\.id) == single.map(\.id))
        }
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
        let count = await container.perform { $0.elementCount }
        #expect(count == 1)
    }

    @Test
    func resetReplacesUnderlyingIndex() async throws {
        let container = HNSWContainer(dimension: 2, maxElements: 4)
        try await container.perform { try $0.addPoint([1, 0], id: 0) }
        await container.reset()
        let count = await container.perform { $0.elementCount }
        #expect(count == 0)
        let maxEl = await container.perform { $0.maxElements }
        #expect(maxEl == 4)
    }
}
