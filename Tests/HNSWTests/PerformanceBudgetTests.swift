//
//  PerformanceBudgetTests.swift
//
//  Copyright © 2024 Noah Kamara.
//

import Dispatch
import Foundation
@testable import HNSW
import Testing

private enum HNSWPerformance {
    static let isEnabled = ProcessInfo.processInfo.environment["RUN_PERF"] == "1"

    static func budgetMilliseconds(_ environmentKey: String, default defaultValue: Double) -> Double {
        ProcessInfo.processInfo.environment[environmentKey].flatMap(Double.init) ?? defaultValue
    }

    static func elapsedMilliseconds(_ body: () throws -> Void) rethrows -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        try body()
        let end = DispatchTime.now().uptimeNanoseconds
        return Double(end - start) / 1000000
    }

    static func vectors(count: Int, dimension: Int, startingAt seedOffset: Int = 0) -> [[Float]] {
        var output: [[Float]] = []
        output.reserveCapacity(count)

        for id in seedOffset..<(seedOffset + count) {
            var state = UInt64(truncatingIfNeeded: id + 1)
            state &+= UInt64(dimension) &* 0x9E3779B97F4A7C15

            var vector = [Float](repeating: 0, count: dimension)
            for i in 0..<dimension {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let fraction = Float((state >> 40) & 0xFFFF) / Float(UInt16.max)
                vector[i] = (fraction * 2) - 1
            }
            output.append(vector)
        }

        return output
    }

    static func index(
        vectors: [[Float]],
        dimension: Int,
        space: HNSWSpaceType = .l2,
        metadata: ((Int) -> String?)? = nil
    ) throws -> HNSWIndex {
        let index = HNSWIndex(
            dimension: dimension,
            maxElements: vectors.count,
            M: 16,
            efConstruction: 100,
            space: space
        )

        for (id, vector) in vectors.enumerated() {
            try index.addPoint(vector, id: Int32(id), metadata: metadata?(id))
        }

        return index
    }

    static func searchLatencies(
        queries: [[Float]],
        _ body: ([Float]) throws -> Void
    ) rethrows -> [Double] {
        var samples: [Double] = []
        samples.reserveCapacity(queries.count)

        for query in queries {
            let elapsed = try self.elapsedMilliseconds {
                try body(query)
            }
            samples.append(elapsed)
        }

        return samples
    }

    static func percentile(_ samples: [Double], _ percentile: Double) -> Double {
        let sorted = samples.sorted()
        guard let first = sorted.first else { return 0 }
        let index = Int((Double(sorted.count - 1) * percentile).rounded(.up))
        return sorted.indices.contains(index) ? sorted[index] : first
    }

    static func report(_ name: String, _ value: Double, unit: String = "ms") {
        print("HNSW performance: \(name)=\(String(format: "%.3f", value))\(unit)")
    }
}

@Suite("Performance budgets", .serialized)
struct PerformanceBudgetTests {
    @Test
    func l2BulkInsertBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        var insertedCount = 0

        let elapsed = try HNSWPerformance.elapsedMilliseconds {
            let index = try HNSWPerformance.index(vectors: vectors, dimension: dimension)
            insertedCount = index.elementCount
        }

        HNSWPerformance.report("l2_insert_5000", elapsed)
        #expect(insertedCount == vectors.count)
        #expect(elapsed <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_INSERT_MS", default: 1000))
    }

    @Test
    func l2SearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 250, dimension: dimension, startingAt: 20000)
        let index = try HNSWPerformance.index(vectors: vectors, dimension: dimension)
        index.setEf(64)

        for query in queries.prefix(25) {
            _ = try index.searchKnn(query, maxResults: 10)
        }

        var resultCount = 0
        let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
            resultCount += try index.searchKnn(query, maxResults: 10).count
        }
        let p95 = HNSWPerformance.percentile(samples, 0.95)

        HNSWPerformance.report("l2_search_p95", p95)
        #expect(resultCount == queries.count * 10)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_SEARCH_P95_MS", default: 1))
    }

    @Test(.disabled(if: !HNSWPerformance.isEnabled))
    func metadataFilteredSearchP95Budget() throws {
        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 150, dimension: dimension, startingAt: 30000)
        let index = try HNSWPerformance.index(
            vectors: vectors,
            dimension: dimension,
            metadata: { $0.isMultiple(of: 8) ? "accept" : "reject" }
        )
        index.setEf(256)

        for query in queries.prefix(15) {
            _ = try index.searchKnn(query, maxResults: 10) { $0 == "accept" }
        }

        var resultCount = 0
        let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
            resultCount += try index.searchKnn(query, maxResults: 10) { $0 == "accept" }.count
        }
        let p95 = HNSWPerformance.percentile(samples, 0.95)

        HNSWPerformance.report("metadata_filtered_search_p95", p95)
        #expect(resultCount > 0)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_METADATA_FILTER_P95_MS", default: 5))
    }

    @Test
    func cosineSearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 200, dimension: dimension, startingAt: 40000)
        let index = try HNSWPerformance.index(vectors: vectors, dimension: dimension, space: .cosine)
        index.setEf(64)

        for query in queries.prefix(20) {
            _ = try index.searchKnn(query, maxResults: 10)
        }

        var resultCount = 0
        let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
            resultCount += try index.searchKnn(query, maxResults: 10).count
        }
        let p95 = HNSWPerformance.percentile(samples, 0.95)

        HNSWPerformance.report("cosine_search_p95", p95)
        #expect(resultCount == queries.count * 10)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_COSINE_SEARCH_P95_MS", default: 1))
    }
}
