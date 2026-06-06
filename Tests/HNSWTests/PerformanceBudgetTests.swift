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

    /// Pinned thread count for batch native APIs (override with `HNSW_PERF_THREADS`).
    static let batchNumThreads: Int = ProcessInfo.processInfo.environment["HNSW_PERF_THREADS"].flatMap(Int.init) ?? 4

    static let iterationCount: Int = {
        let configured = ProcessInfo.processInfo.environment["HNSW_PERF_ITERATIONS"].flatMap(Int.init) ?? 3
        return max(1, configured)
    }()

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

    static func sequentialIndex(
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

    static func batchIndex(
        vectors: [[Float]],
        dimension: Int,
        space: HNSWSpaceType = .l2,
        numThreads: Int = batchNumThreads
    ) throws -> HNSWIndex {
        let index = HNSWIndex(
            dimension: dimension,
            maxElements: vectors.count,
            M: 16,
            efConstruction: 100,
            space: space
        )
        index.numThreads = numThreads
        let ids = vectors.indices.map { Int32($0) }
        try index.addPoints(vectors, ids: ids, numThreads: numThreads)
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

    static func median(_ samples: [Double]) -> Double {
        let sorted = samples.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    static func reportMedian(_ name: String, samples: [Double], unit: String = "ms") {
        let med = self.median(samples)
        let min = samples.min() ?? med
        let max = samples.max() ?? med
        print(
            "HNSW performance: \(name)=\(String(format: "%.3f", med))\(unit) "
                + "(\(samples.count) runs, min=\(String(format: "%.3f", min)), max=\(String(format: "%.3f", max)))"
        )
    }

    static func measureMedian(
        _ name: String,
        iterations: Int = iterationCount,
        warmup: (() throws -> Void)? = nil,
        _ body: () throws -> Void
    ) throws -> Double {
        try self.measureMedianMetric(name, iterations: iterations, warmup: warmup) {
            try self.elapsedMilliseconds(body)
        }
    }

    static func measureMedianMetric(
        _ name: String,
        iterations: Int = iterationCount,
        warmup: (() throws -> Void)? = nil,
        _ body: () throws -> Double
    ) throws -> Double {
        if let warmup {
            try warmup()
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            try samples.append(body())
        }

        self.reportMedian(name, samples: samples)
        return self.median(samples)
    }
}

/// All performance budgets run in one serialized suite so benchmarks do not compete across suites.
@Suite("Performance", .serialized)
struct PerformanceBudgetTests {
    @Test
    func harnessConfiguration() {
        guard HNSWPerformance.isEnabled else { return }

        print(
            "HNSW performance: config iterations=\(HNSWPerformance.iterationCount) "
                + "batch_threads=\(HNSWPerformance.batchNumThreads) "
                + "processors=\(ProcessInfo.processInfo.processorCount)"
        )
    }

    @Test
    func l2BulkInsertBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        var insertedCount = 0

        let median = try HNSWPerformance.measureMedian("l2_insert_5000", warmup: {
            _ = try HNSWPerformance.sequentialIndex(
                vectors: HNSWPerformance.vectors(count: 100, dimension: dimension),
                dimension: dimension
            )
        }) {
            let index = try HNSWPerformance.sequentialIndex(vectors: vectors, dimension: dimension)
            insertedCount = index.elementCount
        }

        #expect(insertedCount == vectors.count)
        #expect(median <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_INSERT_MS", default: 1000))
    }

    @Test
    func l2BatchInsertBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        var insertedCount = 0

        let median = try HNSWPerformance.measureMedian("l2_insert_5000_batch", warmup: {
            _ = try HNSWPerformance.batchIndex(
                vectors: HNSWPerformance.vectors(count: 100, dimension: dimension),
                dimension: dimension
            )
        }) {
            let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)
            insertedCount = index.elementCount
        }

        #expect(insertedCount == vectors.count)
        #expect(median <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_BATCH_INSERT_MS", default: 1000))
    }

    @Test
    func l2SearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 250, dimension: dimension, startingAt: 20000)
        let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)

        for query in queries.prefix(25) {
            _ = try index.searchKnn(query, maxResults: 10, ef: 64)
        }

        var resultCount = 0
        let p95 = try HNSWPerformance.measureMedianMetric("l2_search_p95") {
            var count = 0
            let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
                count += try index.searchKnn(query, maxResults: 10, ef: 64).count
            }
            resultCount = count
            return HNSWPerformance.percentile(samples, 0.95)
        }

        #expect(resultCount == queries.count * 10)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_SEARCH_P95_MS", default: 1))
    }

    @Test
    func l2BulkDeleteBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let count = 5000
        let vectors = HNSWPerformance.vectors(count: count, dimension: dimension)
        let ids = vectors.indices.map { Int32($0) }

        let median = try HNSWPerformance.measureMedianMetric("l2_delete_5000", warmup: {
            let warmupIndex = try HNSWPerformance.batchIndex(
                vectors: HNSWPerformance.vectors(count: 100, dimension: dimension),
                dimension: dimension
            )
            try warmupIndex.markDeleted(0)
        }) {
            let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)
            let elapsed = try HNSWPerformance.elapsedMilliseconds {
                for id in ids {
                    try index.markDeleted(id)
                }
            }
            #expect(try index.isLabelActive(id: 0) == false)
            return elapsed
        }

        #expect(median <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_DELETE_MS", default: 500))
    }

    @Test
    func l2BatchDeleteBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let count = 5000
        let vectors = HNSWPerformance.vectors(count: count, dimension: dimension)
        let ids = vectors.indices.map { Int32($0) }

        let median = try HNSWPerformance.measureMedianMetric("l2_delete_5000_batch", warmup: {
            let warmupIndex = try HNSWPerformance.batchIndex(
                vectors: HNSWPerformance.vectors(count: 100, dimension: dimension),
                dimension: dimension
            )
            try warmupIndex.markDeleted(ids: [0], numThreads: HNSWPerformance.batchNumThreads)
        }) {
            let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)
            let elapsed = try HNSWPerformance.elapsedMilliseconds {
                try index.markDeleted(ids: ids, numThreads: HNSWPerformance.batchNumThreads)
            }
            #expect(try index.isLabelActive(id: 0) == false)
            return elapsed
        }

        #expect(median <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_BATCH_DELETE_MS", default: 500))
    }

    @Test
    func l2BatchSearchTotalBudget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 250, dimension: dimension, startingAt: 20000)
        let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)
        index.numThreads = HNSWPerformance.batchNumThreads

        var flatQueries = [Float]()
        flatQueries.reserveCapacity(queries.count * dimension)
        for query in queries {
            flatQueries.append(contentsOf: query)
        }

        _ = try index.searchKnnBatch(
            queries: flatQueries,
            queryCount: queries.count,
            maxResults: 10,
            ef: 64,
            numThreads: HNSWPerformance.batchNumThreads
        )

        var resultRows = 0
        let median = try HNSWPerformance.measureMedian("l2_batch_search_total") {
            let batch = try index.searchKnnBatch(
                queries: flatQueries,
                queryCount: queries.count,
                maxResults: 10,
                ef: 64,
                numThreads: HNSWPerformance.batchNumThreads
            )
            resultRows = batch.count
        }

        let perQueryMs = median / Double(queries.count)
        print("HNSW performance: l2_batch_search_per_query=\(String(format: "%.3f", perQueryMs))ms")
        #expect(resultRows == queries.count)
        #expect(median <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_L2_BATCH_SEARCH_TOTAL_MS", default: 50))
    }

    @Test
    func metadataFilteredSearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 150, dimension: dimension, startingAt: 30000)
        let index = try HNSWPerformance.sequentialIndex(
            vectors: vectors,
            dimension: dimension,
            metadata: { $0.isMultiple(of: 8) ? "accept" : "reject" }
        )

        for query in queries.prefix(15) {
            _ = try index.searchKnn(query, maxResults: 10, ef: 256) { $0 == "accept" }
        }

        var resultCount = 0
        let p95 = try HNSWPerformance.measureMedianMetric("metadata_filtered_search_p95") {
            var count = 0
            let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
                count += try index.searchKnn(query, maxResults: 10, ef: 256) { $0 == "accept" }.count
            }
            resultCount = count
            return HNSWPerformance.percentile(samples, 0.95)
        }

        #expect(resultCount > 0)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_METADATA_FILTER_P95_MS", default: 5))
    }

    @Test
    func cosineSearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 200, dimension: dimension, startingAt: 40000)
        let index = try HNSWPerformance.sequentialIndex(vectors: vectors, dimension: dimension, space: .cosine)

        for query in queries.prefix(20) {
            _ = try index.searchKnn(query, maxResults: 10, ef: 64)
        }

        var resultCount = 0
        let p95 = try HNSWPerformance.measureMedianMetric("cosine_search_p95") {
            var count = 0
            let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
                count += try index.searchKnn(query, maxResults: 10, ef: 64).count
            }
            resultCount = count
            return HNSWPerformance.percentile(samples, 0.95)
        }

        #expect(resultCount == queries.count * 10)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_COSINE_SEARCH_P95_MS", default: 1))
    }

    @Test
    func allowlistFilteredSearchP95Budget() throws {
        guard HNSWPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = HNSWPerformance.vectors(count: 5000, dimension: dimension)
        let queries = HNSWPerformance.vectors(count: 150, dimension: dimension, startingAt: 30000)
        let index = try HNSWPerformance.batchIndex(vectors: vectors, dimension: dimension)
        let allowlist = HNSWLabelAllowlist(
            maxElements: vectors.count,
            allowing: (0..<vectors.count).lazy.filter { $0.isMultiple(of: 8) }.map(Int32.init)
        )

        for query in queries.prefix(15) {
            _ = try index.searchKnn(query, maxResults: 10, ef: 256, allowlist: allowlist)
        }

        var resultCount = 0
        let p95 = try HNSWPerformance.measureMedianMetric("allowlist_filtered_search_p95") {
            var count = 0
            let samples = try HNSWPerformance.searchLatencies(queries: queries) { query in
                count += try index.searchKnn(query, maxResults: 10, ef: 256, allowlist: allowlist).count
            }
            resultCount = count
            return HNSWPerformance.percentile(samples, 0.95)
        }

        #expect(resultCount > 0)
        #expect(p95 <= HNSWPerformance.budgetMilliseconds("HNSW_PERF_ALLOWLIST_FILTER_P95_MS", default: 5))
    }
}
