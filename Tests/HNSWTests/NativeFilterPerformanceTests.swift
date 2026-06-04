//
//  NativeFilterPerformanceTests.swift
//
//  Copyright © 2024 Noah Kamara.
//

import Dispatch
import Foundation
@testable import HNSW
import Testing

private enum NativeFilterPerformance {
    static let isEnabled = ProcessInfo.processInfo.environment["RUN_PERF"] == "1"

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

@Suite("Native filter performance", .serialized)
struct NativeFilterPerformanceTests {
    @Test
    func allowlistFilteredSearchP95() throws {
        guard NativeFilterPerformance.isEnabled else { return }

        let dimension = 64
        let vectors = NativeFilterPerformance.vectors(count: 5000, dimension: dimension)
        let queries = NativeFilterPerformance.vectors(count: 150, dimension: dimension, startingAt: 30000)
        let index = HNSWIndex(dimension: dimension, maxElements: vectors.count, M: 16, efConstruction: 100)

        for (id, vector) in vectors.enumerated() {
            try index.addPoint(vector, id: Int32(id))
        }
        index.setEf(256)

        let allowlist = HNSWLabelAllowlist(
            maxElements: vectors.count,
            allowing: (0..<vectors.count).lazy.filter { $0.isMultiple(of: 8) }.map(Int32.init)
        )

        for query in queries.prefix(15) {
            _ = try index.searchKnn(query, maxResults: 10, allowlist: allowlist)
        }

        var resultCount = 0
        var samples: [Double] = []
        samples.reserveCapacity(queries.count)
        for query in queries {
            try samples.append(NativeFilterPerformance.elapsedMilliseconds {
                resultCount += try index.searchKnn(query, maxResults: 10, allowlist: allowlist).count
            })
        }

        let p95 = NativeFilterPerformance.percentile(samples, 0.95)
        NativeFilterPerformance.report("allowlist_filtered_search_p95", p95)
        #expect(resultCount > 0)
    }
}
