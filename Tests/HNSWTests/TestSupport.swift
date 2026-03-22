//
//  TestSupport.swift
//

import Foundation

enum HNSWTestPaths {
    /// Runs `body` with a unique base path; removes the index file and `.metadata` sidecar afterward.
    static func withTemporaryIndexBase<T>(
        _ body: (URL) throws -> T
    ) throws -> T {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("hnsw-test-\(UUID().uuidString)")
        let indexURL = base.appendingPathExtension("bin")
        defer {
            try? FileManager.default.removeItem(at: indexURL)
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: indexURL.path + ".metadata"))
        }
        return try body(indexURL)
    }
}
