//
//  HNSWTests.swift
//
//  Copyright © 2024 Noah Kamara.
//

@testable import HNSW
import NaturalLanguage
import Testing

@Suite("Index")
struct IndexTests {
    let embedding = NLEmbedding.wordEmbedding(for: .english)!

    @Test
    func initialize() async throws {
        let vector = try #require(self.embedding.vector(for: "banana")).map(Float.init)
        let dimension = vector.count
        let maxElements = 100
        let index = HNSWIndex(dimension: dimension, maxElements: maxElements)
        #expect(index.dimension == vector.count)
        #expect(index.M == 16)
        #expect(index.maxElements == maxElements)
        #expect(index.elementCount == 0)
        #expect(index.space == "l2")
    }

    @Test
    func addPoint() async throws {
        let vector = try #require(self.embedding.vector(for: "banana")).map(Float.init)
        let index = HNSWIndex(dimension: vector.count, maxElements: 1)

        try index.addPoint(vector, id: 0)
        #expect(index.elementCount == 1)
        let results = try index.searchKnn(vector, maxResults: 1)
        #expect(results.first?.id == 0)
    }

    @Test
    func delete() async throws {
        let vector = try #require(self.embedding.vector(for: "banana")).map(Float.init)
        let index = HNSWIndex(dimension: vector.count, maxElements: 1)
        try index.addPoint(vector, id: 0)

        try index.markDeleted(0)
        var results = try index.searchKnn(vector, maxResults: 1)
        #expect(results.isEmpty)

        try index.unmarkDeleted(0)
        results = try index.searchKnn(vector, maxResults: 1)
        #expect(results.count == 1)
    }

//    @Test
//    func resizeIndex() async throws {
//        let fooVector = try #require(embedding.vector(for: "banana")).map(Float.init)
//        let barVector = try #require(embedding.vector(for: "banana")).map(Float.init)
//        let index = HNSWIndex(dimension: fooVector.count, maxElements: 1)
//
//        #expect(index.maxElements == 1)
//
//        try index.addPoint(fooVector, id: 0)
//
//        #expect(throws: HNSWIndex.VectorMismatch.self) {
//            try index.addPoint(barVector, id: 1)
//        }
//
//        try index.resizeIndex(to: 2)
//        #expect(index.maxElements == 2)
//        try index.addPoint(barVector, id: 1)
//    }

    @Test
    func testSearchKnn() async throws {
        let fruityWords = ["apple", "banana", "orange", "grape", "strawberry"]
        // Create some test words
        let words = fruityWords + ["doctor", "lawyer", "negotiator", "count"]
        let index = try HNSWIndex.testIndex(words: words)

        // Test nearest neighbor search
        let queryWord = "fruit"
        guard let queryVector = embedding.vector(for: queryWord) else {
            throw TestError("Failed to get vector for query word: \(queryWord)")
        }

        let results = try index.searchKnn(
            queryVector.map(Float.init),
            maxResults: fruityWords.count
        )

        let foundWords = Set(results.map { words[$0.id] })
        #expect(foundWords == Set(fruityWords))
    }

    @Test
    func dimensionMismatchError() async throws {
        let index = HNSWIndex(dimension: 2, maxElements: 3)

        try index.addPoint([1.0, 1.2], id: 0)

        #expect(throws: HNSWIndex.VectorMismatch.self, performing: {
            try index.addPoint([1.0, 2.0, 3.0], id: 1)
        })

        #expect(throws: HNSWIndex.VectorMismatch.self, performing: {
            try index.addPoint([1.0], id: 2)
        })

        #expect(throws: HNSWIndex.VectorMismatch.self, performing: {
            try index.searchKnn([1.0, 2.0, 3.0], maxResults: 1)
        })

        #expect(throws: HNSWIndex.VectorMismatch.self, performing: {
            try index.searchKnn([1.0], maxResults: 1)
        })
    }
}

// Helper error type for tests
struct TestError: Error {
    let message: String
    init(_ message: String) {
        self.message = message
    }
}
