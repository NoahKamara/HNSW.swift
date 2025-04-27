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
        #expect(index.maxElements == maxElements)
        #expect(index.elementCount == 0)
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
    func metadata() async throws {
        let vector = try #require(self.embedding.vector(for: "banana")).map(Float.init)
        let index = HNSWIndex(dimension: vector.count, maxElements: 1)

        let metadata = """
        {
            "title": "Document 1",
            "author": "John Doe",
            "tags": ["important", "reference"]
        }
        """
        try index.addPoint(vector, id: 0, metadata: metadata)

        // Get metadata
        let retrievedMetadata = try #require(index.getMetadata(for: 0))
        #expect(retrievedMetadata == metadata)

        // Update metadata
        let updatedMetadata = """
        {
            "title": "Updated Document 1",
            "author": "John Doe",
            "tags": ["important", "reference", "updated"]
        }
        """
        index.setMetadata(updatedMetadata, for: 0)
        let updatedRetrievedMetadata = try #require(index.getMetadata(for: 0))
        #expect(updatedRetrievedMetadata == updatedMetadata)

        // Remove metadata
        index.removeMetadata(for: 0)
        #expect(index.getMetadata(for: 0) == nil)
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

    @Test
    func resizeIndex() async throws {
        let fooVector = try #require(embedding.vector(for: "apple")).map(Float.init)
        let barVector = try #require(embedding.vector(for: "banana")).map(Float.init)
        let index = HNSWIndex(dimension: fooVector.count, maxElements: 1)

        #expect(index.maxElements == 1)

        // Add first point
        try index.addPoint(fooVector, id: 0)
        #expect(index.elementCount == 1)

        // Verify we can't add more points
        #expect(throws: HNSWError.self) {
            try index.addPoint(barVector, id: 1)
        }

        // Resize the index
        try index.resizeIndex(to: 2)
        #expect(index.maxElements == 2)
        #expect(index.elementCount == 1)  // Element count should remain the same after resize
        
        // Add second point after resize
        try index.addPoint(barVector, id: 1)
        #expect(index.elementCount == 2)
        
        // Verify both points are searchable
        let results = try index.searchKnn(fooVector, maxResults: 2)
        #expect(results.count == 2)
        #expect(Set(results.map { $0.id }) == Set([0, 1]))
        
        // Verify we can still search with the second vector
        let results2 = try index.searchKnn(barVector, maxResults: 2)
        #expect(results2.count == 2)
        #expect(Set(results2.map { $0.id }) == Set([0, 1]))
    }

    @Test
    func searchKnn() async throws {
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
    func searchKnnWithMetadataFilter() async throws {
        let peelableWords = ["banana", "orange"]
        let words = ["apple", "grape", "strawberry"] + peelableWords
        let index = try HNSWIndex.testIndex(words: words)

        var i = 0
        for word in words + peelableWords {
            defer { i += 1 }
            let peel = peelableWords.contains(word) ? true : false
            let metadata = """
            {
                "peel": \(peel),
                "word": "\(word)"
            }
            """
            index.setMetadata(metadata, for: Int32(i))
        }

        // Test nearest neighbor search with category filter
        let queryWord = "fruit"
        guard let queryVector = embedding.vector(for: queryWord) else {
            throw TestError("Failed to get vector for query word: \(queryWord)")
        }

        // Search only for fruits
        let results = try index.searchKnn(
            queryVector.map(Float.init),
            maxResults: words.count
        ) { metadata in
            guard let metadata = metadata else { return false }
            return metadata.contains("\"peel\": true")
        }

        // Verify all results are fruits
        let foundWords = Set(results.map { words[$0.id] })
        #expect(foundWords.isSubset(of: Set(peelableWords)))
        #expect(foundWords.count > 0)
    }


    @Test
    func dimensionMismatchError() async throws {
        let index = HNSWIndex(dimension: 2, maxElements: 3)

        try index.addPoint([1.0, 1.2], id: 0)

        #expect(throws: HNSWError.self) {
            try index.addPoint([1.0, 2.0, 3.0], id: 1)
        }

        #expect(throws: HNSWError.self) {
            try index.addPoint([1.0], id: 2)
        }

        #expect(throws: HNSWError.self) {
            try index.searchKnn([1.0, 2.0, 3.0], maxResults: 1)
        }

        #expect(throws: HNSWError.self) {
            try index.searchKnn([1.0], maxResults: 1)
        }
    }
}

// Helper error type for tests
struct TestError: Error {
    let message: String
    init(_ message: String) {
        self.message = message
    }
}
