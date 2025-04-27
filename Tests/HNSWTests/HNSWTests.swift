import Testing
@testable import HNSW
import NaturalLanguage


@Suite("Index")
struct IndexTests {
    @Test
    func testSearch() async throws {
        // Create an NLEmbedding for English
        guard let embedding = NLEmbedding.wordEmbedding(for: .english) else {
            throw TestError("Failed to create NLEmbedding")
        }
        
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

        let foundWords = Set(results.map({ words[$0.id] }))
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
