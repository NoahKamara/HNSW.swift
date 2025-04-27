//
//  SwiftUIView.swift
//  HNSW
//
//  Created by Noah Kamara on 27.04.2025.
//

import HNSW
import NaturalLanguage

extension HNSWIndex {
    static func testIndex(
        embedding: NLEmbedding = .wordEmbedding(for: .english)!,
        words: [String]
    ) throws -> HNSWIndex {
        // Create an NLEmbedding for English
        guard let embedding = NLEmbedding.wordEmbedding(for: .english) else {
            throw TestError("Failed to create NLEmbedding")
        }
                
        // Get embeddings for the words
        var vectors: [[Float]] = []
        for word in words {
            guard let vector = embedding.vector(for: word) else {
                throw TestError("Failed to get vector for word: \(word)")
            }
            vectors.append(vector.map(Float.init))
        }
        
        // Create HNSW index
        let dimension = vectors[0].count
        let maxElements = 1000
        let index = HNSWIndex(dimension: dimension, maxElements: maxElements)
        
        // Add vectors to the index
        for (i, vector) in vectors.enumerated() {
            try index.addPoint(vector, id: i)
        }
        
        return index
    }
}

