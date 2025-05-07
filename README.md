# HNSW Swift Package

A Swift package that provides Swift bindings for [hnswlib](https://github.com/nmslib/hnswlib), a header-only C++ library for fast approximate nearest neighbor search using the Hierarchical Navigable Small World (HNSW) algorithm. This package enables high-performance vector similarity search in Swift applications.

## Features

- Swift bindings for the lightweight, header-only hnswlib C++ library
- Support for multiple distance metrics:
  - Squared L2 (Euclidean) distance
  - Inner product
  - Cosine similarity
- Full support for incremental index construction and updates
- Support for element deletions and memory reuse
- Thread-safe implementation
- Support for macOS 11+ and iOS 16+

## Requirements

- Swift 6.0 or later
- macOS 11.0+ / iOS 16.0+

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/HNSW.git", branch: "main")
]
```

## Usage

```swift
import HNSW

// Create an HNSW index
let dimension = 128
let maxElements = 10_000
let index = try HNSWIndex(dimension: dimension, maxElements: maxElements)

// Add vectors to the index
let vector = [Float](repeating: 0.0, count: dimension)
try index.add(vector: vector, id: 0)

// Search for nearest neighbors
let queryVector = [Float](repeating: 0.0, count: dimension)
let results = try index.search(vector: queryVector, k: 10)
```

You can checkout the Tests/ directory for an example on how to use NLEmbeddings

## Project Structure

- `Sources/HNSW/` - Swift implementation and public API
- `Sources/CHNSWLib/` - C++ interop layer
- `Tests/` - Unit tests

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository.

## Acknowledgments

- [hnswlib](https://github.com/nmslib/hnswlib) - The lightweight, header-only C++ implementation of HNSW algorithm 