// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "HNSW",
    products: [
        .library(
            name: "HNSW",
            targets: ["HNSW"]),
    ],
    targets: [
        .target(
            name: "CHNSWLib",
            path: "Sources/CHNSWLib",
            exclude: [],
            sources: ["hnswlib_wrapper.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../../hnswlib"),
                .headerSearchPath("../../hnswlib/hnswlib")
            ]
        ),
        .target(
            name: "HNSW",
            dependencies: ["CHNSWLib"],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
        .testTarget(
            name: "HNSWTests",
            dependencies: ["HNSW"],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
