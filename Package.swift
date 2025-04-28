// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "HNSW",
    platforms: [.macOS(.v11), .iOS(.v16)],
    products: [
        .library(
            name: "HNSW",
            targets: ["HNSW"]
        ),
    ],
    targets: [
        // C++ interop target (for HNSW)
        .target(
            name: "CHNSWLib",
            path: "Sources/CHNSWLib",
            exclude: [],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../../hnswlib"),
                .headerSearchPath("../../hnswlib/hnswlib"),
            ]
        ),
        .target(
            name: "HNSW",
            dependencies: ["CHNSWLib"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "HNSWTests",
            dependencies: ["HNSW"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
