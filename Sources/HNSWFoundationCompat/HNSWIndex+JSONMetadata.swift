//
//  HNSWIndex+JSONMetadata.swift
//
//  Copyright © 2024 Noah Kamara.
//

import Foundation
import HNSW

private func utf8String(from data: Data) throws -> String {
    guard let string = String(data: data, encoding: .utf8) else {
        throw HNSWError.jsonEncodedMetadataNotUTF8
    }
    return string
}

public extension HNSWIndex {
    /// Decodes stored metadata as JSON into `T`, or returns `nil` when no metadata exists.
    /// - Parameters:
    ///   - id: Non-negative label whose metadata was written as UTF-8 JSON (for example via
    /// ``setJSONMetadata(_:for:encoder:)``).
    ///   - type: Decodable type to decode into.
    ///   - decoder: Decoder instance; defaults to `JSONDecoder()`.
    /// - Returns: A decoded value, or `nil` if ``getMetadata(for:)`` returns `nil`.
    /// - Throws: ``HNSWError/invalidLabel(id:)``, decoding errors from `JSONDecoder`, or UTF-8 issues if metadata is
    /// not valid JSON for `T`.
    func getJSONMetadata<T: Decodable>(
        for id: Int32,
        as type: T.Type,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> T? {
        guard let rawMetadata = try self.getMetadata(for: id)?.data(using: .utf8) else {
            return nil
        }

        return try decoder.decode(T.self, from: rawMetadata)
    }

    /// Encodes `metadata` as JSON and stores it as the label’s metadata string.
    /// - Parameters:
    ///   - metadata: Value to encode with `encoder`.
    ///   - id: Non-negative label to update.
    ///   - encoder: Encoder instance; defaults to `JSONEncoder()`.
    /// - Throws: ``HNSWError/jsonEncodedMetadataNotUTF8`` if the encoded bytes are not UTF-8, encoding errors from
    /// `JSONEncoder`, or ``HNSWError/invalidLabel(id:)``.
    func setJSONMetadata(
        _ metadata: some Encodable,
        for id: Int32,
        encoder: JSONEncoder = JSONEncoder()
    ) throws {
        let rawMetadata = try encoder.encode(metadata)
        try self.setMetadata(utf8String(from: rawMetadata), for: id)
    }

    /// Inserts `vector` under `id`, storing `jsonMetadata` encoded as a UTF-8 JSON string.
    /// - Parameters:
    ///   - vector: Embedding whose length must equal ``dimension``.
    ///   - id: Non-negative unique label.
    ///   - jsonMetadata: Encodable payload stored in the metadata side table.
    ///   - encoder: Encoder instance; defaults to `JSONEncoder()`.
    /// - Throws: Same failures as ``addPoint(_:id:metadata:)``, plus encoding errors or
    /// ``HNSWError/jsonEncodedMetadataNotUTF8``.
    func addPoint(
        _ vector: [Float],
        id: Int32,
        jsonMetadata: some Encodable,
        encoder: JSONEncoder = JSONEncoder()
    ) throws {
        let rawMetadata = try encoder.encode(jsonMetadata)
        try self.addPoint(vector, id: id, metadata: utf8String(from: rawMetadata))
    }
}
