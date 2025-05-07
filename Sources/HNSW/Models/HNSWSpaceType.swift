//
//  File.swift
//  HNSW
//
//  Created by Noah Kamara on 29.04.2025.
//

import CHNSWLib

public enum HNSWSpaceType: Sendable, CustomStringConvertible {
    case l2
    case cosine
    
    public var description: String {
        switch self {
        case .l2: "L2"
        case .cosine: "Cosine"
        }
    }
    
    var cValue: CHNSWLib.HNSWSpaceType {
        switch self {
        case .l2:
            return HNSW_SPACE_L2
        case .cosine:
            return HNSW_SPACE_COSINE
        }
    }
    
    init(cValue: CHNSWLib.HNSWSpaceType) {
        switch cValue {
        case HNSW_SPACE_L2:
            self = .l2
        case HNSW_SPACE_COSINE:
            self = .cosine
        default:
            self = .l2
        }
    }
}
