#include <cstdint>
#include "hnswlib_wrapper.h"
#include "hnswlib/hnswlib.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <functional>
#include <cstring>
#include <fstream>

struct HNSWIndexWrapper {
    hnswlib::HierarchicalNSW<float>* index;
    hnswlib::SpaceInterface<float>* space;  // Store the space interface
    int dimension;
    std::unordered_map<int, std::string> metadata;  // Map of ID to metadata string
    bool (*filter_func)(const char*);  // Current filter function
    HNSWSpaceType space_type;  // Store the space type
};

// Custom filter functor that checks metadata
class MetadataFilterFunctor : public hnswlib::BaseFilterFunctor {
private:
    HNSWIndexWrapper* wrapper;

public:
    MetadataFilterFunctor(HNSWIndexWrapper* wrapper) : wrapper(wrapper) {}

    bool operator()(hnswlib::labeltype id) override {
        // Skip negative IDs
        if (id < 0) return false;
        
        auto it = wrapper->metadata.find(static_cast<int>(id));
        if (it == wrapper->metadata.end()) {
            return false;  // No metadata means no match
        }
        return wrapper->filter_func ? wrapper->filter_func(it->second.c_str()) : true;
    }
};

// Add these functions before the extern "C" block
void saveMetadata(const std::unordered_map<int, std::string>& metadata, const std::string& path) {
    std::string metadataPath = path + ".metadata";
    std::ofstream file(metadataPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open metadata file for writing");
    }
    
    // Write number of entries
    size_t size = metadata.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // Write each entry
    for (const auto& pair : metadata) {
        // Write ID
        file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        
        // Write string length and content
        size_t strLen = pair.second.length();
        file.write(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
        file.write(pair.second.c_str(), strLen);
    }
}

void loadMetadata(std::unordered_map<int, std::string>& metadata, const std::string& path) {
    std::string metadataPath = path + ".metadata";
    std::ifstream file(metadataPath, std::ios::binary);
    if (!file) {
        return; // No metadata file exists, that's okay
    }
    
    // Read number of entries
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    // Read each entry
    for (size_t i = 0; i < size; i++) {
        // Read ID
        int id;
        file.read(reinterpret_cast<char*>(&id), sizeof(id));
        
        // Read string length and content
        size_t strLen;
        file.read(reinterpret_cast<char*>(&strLen), sizeof(strLen));
        std::string str(strLen, '\0');
        file.read(&str[0], strLen);
        
        metadata[id] = str;
    }
}

extern "C" {
    using namespace hnswlib;
    
    void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction, HNSWSpaceType space_type) {
        hnswlib::SpaceInterface<float>* space;
        if (space_type == HNSW_SPACE_COSINE) {
            space = new hnswlib::InnerProductSpace(dim);  // Use InnerProductSpace for cosine similarity
        } else {
            space = new hnswlib::L2Space(dim);
        }
        
        hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction);
        HNSWIndexWrapper* wrapper = new HNSWIndexWrapper{index, space, dim, {}, nullptr, space_type};
        return static_cast<void*>(wrapper);
    }
    
    void hnswlib_free_index(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        delete wrapper->index;
        delete wrapper->space;  // Free the space interface
        delete wrapper;
    }
    
    int hnswlib_add_point(void* index_ptr, const float* vector, int id) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            
            // Verify the index is in a valid state
            if (wrapper->index == nullptr) {
                return -1;  // Index not initialized
            }
            
            // Verify we have space for the new point
            if (id >= wrapper->index->max_elements_) {
                return -2;  // ID exceeds maximum elements
            }
            
            // Verify the point isn't already added
            if (wrapper->index->label_lookup_.find(id) != wrapper->index->label_lookup_.end()) {
                return -3;  // Point with ID already exists
            }
            
            wrapper->index->addPoint(vector, id);
            return 0;  // Success
        } catch (const std::exception& e) {
            return -4;  // General error
        }
    }
    
    int hnswlib_add_point_with_metadata(void* index_ptr, const float* vector, int id, const char* metadata) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            
            // Verify the index is in a valid state
            if (wrapper->index == nullptr) {
                return -1;  // Index not initialized
            }
            
            // Verify we have space for the new point
            if (id >= wrapper->index->max_elements_) {
                return -2;  // ID exceeds maximum elements
            }
            
            // Verify the point isn't already added
            if (wrapper->index->label_lookup_.find(id) != wrapper->index->label_lookup_.end()) {
                return -3;  // Point with ID already exists
            }
            
            wrapper->index->addPoint(vector, id);
            if (metadata != nullptr) {
                wrapper->metadata[id] = std::string(metadata);
            }
            return 0;  // Success
        } catch (const std::exception& e) {
            return -4;  // General error
        }
    }
    
    const char* hnswlib_get_metadata(void* index_ptr, int id) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        auto it = wrapper->metadata.find(id);
        if (it != wrapper->metadata.end()) {
            return it->second.c_str();
        }
        return nullptr;
    }
    
    void hnswlib_set_metadata(void* index_ptr, int id, const char* metadata) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        if (metadata != nullptr) {
            wrapper->metadata[id] = std::string(metadata);
        } else {
            wrapper->metadata.erase(id);
        }
    }
    
    void hnswlib_remove_metadata(void* index_ptr, int id) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        wrapper->metadata.erase(id);
    }
    
    void hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        std::priority_queue<std::pair<float, labeltype>> result = wrapper->index->searchKnn(query, k);
        
        std::vector<std::pair<float, labeltype>> sorted_results;
        while (!result.empty()) {
            sorted_results.push_back(result.top());
            result.pop();
        }
        
        // Fill in the results
        for (size_t i = 0; i < sorted_results.size(); i++) {
            ids[i] = static_cast<int>(sorted_results[i].second);
            distances[i] = sorted_results[i].first;
        }
    }

    int hnswlib_set_ef(void* index_ptr, int ef) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->setEf(ef);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_save_index(void* index_ptr, const char* path) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->saveIndex(path);
            saveMetadata(wrapper->metadata, path);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_load_index(void* index_ptr, const char* path, int max_elements) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->loadIndex(path, wrapper->space, max_elements);
            loadMetadata(wrapper->metadata, path);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_mark_deleted(void* index_ptr, int id) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->markDelete(id);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_unmark_deleted(void* index_ptr, int id) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->unmarkDelete(id);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_resize_index(void* index_ptr, int new_size) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            
            // Verify the current state
            if (new_size < wrapper->index->cur_element_count) {
                return -1;  // Cannot resize to smaller than current count
            }
            
            // Store current state for verification
            size_t current_count = wrapper->index->cur_element_count;
            
            // Perform the resize
            wrapper->index->resizeIndex(new_size);
            
            // Verify the resize operation maintained the correct state
            if (wrapper->index->cur_element_count != current_count) {
                return -2;  // Element count changed during resize
            }
            
            if (wrapper->index->max_elements_ != new_size) {
                return -3;  // Max elements not updated correctly
            }
            
            return 0;
        } catch (...) {
            return -4;  // General error
        }
    }

    HNSWSpaceType hnswlib_get_space_type(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->space_type;
    }

    int hnswlib_get_dim(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->dimension;
    }

    unsigned long hnswlib_get_M(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->index->M_;
    }

    unsigned long hnswlib_get_ef_construction(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->index->ef_construction_;
    }

    unsigned long hnswlib_get_max_elements(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->index->max_elements_;
    }

    unsigned long hnswlib_get_current_count(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->index->cur_element_count;
    }

    void hnswlib_set_filter(void* index_ptr, bool (*filter_func)(const char*)) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        wrapper->filter_func = filter_func;
    }

    void hnswlib_search_knn_with_filter(void* index_ptr, const float* query, int* ids, float* distances, int k) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        
        if (wrapper->filter_func) {
            MetadataFilterFunctor filter(wrapper);
            std::priority_queue<std::pair<float, labeltype>> result = wrapper->index->searchKnn(query, k, &filter);
            
            std::vector<std::pair<float, labeltype>> sorted_results;
            while (!result.empty()) {
                sorted_results.push_back(result.top());
                result.pop();
            }
            
            // Fill in the results
            for (size_t i = 0; i < sorted_results.size(); i++) {
                ids[i] = static_cast<int>(sorted_results[i].second);
                distances[i] = sorted_results[i].first;
            }
        } else {
            hnswlib_search_knn(index_ptr, query, ids, distances, k);
        }
    }
} 
