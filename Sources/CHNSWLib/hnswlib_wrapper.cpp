#include <cstdint>
#include "hnswlib_wrapper.h"
#include "hnswlib/hnswlib.h"
#include <vector>
#include <queue>
#include <string>
#include <functional>
#include <cstring>
#include <fstream>

struct HNSWIndexWrapper {
    hnswlib::HierarchicalNSW<float>* index;
    hnswlib::SpaceInterface<float>* space;  // Store the space interface
    int dimension;
    std::vector<std::string> metadata;  // Dense metadata storage keyed by external label
    std::vector<uint8_t> has_metadata;
    HNSWSpaceType space_type;  // Store the space type
    int last_loaded_dimension;
    HNSWSpaceType last_loaded_space_type;
};

struct HNSWWrapperMetadata {
    uint32_t magic;
    uint32_t version;
    int32_t dimension;
    int32_t space_type;
};

static constexpr uint32_t HNSW_WRAPPER_METADATA_MAGIC = 0x48535731;  // "HSW1"
static constexpr uint32_t HNSW_WRAPPER_METADATA_VERSION = 1;

std::string wrapperMetadataPath(const std::string& path) {
    return path + ".index";
}

void saveWrapperMetadata(const HNSWIndexWrapper& wrapper, const std::string& path) {
    std::ofstream file(wrapperMetadataPath(path), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open wrapper metadata file for writing");
    }

    HNSWWrapperMetadata metadata{
        HNSW_WRAPPER_METADATA_MAGIC,
        HNSW_WRAPPER_METADATA_VERSION,
        static_cast<int32_t>(wrapper.dimension),
        static_cast<int32_t>(wrapper.space_type)
    };
    file.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
    if (!file) {
        throw std::runtime_error("Failed to write wrapper metadata");
    }
}

HNSWLoadResult loadWrapperMetadata(HNSWWrapperMetadata& metadata, const std::string& path) {
    std::ifstream file(wrapperMetadataPath(path), std::ios::binary);
    if (!file) {
        return HNSW_LOAD_MISSING_WRAPPER_METADATA;
    }

    file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(metadata))) {
        return HNSW_LOAD_INVALID_WRAPPER_METADATA;
    }

    char extraByte;
    if (file.read(&extraByte, 1)) {
        return HNSW_LOAD_INVALID_WRAPPER_METADATA;
    }

    if (metadata.magic != HNSW_WRAPPER_METADATA_MAGIC ||
        metadata.version != HNSW_WRAPPER_METADATA_VERSION ||
        metadata.dimension <= 0 ||
        (metadata.space_type != HNSW_SPACE_L2 && metadata.space_type != HNSW_SPACE_COSINE)) {
        return HNSW_LOAD_INVALID_WRAPPER_METADATA;
    }

    return HNSW_LOAD_OK;
}

class LabelFilterFunctor : public hnswlib::BaseFilterFunctor {
private:
    void* user_data;
    HNSWLabelFilterFn c_fn;

public:
    LabelFilterFunctor(void* user_data, HNSWLabelFilterFn c_fn) : user_data(user_data), c_fn(c_fn) {}

    bool operator()(hnswlib::labeltype id) override {
        return c_fn(user_data, static_cast<int32_t>(id));
    }
};

class DenseAllowListFilterFunctor : public hnswlib::BaseFilterFunctor {
private:
    const uint8_t* allowlist;
    size_t allowlist_count;

public:
    DenseAllowListFilterFunctor(const uint8_t* allowlist, size_t allowlist_count)
        : allowlist(allowlist), allowlist_count(allowlist_count) {}

    bool operator()(hnswlib::labeltype id) override {
        return id < allowlist_count && allowlist[id] != 0;
    }
};

// Add these functions before the extern "C" block
void saveMetadata(
    const std::vector<std::string>& metadata,
    const std::vector<uint8_t>& has_metadata,
    const std::string& path) {
    std::string metadataPath = path + ".metadata";
    std::ofstream file(metadataPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open metadata file for writing");
    }
    
    // Write number of entries
    size_t size = 0;
    for (uint8_t has_entry : has_metadata) {
        if (has_entry) {
            ++size;
        }
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // Write each entry
    for (size_t id = 0; id < has_metadata.size(); ++id) {
        if (!has_metadata[id]) {
            continue;
        }

        // Write ID
        int stored_id = static_cast<int>(id);
        file.write(reinterpret_cast<const char*>(&stored_id), sizeof(stored_id));
        
        // Write string length and content
        size_t strLen = metadata[id].length();
        file.write(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
        file.write(metadata[id].c_str(), strLen);
    }
}

void loadMetadata(
    std::vector<std::string>& metadata,
    std::vector<uint8_t>& has_metadata,
    const std::string& path) {
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
        
        if (id >= 0 && static_cast<size_t>(id) < metadata.size()) {
            metadata[id] = str;
            has_metadata[id] = 1;
        }
    }
}

extern "C" {
    using namespace hnswlib;
    
    void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction, HNSWSpaceType space_type, bool allow_replace_deleted) {
        hnswlib::SpaceInterface<float>* space;
        if (space_type == HNSW_SPACE_COSINE) {
            space = new hnswlib::InnerProductSpace(dim);  // Use InnerProductSpace for cosine similarity
        } else {
            space = new hnswlib::L2Space(dim);
        }
        
        // random_seed uses hnswlib's default (100); allow_replace_deleted enables reuse of soft-deleted slots.
        hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction, 100, allow_replace_deleted);
        HNSWIndexWrapper* wrapper = new HNSWIndexWrapper{
            index,
            space,
            dim,
            std::vector<std::string>(max_elements),
            std::vector<uint8_t>(max_elements, 0),
            space_type,
            dim,
            space_type
        };
        return static_cast<void*>(wrapper);
    }
    
    void hnswlib_free_index(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        delete wrapper->index;
        delete wrapper->space;  // Free the space interface
        delete wrapper;
    }
    
    int hnswlib_add_point(void* index_ptr, const float* vector, int id, bool replace_deleted) {
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
            
            // Reject duplicate labels unless the index was created with in-place replacement enabled.
            if (wrapper->index->label_lookup_.find(id) != wrapper->index->label_lookup_.end()
                && !wrapper->index->allow_replace_deleted_) {
                return -3;  // Point with ID already exists
            }
            
            wrapper->index->addPoint(vector, id, replace_deleted);
            return 0;  // Success
        } catch (const std::exception& e) {
            return -4;  // General error
        }
    }
    
    int hnswlib_add_point_with_metadata(void* index_ptr, const float* vector, int id, const char* metadata, bool replace_deleted) {
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
            
            // Reject duplicate labels unless the index was created with in-place replacement enabled.
            if (wrapper->index->label_lookup_.find(id) != wrapper->index->label_lookup_.end()
                && !wrapper->index->allow_replace_deleted_) {
                return -3;  // Point with ID already exists
            }
            
            wrapper->index->addPoint(vector, id, replace_deleted);
            if (metadata != nullptr) {
                wrapper->metadata[id] = std::string(metadata);
                wrapper->has_metadata[id] = 1;
            }
            return 0;  // Success
        } catch (const std::exception& e) {
            return -4;  // General error
        }
    }
    
    const char* hnswlib_get_metadata(void* index_ptr, int id) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        if (id >= 0 &&
            static_cast<size_t>(id) < wrapper->metadata.size() &&
            wrapper->has_metadata[id]) {
            return wrapper->metadata[id].c_str();
        }
        return nullptr;
    }
    
    void hnswlib_set_metadata(void* index_ptr, int id, const char* metadata) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        if (id < 0 || static_cast<size_t>(id) >= wrapper->metadata.size()) {
            return;
        }
        if (metadata != nullptr) {
            wrapper->metadata[id] = std::string(metadata);
            wrapper->has_metadata[id] = 1;
        } else {
            wrapper->metadata[id].clear();
            wrapper->has_metadata[id] = 0;
        }
    }
    
    void hnswlib_remove_metadata(void* index_ptr, int id) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        if (id >= 0 && static_cast<size_t>(id) < wrapper->metadata.size()) {
            wrapper->metadata[id].clear();
            wrapper->has_metadata[id] = 0;
        }
    }
    
    int hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        std::priority_queue<std::pair<float, labeltype>> result = wrapper->index->searchKnn(query, k);

        int count = static_cast<int>(result.size());
        int index = count;
        while (!result.empty()) {
            const auto& top = result.top();
            --index;
            ids[index] = static_cast<int>(top.second);
            distances[index] = top.first;
            result.pop();
        }

        return count;
    }

    int hnswlib_search_knn_with_label_filter(
        void* index_ptr,
        const float* query,
        int* ids,
        float* distances,
        int k,
        void* user_data,
        HNSWLabelFilterFn filter_fn) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        LabelFilterFunctor filter(user_data, filter_fn);
        std::priority_queue<std::pair<float, labeltype>> result =
            wrapper->index->searchKnn(query, static_cast<size_t>(k), &filter);

        int count = static_cast<int>(result.size());
        int index = count;
        while (!result.empty()) {
            const auto& top = result.top();
            --index;
            ids[index] = static_cast<int>(top.second);
            distances[index] = top.first;
            result.pop();
        }

        return count;
    }

    int hnswlib_search_knn_with_allowlist(
        void* index_ptr,
        const float* query,
        int* ids,
        float* distances,
        int k,
        const uint8_t* allowlist,
        int allowlistCount) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        DenseAllowListFilterFunctor filter(allowlist, static_cast<size_t>(allowlistCount));
        std::priority_queue<std::pair<float, labeltype>> result =
            wrapper->index->searchKnn(query, static_cast<size_t>(k), &filter);

        int count = static_cast<int>(result.size());
        int index = count;
        while (!result.empty()) {
            const auto& top = result.top();
            --index;
            ids[index] = static_cast<int>(top.second);
            distances[index] = top.first;
            result.pop();
        }

        return count;
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
            saveWrapperMetadata(*wrapper, path);
            saveMetadata(wrapper->metadata, wrapper->has_metadata, path);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_load_index(void* index_ptr, const char* path, int max_elements) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            HNSWWrapperMetadata metadata{};
            HNSWLoadResult metadataResult = loadWrapperMetadata(metadata, path);
            if (metadataResult != HNSW_LOAD_OK) {
                return metadataResult;
            }

            wrapper->last_loaded_dimension = metadata.dimension;
            wrapper->last_loaded_space_type = static_cast<HNSWSpaceType>(metadata.space_type);

            if (wrapper->last_loaded_space_type != wrapper->space_type) {
                return HNSW_LOAD_SPACE_MISMATCH;
            }

            if (wrapper->last_loaded_dimension != wrapper->dimension) {
                return HNSW_LOAD_DIMENSION_MISMATCH;
            }

            wrapper->index->loadIndex(path, wrapper->space, max_elements);
            wrapper->metadata.assign(wrapper->index->max_elements_, std::string());
            wrapper->has_metadata.assign(wrapper->index->max_elements_, 0);
            loadMetadata(wrapper->metadata, wrapper->has_metadata, path);
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

    int hnswlib_label_exists(void* index_ptr, int id) {
        if (id < 0) {
            return -1;
        }
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            auto label = static_cast<hnswlib::labeltype>(id);
            std::unique_lock<std::mutex> lock(wrapper->index->label_lookup_lock);
            return wrapper->index->label_lookup_.find(label)
                       != wrapper->index->label_lookup_.end() ? 1 : 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_label_is_active(void* index_ptr, int id) {
        if (id < 0) {
            return -1;
        }
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            auto label = static_cast<hnswlib::labeltype>(id);
            std::unique_lock<std::mutex> lock(wrapper->index->label_lookup_lock);
            auto search = wrapper->index->label_lookup_.find(label);
            if (search == wrapper->index->label_lookup_.end()) {
                return 0;
            }
            auto internalId = search->second;
            lock.unlock();
            return wrapper->index->isMarkedDeleted(internalId) ? 0 : 1;
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

            wrapper->metadata.resize(new_size);
            wrapper->has_metadata.resize(new_size, 0);
            
            return 0;
        } catch (...) {
            return -4;  // General error
        }
    }

    HNSWSpaceType hnswlib_get_space_type(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->space_type;
    }

    HNSWSpaceType hnswlib_get_last_loaded_space_type(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->last_loaded_space_type;
    }

    int hnswlib_get_dim(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->dimension;
    }

    int hnswlib_get_last_loaded_dim(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return wrapper->last_loaded_dimension;
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

} 
