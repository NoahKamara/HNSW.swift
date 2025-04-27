#include <cstdint>
#include "hnswlib_wrapper.h"
#include "hnswlib/hnswlib.h"
#include <vector>
#include <queue>

struct HNSWIndexWrapper {
    hnswlib::HierarchicalNSW<float>* index;
    int dimension;
};

extern "C" {
    using namespace hnswlib;
    
    void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction) {
        SpaceInterface<float>* space = new L2Space(dim);
        HierarchicalNSW<float>* index = new HierarchicalNSW<float>(space, max_elements, M, ef_construction);
        HNSWIndexWrapper* wrapper = new HNSWIndexWrapper{index, dim};
        return static_cast<void*>(wrapper);
    }
    
    void hnswlib_free_index(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        delete wrapper->index;
        delete wrapper;
    }
    
    void hnswlib_add_point(void* index_ptr, const float* vector, int id) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        wrapper->index->addPoint(vector, id);
    }
    
    void hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        std::priority_queue<std::pair<float, labeltype>> result = wrapper->index->searchKnn(query, k);
        
        std::vector<std::pair<float, labeltype>> sorted_results;
        while (!result.empty()) {
            sorted_results.push_back(result.top());
            result.pop();
        }
        
        for (int i = sorted_results.size() - 1; i >= 0; i--) {
            int idx = sorted_results.size() - 1 - i;
            distances[idx] = sorted_results[i].first;
            ids[idx] = static_cast<int>(sorted_results[i].second);
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
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int hnswlib_load_index(void* index_ptr, const char* path, int max_elements) {
        try {
            auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
            wrapper->index->loadIndex(path, nullptr, max_elements);
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
            wrapper->index->resizeIndex(new_size);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    const char* hnswlib_get_space(void* index_ptr) {
        auto* wrapper = static_cast<HNSWIndexWrapper*>(index_ptr);
        return "l2"; // Currently only L2 space is supported
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
} 
