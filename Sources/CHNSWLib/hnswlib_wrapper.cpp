#include <cstdint>
#include "hnswlib_wrapper.h"
#include "hnswlib/hnswlib.h"
#include <vector>
#include <queue>

extern "C" {
    using namespace hnswlib;
    
    void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction) {
        SpaceInterface<float>* space = new L2Space(dim);
        HierarchicalNSW<float>* index = new HierarchicalNSW<float>(space, max_elements, M, ef_construction);
        return static_cast<void*>(index);
    }
    
    void hnswlib_free_index(void* index_ptr) {
        auto* index = static_cast<HierarchicalNSW<float>*>(index_ptr);
        delete index;
    }
    
    void hnswlib_add_point(void* index_ptr, const float* vector, int id) {
        auto* index = static_cast<HierarchicalNSW<float>*>(index_ptr);
        index->addPoint(vector, id);
    }
    
    void hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k) {
        auto* index = static_cast<HierarchicalNSW<float>*>(index_ptr);
        std::priority_queue<std::pair<float, labeltype>> result = index->searchKnn(query, k);
        
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
} 