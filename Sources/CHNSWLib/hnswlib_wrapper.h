#ifndef HNSWLIB_WRAPPER_H
#define HNSWLIB_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif


void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction);
void hnswlib_free_index(void* index_ptr);
void hnswlib_add_point(void* index_ptr, const float* vector, int id);
void hnswlib_search_knn(void* index_ptr, const float* query, int32_t* ids, float* distances, int k);

#ifdef __cplusplus
}
#endif

#endif // HNSWLIB_WRAPPER_H 
