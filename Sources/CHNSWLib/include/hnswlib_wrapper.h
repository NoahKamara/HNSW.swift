#ifndef HNSWLIB_WRAPPER_H
#define HNSWLIB_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new HNSW index with the specified parameters.
 * 
 * @param dim The dimensionality of the vectors
 * @param max_elements The maximum number of elements that can be stored in the index
 * @param M The maximum number of outgoing connections in the graph
 * @param ef_construction The construction time/accuracy trade-off parameter
 * @return A pointer to the created index
 */
void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction);

/**
 * Frees the memory allocated for an HNSW index.
 * 
 * @param index_ptr Pointer to the index to be freed
 */
void hnswlib_free_index(void* index_ptr);

/**
 * Adds a vector to the index with the specified ID.
 * 
 * @param index_ptr Pointer to the index
 * @param vector The vector to add (array of floats)
 * @param id The integer ID to associate with the vector
 */
void hnswlib_add_point(void* index_ptr, const float* vector, int id);

/**
 * Searches for k nearest neighbors of a query vector.
 * 
 * @param index_ptr Pointer to the index
 * @param query The query vector (array of floats)
 * @param ids Array to store the IDs of the k nearest neighbors
 * @param distances Array to store the distances to the k nearest neighbors
 * @param k The number of nearest neighbors to find
 */
void hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k);

#ifdef __cplusplus
}
#endif

#endif // HNSWLIB_WRAPPER_H 