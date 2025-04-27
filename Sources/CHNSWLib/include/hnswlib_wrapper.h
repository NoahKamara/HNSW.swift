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
 * @return 0 on success, negative value on error:
 *         -1: Index not initialized
 *         -2: ID exceeds maximum elements
 *         -3: Point with ID already exists
 *         -4: General error
 */
int hnswlib_add_point(void* index_ptr, const float* vector, int id);

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

/**
 * Sets the current filter function for the index.
 * 
 * @param index_ptr Pointer to the index
 * @param filter_func A function that takes a metadata string and returns true if the vector should be included in results
 */
void hnswlib_set_filter(void* index_ptr, bool (*filter_func)(const char*));

/**
 * Searches for k nearest neighbors of a query vector with the current filter.
 * 
 * @param index_ptr Pointer to the index
 * @param query The query vector
 * @param ids Array to store the IDs of the nearest neighbors
 * @param distances Array to store the distances to the nearest neighbors
 * @param k The number of nearest neighbors to find
 */
void hnswlib_search_knn_with_filter(void* index_ptr, const float* query, int* ids, float* distances, int k);

/**
 * Sets the query time accuracy/speed trade-off parameter.
 * 
 * @param index_ptr Pointer to the index
 * @param ef The ef parameter value
 * @return 0 on success, non-zero on failure
 */
int hnswlib_set_ef(void* index_ptr, int ef);

/**
 * Saves the index to a file.
 * 
 * @param index_ptr Pointer to the index
 * @param path The path where to save the index
 * @return 0 on success, non-zero on failure
 */
int hnswlib_save_index(void* index_ptr, const char* path);

/**
 * Loads an index from a file.
 * 
 * @param index_ptr Pointer to the index
 * @param path The path to the index file
 * @param max_elements The maximum number of elements that can be stored in the index
 * @return 0 on success, non-zero on failure
 */
int hnswlib_load_index(void* index_ptr, const char* path, int max_elements);

/**
 * Marks an element as deleted.
 * 
 * @param index_ptr Pointer to the index
 * @param id The ID of the element to mark as deleted
 * @return 0 on success, non-zero on failure
 */
int hnswlib_mark_deleted(void* index_ptr, int id);

/**
 * Unmarks an element as deleted.
 * 
 * @param index_ptr Pointer to the index
 * @param id The ID of the element to unmark
 * @return 0 on success, non-zero on failure
 */
int hnswlib_unmark_deleted(void* index_ptr, int id);

/**
 * Changes the maximum capacity of the index.
 * 
 * @param index_ptr Pointer to the index
 * @param new_size The new maximum capacity
 * @return 0 on success, non-zero on failure
 */
int hnswlib_resize_index(void* index_ptr, int new_size);

/**
 * Gets the space name of the index.
 * 
 * @param index_ptr Pointer to the index
 * @return The space name ("l2", "ip", or "cosine")
 */
const char* hnswlib_get_space(void* index_ptr);

/**
 * Gets the dimensionality of the space.
 * 
 * @param index_ptr Pointer to the index
 * @return The dimensionality
 */
int hnswlib_get_dim(void* index_ptr);

/**
 * Gets the M parameter (maximum number of outgoing connections).
 * 
 * @param index_ptr Pointer to the index
 * @return The M parameter value
 */
unsigned long hnswlib_get_M(void* index_ptr);

/**
 * Gets the ef_construction parameter.
 * 
 * @param index_ptr Pointer to the index
 * @return The ef_construction parameter value
 */
unsigned long hnswlib_get_ef_construction(void* index_ptr);

/**
 * Gets the maximum number of elements that can be stored in the index.
 * 
 * @param index_ptr Pointer to the index
 * @return The maximum number of elements
 */
unsigned long hnswlib_get_max_elements(void* index_ptr);

/**
 * Gets the current number of elements in the index.
 * 
 * @param index_ptr Pointer to the index
 * @return The current number of elements
 */
unsigned long hnswlib_get_current_count(void* index_ptr);

/**
 * Adds a vector to the index with the specified ID and metadata.
 * 
 * @param index_ptr Pointer to the index
 * @param vector The vector to add (array of floats)
 * @param id The integer ID to associate with the vector
 * @param metadata The metadata string to associate with the vector
 * @return 0 on success, negative value on error:
 *         -1: Index not initialized
 *         -2: ID exceeds maximum elements
 *         -3: Point with ID already exists
 *         -4: General error
 */
int hnswlib_add_point_with_metadata(void* index_ptr, const float* vector, int id, const char* metadata);

/**
 * Gets the metadata associated with a vector ID.
 * 
 * @param index_ptr Pointer to the index
 * @param id The ID of the vector
 * @return The metadata string, or nullptr if no metadata exists
 */
const char* hnswlib_get_metadata(void* index_ptr, int id);

/**
 * Sets or updates the metadata for a vector ID.
 * 
 * @param index_ptr Pointer to the index
 * @param id The ID of the vector
 * @param metadata The metadata string to associate with the vector
 */
void hnswlib_set_metadata(void* index_ptr, int id, const char* metadata);

/**
 * Removes the metadata associated with a vector ID.
 * 
 * @param index_ptr Pointer to the index
 * @param id The ID of the vector
 */
void hnswlib_remove_metadata(void* index_ptr, int id);

#ifdef __cplusplus
}
#endif

#endif // HNSWLIB_WRAPPER_H 