#ifndef HNSWLIB_WRAPPER_H
#define HNSWLIB_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Space type for HNSW index
 */
typedef enum {
    HNSW_SPACE_L2 = 0,
    HNSW_SPACE_COSINE = 1
} HNSWSpaceType;

/**
 * Result codes returned by hnswlib_load_index.
 */
typedef enum {
    HNSW_LOAD_OK = 0,
    HNSW_LOAD_NATIVE_FAILURE = -1,
    HNSW_LOAD_MISSING_WRAPPER_METADATA = -2,
    HNSW_LOAD_INVALID_WRAPPER_METADATA = -3,
    HNSW_LOAD_SPACE_MISMATCH = -4,
    HNSW_LOAD_DIMENSION_MISMATCH = -5
} HNSWLoadResult;

/**
 * Creates a new HNSW index with the specified parameters.
 * 
 * @param dim The dimensionality of the vectors
 * @param max_elements The maximum number of elements that can be stored in the index
 * @param M The maximum number of outgoing connections in the graph
 * @param ef_construction The construction time/accuracy trade-off parameter
 * @param space_type The space type to use for distance calculations
 * @param allow_replace_deleted When true, soft-deleted slots can be reused by add_point with replace_deleted=true
 * @return A pointer to the created index
 */
void* hnswlib_create_index(int dim, int max_elements, int M, int ef_construction, HNSWSpaceType space_type, bool allow_replace_deleted);

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
 * @param replace_deleted When true, reuse a previously soft-deleted slot instead of growing the index;
 *        requires the index to have been created with allow_replace_deleted=true
 * @return 0 on success, negative value on error:
 *         -1: Index not initialized
 *         -2: ID exceeds maximum elements
 *         -3: Point with ID already exists
 *         -4: General error
 */
int hnswlib_add_point(void* index_ptr, const float* vector, int id, bool replace_deleted);

/**
 * Searches for k nearest neighbors of a query vector.
 * 
 * @param index_ptr Pointer to the index
 * @param query The query vector (array of floats)
 * @param ids Array to store the IDs of the nearest neighbors (ascending distance, nearest first)
 * @param distances Array to store the distances to the nearest neighbors (same order as ids)
 * @param k The number of nearest neighbors to find
 * @param ef Per-query candidate list size.
 * @return The number of valid entries written to ids/distances.
 */
int hnswlib_search_knn(void* index_ptr, const float* query, int* ids, float* distances, int k, int ef);

/**
 * Per-query label filter. @p labelId is the external label (the same integer id used with add_point).
 * Called during the graph walk for each candidate; return true to allow the label in results.
 */
typedef bool (*HNSWLabelFilterFn)(void* userData, int32_t labelId);

/**
 * Searches for k nearest neighbors whose labels pass the filter, evaluated during the graph walk.
 *
 * Valid entries in @p ids and @p distances are ascending by distance (nearest first); fewer than @p k
 * when the filter is selective. Returns the number of valid entries written.
 *
 * @param userData Opaque pointer passed to @p filterFn
 * @param filterFn Called for each candidate label; must not be NULL
 */
int hnswlib_search_knn_with_label_filter(
    void* index_ptr,
    const float* query,
    int* ids,
    float* distances,
    int k,
    int ef,
    void* userData,
    HNSWLabelFilterFn filterFn);

/**
 * Searches for k nearest neighbors whose integer labels are enabled in a dense byte allowlist.
 *
 * A label is allowed when label >= 0, label < allowlistCount, and allowlist[label] is non-zero. Returns the number of
 * valid entries written to ids/distances.
 */
int hnswlib_search_knn_with_allowlist(
    void* index_ptr,
    const float* query,
    int* ids,
    float* distances,
    int k,
    int ef,
    const uint8_t* allowlist,
    int allowlistCount);

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
 * @return HNSWLoadResult
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
 * Returns whether an external label is present in the index (including soft-deleted points).
 *
 * @param index_ptr Pointer to the index
 * @param id The external label (same integer as add_point / mark_deleted)
 * @return 1 if the label is present (including soft-deleted), 0 if absent, -1 if invalid (e.g. id < 0) or failure
 */
int hnswlib_label_exists(void* index_ptr, int id);

/**
 * Returns whether a label is present and not soft-deleted (can appear in an unfiltered search).
 *
 * @param index_ptr Pointer to the index
 * @param id The external label (same integer as add_point / mark_deleted)
 * @return 1 if present and active, 0 if absent or soft-deleted, -1 if invalid (e.g. id < 0) or failure
 */
int hnswlib_label_is_active(void* index_ptr, int id);

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
 * @param replace_deleted When true, reuse a previously soft-deleted slot instead of growing the index;
 *        requires the index to have been created with allow_replace_deleted=true
 * @return 0 on success, negative value on error:
 *         -1: Index not initialized
 *         -2: ID exceeds maximum elements
 *         -3: Point with ID already exists
 *         -4: General error
 */
int hnswlib_add_point_with_metadata(void* index_ptr, const float* vector, int id, const char* metadata, bool replace_deleted);

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

/**
 * Gets the space type of the index.
 * 
 * @param index_ptr Pointer to the index
 * @return The space type of the index
 */
HNSWSpaceType hnswlib_get_space_type(void* index_ptr);

/**
 * Gets the space type read from the most recent wrapper metadata sidecar.
 *
 * @param index_ptr Pointer to the index
 * @return The sidecar space type, meaningful after hnswlib_load_index has read valid wrapper metadata
 */
HNSWSpaceType hnswlib_get_last_loaded_space_type(void* index_ptr);

/**
 * Gets the dimension read from the most recent wrapper metadata sidecar.
 *
 * @param index_ptr Pointer to the index
 * @return The sidecar dimension, meaningful after hnswlib_load_index has read valid wrapper metadata
 */
int hnswlib_get_last_loaded_dim(void* index_ptr);

#ifdef __cplusplus
}
#endif

#endif // HNSWLIB_WRAPPER_H 
