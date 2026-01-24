
#include <assert.h>

#include <algorithm>
#include <vector>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;
  cache_obj_t *pointer;

  void *embedding_manager;  // EmbeddingManager*

  int num_candidates;  // default: 8
  int recent_window;   // default: 16
} SieveEmb_params_t;

// Default parameters
static const int DEFAULT_NUM_CANDIDATES = 8;
static const int DEFAULT_RECENT_WINDOW = 16;

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************
static void SieveEmb_free(cache_t *cache);
static bool SieveEmb_get(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmb_find(cache_t *cache, const request_t *req,
                                  const bool update_cache);
static cache_obj_t *SieveEmb_insert(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmb_to_evict(cache_t *cache, const request_t *req);
static void SieveEmb_evict(cache_t *cache, const request_t *req);
static bool SieveEmb_remove(cache_t *cache, const obj_id_t obj_id);
static void SieveEmb_parse_params(cache_t *cache,
                                  const char *cache_specific_params);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ****                       init, free, get                         ****
// ***********************************************************************

static void SieveEmb_parse_params(cache_t *cache,
                                  const char *cache_specific_params) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  char *params_str = NULL;
  char *params_str_orig = NULL;  // save original pointer for free
  if (cache_specific_params != NULL) {
    params_str = strdup(cache_specific_params);
    params_str_orig = params_str;
  }

  while (params_str != NULL && params_str[0] != '\0') {
    /* different parameters are separated by comma,
     * key and value are separated by = */
    char *key = strsep(&params_str, "=");
    char *value = strsep(&params_str, ",");

    // skip spaces
    while (key != NULL && *key == ' ') key++;
    while (value != NULL && *value == ' ') value++;

    if (key == NULL) {
      break;
    }

    // Handle print parameter (no value needed)
    if (strcasecmp(key, "print") == 0) {
      printf("SieveEmb parameters: num-candidates=%d, recent-window=%d\n",
             params->num_candidates, params->recent_window);
      continue;
    }

    if (value == NULL) {
      break;
    }

    if (strcasecmp(key, "num-candidates") == 0 ||
        strcasecmp(key, "num_candidates") == 0) {
      params->num_candidates = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0 ||
               strcasecmp(key, "recent_window") == 0) {
      params->recent_window = atoi(value);
    } else {
      ERROR("SieveEmb does not have parameter %s\n", key);
      abort();
    }
  }

  if (params_str_orig != NULL) {
    free(params_str_orig);
  }
}

/**
 * @brief initialize cache
 *
 * @param ccache_params some common cache parameters
 * @param cache_specific_params cache specific parameters, see parse_params
 * function or use -e "print" with the cachesim binary
 */
cache_t *SieveEmb_init(const common_cache_params_t ccache_params,
                       const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("SieveEmb", ccache_params, cache_specific_params);
  cache->cache_init = SieveEmb_init;
  cache->cache_free = SieveEmb_free;
  cache->get = SieveEmb_get;
  cache->find = SieveEmb_find;
  cache->insert = SieveEmb_insert;
  cache->evict = SieveEmb_evict;
  cache->remove = SieveEmb_remove;
  cache->to_evict = SieveEmb_to_evict;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 1;
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = my_malloc(SieveEmb_params_t);
  memset(cache->eviction_params, 0, sizeof(SieveEmb_params_t));
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  params->pointer = NULL;
  params->q_head = NULL;
  params->q_tail = NULL;
  params->num_candidates = DEFAULT_NUM_CANDIDATES;
  params->recent_window = DEFAULT_RECENT_WINDOW;

  // Parse any custom parameters
  if (cache_specific_params != NULL) {
    SieveEmb_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  params->embedding_manager = static_cast<void *>(emb);

  return cache;
}

/**
 * free resources used by this cache
 *
 * @param cache
 */
static void SieveEmb_free(cache_t *cache) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

/**
 * @brief this function is the user facing API
 */
static bool SieveEmb_get(cache_t *cache, const request_t *req) {
  bool ck_hit = cache_get_base(cache, req);
  return ck_hit;
}

// ***********************************************************************
// ****                                                               ****
// ****       developer facing APIs (used by cache developer)         ****
// ****                                                               ****
// ***********************************************************************

/**
 * @brief find an object in the cache
 *
 * @param cache
 * @param req
 * @param update_cache whether to update the cache,
 *  if true, the object is promoted
 *  and if the object is expired, it is removed from the cache
 * @return the object or NULL if not found
 */
static cache_obj_t *SieveEmb_find(cache_t *cache, const request_t *req,
                                  const bool update_cache) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);
  if (cache_obj != NULL && update_cache) {
    cache_obj->sieve.freq = 1;
    // Update embedding on cache hit
    emb->on_access(req->obj_id);
  }

  return cache_obj;
}

/**
 * @brief insert an object into the cache,
 * update the hash table and cache metadata
 * this function assumes the cache has enough space
 * eviction should be
 * performed before calling this function
 *
 * @param cache
 * @param req
 * @return the inserted object
 */
static cache_obj_t *SieveEmb_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);
  obj->sieve.freq = 0;

  // Track recent accesses after insert
  emb->update_recent(req->obj_id);

  return obj;
}

/**
 * @brief find the object to be evicted using embedding-augmented selection
 *
 * Modified SIEVE eviction:
 * 1. Scan from hand pointer, collect up to num_candidates objects with visited
 * bit = false
 * 2. For each unvisited candidate, compute similarity score vs recent accesses
 * 3. Select victim with lowest similarity (least related to recent access
 * pattern)
 * 4. If no unvisited candidates found, fallback to baseline SIEVE behavior
 *
 * @param cache the cache
 * @return the object to be evicted
 */
static cache_obj_t *SieveEmb_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  cache_obj_t *pointer = params->pointer;

  // If we have run one full around or first eviction
  if (pointer == NULL) {
    pointer = params->q_tail;
  }

  // Collect unvisited candidates
  std::vector<std::pair<cache_obj_t *, double>> candidates;
  cache_obj_t *scan = pointer;
  cache_obj_t *start = pointer;
  bool first_loop = true;

  while (static_cast<int>(candidates.size()) < params->num_candidates) {
    if (scan == NULL) {
      break;
    }

    // Check if unvisited (freq == 0)
    if (scan->sieve.freq == 0) {
      double score = emb->max_similarity_to_recent(scan->obj_id);
      candidates.push_back({scan, score});
    }

    // Move toward tail (prev), wrap to tail if we hit head
    scan = scan->queue.prev;
    if (scan == NULL) {
      scan = params->q_tail;
    }

    // Check if we've completed a full loop
    if (scan == start && !first_loop) {
      break;
    }
    first_loop = false;
  }

  if (!candidates.empty()) {
    // Find candidate with lowest score (least related to recent accesses)
    auto victim_it = std::min_element(
        candidates.begin(), candidates.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    return victim_it->first;
  }

  // Fallback: all items were visited, use baseline SIEVE behavior
  // Find first object with freq <= to_evict_freq
  int to_evict_freq = 0;
  while (true) {
    pointer = params->pointer == NULL ? params->q_tail : params->pointer;
    while (pointer != NULL && pointer->sieve.freq > to_evict_freq) {
      pointer = pointer->queue.prev;
    }

    if (pointer == NULL) {
      pointer = params->q_tail;
      while (pointer != NULL && pointer->sieve.freq > to_evict_freq) {
        pointer = pointer->queue.prev;
      }
    }

    if (pointer != NULL) {
      return pointer;
    }

    to_evict_freq++;
    if (to_evict_freq > 100) {
      // Safety: should never happen
      return params->q_tail;
    }
  }
}

/**
 * @brief evict an object from the cache
 * it needs to call cache_evict_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param req not used
 */
static void SieveEmb_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);

  // Get the victim using to_evict
  cache_obj_t *obj_to_evict = SieveEmb_to_evict(cache, req);

  // Get current hand position
  cache_obj_t *pointer = params->pointer;
  if (pointer == NULL) {
    pointer = params->q_tail;
  }

  // Advance hand from current position to the victim, clearing visited bits
  cache_obj_t *clear_ptr = pointer;
  while (clear_ptr != NULL && clear_ptr != obj_to_evict) {
    clear_ptr->sieve.freq = 0;
    clear_ptr = clear_ptr->queue.prev;
    if (clear_ptr == NULL) {
      clear_ptr = params->q_tail;
    }
    if (clear_ptr == pointer) {
      // Wrapped around without finding victim - shouldn't happen
      break;
    }
  }

  // Move pointer past the victim
  params->pointer = obj_to_evict->queue.prev;

  remove_obj_from_list(&params->q_head, &params->q_tail, obj_to_evict);
  cache_evict_base(cache, obj_to_evict, true);
}

static void SieveEmb_remove_obj(cache_t *cache, cache_obj_t *obj_to_remove) {
  DEBUG_ASSERT(obj_to_remove != NULL);
  auto *params = static_cast<SieveEmb_params_t *>(cache->eviction_params);
  if (obj_to_remove == params->pointer) {
    params->pointer = obj_to_remove->queue.prev;
  }
  remove_obj_from_list(&params->q_head, &params->q_tail, obj_to_remove);
  cache_remove_obj_base(cache, obj_to_remove, true);
}

/**
 * @brief remove an object from the cache
 * this is different from cache_evict because it is used to for user trigger
 * remove, and eviction is used by the cache to make space for new objects
 *
 * it needs to call cache_remove_obj_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param obj_id
 * @return true if the object is removed, false if the object is not in the
 * cache
 */
static bool SieveEmb_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }

  SieveEmb_remove_obj(cache, obj);

  return true;
}

#ifdef __cplusplus
}
#endif
