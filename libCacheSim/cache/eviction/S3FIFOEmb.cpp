//
//  S3-FIFOEmb: A variant of S3-FIFO that uses embedding similarity
//  to break ties during Main FIFO eviction.
//
//  Small FIFO eviction remains unchanged since those objects have
//  uninformative embeddings (accessed < 2 times).
//
//  Main FIFO eviction: Collect K candidates with freq == 0,
//  evict the one with lowest similarity to recent accesses.
//
//  S3FIFOEmb.cpp
//  libCacheSim
//

#include <algorithm>
#include <vector>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cache_t *small_fifo;
  cache_t *ghost_fifo;
  cache_t *main_fifo;
  bool hit_on_ghost;

  int move_to_main_threshold;
  double small_size_ratio;
  double ghost_size_ratio;

  bool has_evicted;
  request_t *req_local;

  // Embedding fields
  void *embedding_manager;  // EmbeddingManager*
  int num_candidates;       // default: 8
  int recent_window;        // default: 16
} S3FIFOEmb_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2,"
    "num-candidates=8,recent-window=16";

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************
static void S3FIFOEmb_free(cache_t *cache);
static bool S3FIFOEmb_get(cache_t *cache, const request_t *req);

static cache_obj_t *S3FIFOEmb_find(cache_t *cache, const request_t *req,
                                   const bool update_cache);
static cache_obj_t *S3FIFOEmb_insert(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFOEmb_to_evict(cache_t *cache, const request_t *req);
static void S3FIFOEmb_evict(cache_t *cache, const request_t *req);
static bool S3FIFOEmb_remove(cache_t *cache, const obj_id_t obj_id);
static inline int64_t S3FIFOEmb_get_occupied_byte(const cache_t *cache);
static inline int64_t S3FIFOEmb_get_n_obj(const cache_t *cache);
static inline bool S3FIFOEmb_can_insert(cache_t *cache, const request_t *req);
static void S3FIFOEmb_parse_params(cache_t *cache,
                                   const char *cache_specific_params);

static void S3FIFOEmb_evict_small(cache_t *cache, const request_t *req);
static void S3FIFOEmb_evict_main(cache_t *cache, const request_t *req);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ***********************************************************************

cache_t *S3FIFOEmb_init(const common_cache_params_t ccache_params,
                        const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("S3FIFOEmb", ccache_params, cache_specific_params);
  cache->cache_init = S3FIFOEmb_init;
  cache->cache_free = S3FIFOEmb_free;
  cache->get = S3FIFOEmb_get;
  cache->find = S3FIFOEmb_find;
  cache->insert = S3FIFOEmb_insert;
  cache->evict = S3FIFOEmb_evict;
  cache->remove = S3FIFOEmb_remove;
  cache->to_evict = S3FIFOEmb_to_evict;
  cache->get_n_obj = S3FIFOEmb_get_n_obj;
  cache->get_occupied_byte = S3FIFOEmb_get_occupied_byte;
  cache->can_insert = S3FIFOEmb_can_insert;

  cache->obj_md_size = 0;

  cache->eviction_params = malloc(sizeof(S3FIFOEmb_params_t));
  memset(cache->eviction_params, 0, sizeof(S3FIFOEmb_params_t));
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  params->req_local = new_request();
  params->hit_on_ghost = false;
  params->num_candidates = 8;
  params->recent_window = 16;

  S3FIFOEmb_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    S3FIFOEmb_parse_params(cache, cache_specific_params);
  }

  int64_t small_fifo_size =
      (int64_t)ccache_params.cache_size * params->small_size_ratio;
  int64_t main_fifo_size = ccache_params.cache_size - small_fifo_size;
  int64_t ghost_fifo_size =
      (int64_t)(ccache_params.cache_size * params->ghost_size_ratio);

  common_cache_params_t ccache_params_local = ccache_params;
  ccache_params_local.cache_size = small_fifo_size;
  params->small_fifo = FIFO_init(ccache_params_local, NULL);
  params->has_evicted = false;

  if (ghost_fifo_size > 0) {
    ccache_params_local.cache_size = ghost_fifo_size;
    params->ghost_fifo = FIFO_init(ccache_params_local, NULL);
    snprintf(params->ghost_fifo->cache_name, CACHE_NAME_ARRAY_LEN,
             "FIFO-ghost");
  } else {
    params->ghost_fifo = NULL;
  }

  ccache_params_local.cache_size = main_fifo_size;
  params->main_fifo = FIFO_init(ccache_params_local, NULL);

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  params->embedding_manager = static_cast<void *>(emb);

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "S3FIFOEmb-%.4lf-%d-%d-%d",
           params->small_size_ratio, params->move_to_main_threshold,
           params->num_candidates, params->recent_window);

  return cache;
}

/**
 * free resources used by this cache
 *
 * @param cache
 */
static void S3FIFOEmb_free(cache_t *cache) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);

  // Delete embedding manager
  auto *emb =
      static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
  delete emb;

  free_request(params->req_local);
  params->small_fifo->cache_free(params->small_fifo);
  if (params->ghost_fifo != NULL) {
    params->ghost_fifo->cache_free(params->ghost_fifo);
  }
  params->main_fifo->cache_free(params->main_fifo);
  free(cache->eviction_params);
  cache_struct_free(cache);
}

/**
 * @brief this function is the user facing API
 * it performs the following logic
 *
 * ```
 * if obj in cache:
 *    update_metadata
 *    return true
 * else:
 *    if cache does not have enough space:
 *        evict until it has space to insert
 *    insert the object
 *    return false
 * ```
 *
 * @param cache
 * @param req
 * @return true if cache hit, false if cache miss
 */
static bool S3FIFOEmb_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  DEBUG_ASSERT(params->small_fifo->get_occupied_byte(params->small_fifo) +
                   params->main_fifo->get_occupied_byte(params->main_fifo) <=
               cache->cache_size);

  // Call on_access for EVERY request (hit or miss) to:
  // 1. Perturb context
  // 2. Track access count and create/update embeddings
  emb->on_access(req->obj_id);

  bool cache_hit = cache_get_base(cache, req);

  return cache_hit;
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
static cache_obj_t *S3FIFOEmb_find(cache_t *cache, const request_t *req,
                                   const bool update_cache) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  auto *emb =
      static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // if update cache is false, we only check the fifo and main caches
  if (!update_cache) {
    cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, false);
    if (obj != NULL) {
      return obj;
    }
    obj = params->main_fifo->find(params->main_fifo, req, false);
    if (obj != NULL) {
      return obj;
    }
    return NULL;
  }

  /* update cache is true from now */
  params->hit_on_ghost = false;
  cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
    return obj;
  }

  if (params->ghost_fifo != NULL &&
      params->ghost_fifo->remove(params->ghost_fifo, req->obj_id)) {
    // if object in ghost_fifo, remove will return true
    params->hit_on_ghost = true;
  }

  obj = params->main_fifo->find(params->main_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
  }

  return obj;
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
static cache_obj_t *S3FIFOEmb_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  auto *emb =
      static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
  cache_obj_t *obj = NULL;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (params->hit_on_ghost) {
    /* insert into main FIFO */
    params->hit_on_ghost = false;
    obj = main_fifo->insert(main_fifo, req);
  } else {
    /* insert into small fifo */
    // NOTE: Inserting an object whose size equals the size of small fifo is
    // NOT allowed. Doing so would completely fill the small fifo, causing all
    // objects in small fifo to be evicted. This scenario may occur
    // when using a tiny cache size.
    if (req->obj_size >= small_fifo->cache_size) {
      return NULL;
    }

    if (!params->has_evicted &&
        small_fifo->get_occupied_byte(small_fifo) >= small_fifo->cache_size) {
      obj = main_fifo->insert(main_fifo, req);
    } else {
      obj = small_fifo->insert(small_fifo, req);
    }
  }

  obj->S3FIFO.freq = 0;

  return obj;
}

/**
 * @brief find the object to be evicted
 * this function does not actually evict the object or update metadata
 * not all eviction algorithms support this function
 * because the eviction logic cannot be decoupled from finding eviction
 * candidate, so use assert(false) if you cannot support this function
 *
 * @param cache the cache
 * @return the object to be evicted
 */
static cache_obj_t *S3FIFOEmb_to_evict(cache_t *cache, const request_t *req) {
  assert(false);
  return NULL;
}

/**
 * @brief evict from small FIFO (unchanged from S3-FIFO)
 * if object in the small is accessed (freq >= threshold),
 *     reinsert to main FIFO,
 * else
 *     evict and insert to the ghost
 */
static void S3FIFOEmb_evict_small(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  cache_t *small_fifo = params->small_fifo;
  cache_t *ghost_fifo = params->ghost_fifo;
  cache_t *main_fifo = params->main_fifo;

  bool has_evicted = false;
  while (!has_evicted && small_fifo->get_occupied_byte(small_fifo) > 0) {
    cache_obj_t *obj_to_evict = small_fifo->to_evict(small_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    // need to copy the object before it is evicted
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    if (obj_to_evict->S3FIFO.freq >= params->move_to_main_threshold) {
      main_fifo->insert(main_fifo, params->req_local);
    } else {
      // insert to ghost
      if (ghost_fifo != NULL) {
        ghost_fifo->get(ghost_fifo, params->req_local);
      }
      has_evicted = true;
    }

    // remove from small fifo, but do not update stat
    bool removed = small_fifo->remove(small_fifo, params->req_local->obj_id);
    DEBUG_ASSERT(removed);
  }
}

/**
 * @brief evict from main FIFO with embedding-based tie-breaking
 *
 * Modified eviction:
 * 1. Scan from head, collect freq==0 candidates WITHOUT removing them
 * 2. Score each candidate by similarity to recent accesses
 * 3. Evict the one with lowest similarity
 * 4. If no freq==0 candidates found, continue scanning (fallback)
 */
static void S3FIFOEmb_evict_main(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  auto *emb =
      static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
  cache_t *main_fifo = params->main_fifo;

  // Access main_fifo's internal queue directly
  auto *fifo_params = static_cast<FIFO_params_t *>(main_fifo->eviction_params);

  // Collect freq==0 candidates by scanning from tail (FIFO eviction order)
  std::vector<std::pair<cache_obj_t *, double>> candidates;
  cache_obj_t *obj = fifo_params->q_tail;  // Start from tail (oldest)

  while (obj != NULL &&
         static_cast<int>(candidates.size()) < params->num_candidates) {
    cache_obj_t *prev = obj->queue.prev;  // save prev before potential removal

    if (obj->S3FIFO.freq >= 1) {
      // Give second chance: remove and reinsert at head (newest position)
      copy_cache_obj_to_request(params->req_local, obj);
      int freq = obj->S3FIFO.freq;
      main_fifo->remove(main_fifo, obj->obj_id);
      cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
      // clock with 2-bit counter
      new_obj->S3FIFO.freq = MIN(freq, 3) - 1;
    } else {
      // freq == 0: add to candidates, DO NOT remove
      double score = emb->max_similarity_to_recent(obj->obj_id);
      candidates.push_back({obj, score});
    }

    obj = prev;
  }

  // If we found candidates, evict the one with lowest similarity
  if (!candidates.empty()) {
    auto victim_it = std::min_element(
        candidates.begin(), candidates.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });

    main_fifo->remove(main_fifo, victim_it->first->obj_id);
    return;
  }

  // Fallback: no freq==0 candidates found in first num_candidates items,
  // continue with standard second-chance eviction
  bool has_evicted = false;
  while (!has_evicted && main_fifo->get_occupied_byte(main_fifo) > 0) {
    cache_obj_t *obj_to_evict = main_fifo->to_evict(main_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    int freq = obj_to_evict->S3FIFO.freq;
    copy_cache_obj_to_request(params->req_local, obj_to_evict);
    if (freq >= 1) {
      // we need to evict first because the object to insert has the same obj_id
      main_fifo->remove(main_fifo, obj_to_evict->obj_id);
      obj_to_evict = NULL;

      cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
      // clock with 2-bit counter
      new_obj->S3FIFO.freq = MIN(freq, 3) - 1;

    } else {
      bool removed = main_fifo->remove(main_fifo, obj_to_evict->obj_id);
      DEBUG_ASSERT(removed);

      has_evicted = true;
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
 * @param evicted_obj if not NULL, return the evicted object to caller
 */
static void S3FIFOEmb_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  params->has_evicted = true;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (main_fifo->get_occupied_byte(main_fifo) > main_fifo->cache_size ||
      small_fifo->get_occupied_byte(small_fifo) == 0) {
    S3FIFOEmb_evict_main(cache, req);
  } else {
    S3FIFOEmb_evict_small(cache, req);
  }
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
static bool S3FIFOEmb_remove(cache_t *cache, const obj_id_t obj_id) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  bool removed = false;
  removed = removed || params->small_fifo->remove(params->small_fifo, obj_id);
  removed = removed || (params->ghost_fifo &&
                        params->ghost_fifo->remove(params->ghost_fifo, obj_id));
  removed = removed || params->main_fifo->remove(params->main_fifo, obj_id);

  return removed;
}

static inline int64_t S3FIFOEmb_get_occupied_byte(const cache_t *cache) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  return params->small_fifo->get_occupied_byte(params->small_fifo) +
         params->main_fifo->get_occupied_byte(params->main_fifo);
}

static inline int64_t S3FIFOEmb_get_n_obj(const cache_t *cache) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);
  return params->small_fifo->get_n_obj(params->small_fifo) +
         params->main_fifo->get_n_obj(params->main_fifo);
}

static inline bool S3FIFOEmb_can_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);

  return req->obj_size <= params->small_fifo->cache_size &&
         cache_can_insert_default(cache, req);
}

// ***********************************************************************
// ****                                                               ****
// ****                parameter set up functions                     ****
// ****                                                               ****
// ***********************************************************************
static const char *S3FIFOEmb_current_params(S3FIFOEmb_params_t *params) {
  static __thread char params_str[256];
  snprintf(params_str, 256,
           "small-size-ratio=%.4lf,ghost-size-ratio=%.4lf,move-to-main-"
           "threshold=%d,num-candidates=%d,recent-window=%d\n",
           params->small_size_ratio, params->ghost_size_ratio,
           params->move_to_main_threshold, params->num_candidates,
           params->recent_window);
  return params_str;
}

static void S3FIFOEmb_parse_params(cache_t *cache,
                                   const char *cache_specific_params) {
  auto *params = static_cast<S3FIFOEmb_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    /* different parameters are separated by comma,
     * key and value are separated by = */
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    // skip the white space
    while (params_str != NULL && *params_str == ' ') {
      params_str++;
    }

    if (strcasecmp(key, "fifo-size-ratio") == 0 ||
        strcasecmp(key, "small-size-ratio") == 0) {
      params->small_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "ghost-size-ratio") == 0) {
      params->ghost_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "move-to-main-threshold") == 0) {
      params->move_to_main_threshold = atoi(value);
    } else if (strcasecmp(key, "num-candidates") == 0 ||
               strcasecmp(key, "num_candidates") == 0) {
      params->num_candidates = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0 ||
               strcasecmp(key, "recent_window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "print") == 0) {
      printf("parameters: %s\n", S3FIFOEmb_current_params(params));
      exit(0);
    } else {
      ERROR("%s does not have parameter %s\n", cache->cache_name, key);
      exit(1);
    }
  }

  free(old_params_str);
}

#ifdef __cplusplus
}
#endif
