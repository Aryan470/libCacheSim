
#include <cstring>

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

  int recent_window;        // default: 16
  double forgive_threshold; // similarity above this gets forgiven

  // Stats
  int64_t n_forgive;        // number of times we forgave an object
  int64_t n_evict;          // number of evictions
} SieveEmbForgive_params_t;

// Default parameters
static const int DEFAULT_RECENT_WINDOW = 16;
static const double DEFAULT_FORGIVE_THRESHOLD = 0.325;

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************
static void SieveEmbForgive_free(cache_t *cache);
static bool SieveEmbForgive_get(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmbForgive_find(cache_t *cache, const request_t *req,
                                         const bool update_cache);
static cache_obj_t *SieveEmbForgive_insert(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmbForgive_to_evict(cache_t *cache, const request_t *req);
static void SieveEmbForgive_evict(cache_t *cache, const request_t *req);
static bool SieveEmbForgive_remove(cache_t *cache, const obj_id_t obj_id);
static void SieveEmbForgive_parse_params(cache_t *cache,
                                         const char *cache_specific_params);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ***********************************************************************

static void SieveEmbForgive_parse_params(cache_t *cache,
                                         const char *cache_specific_params) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  char *params_str = NULL;
  char *params_str_orig = NULL;
  if (cache_specific_params != NULL) {
    params_str = strdup(cache_specific_params);
    params_str_orig = params_str;
  }

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep(&params_str, "=");
    char *value = strsep(&params_str, ",");

    while (key != NULL && *key == ' ') key++;
    while (value != NULL && *value == ' ') value++;

    if (key == NULL) break;

    if (strcasecmp(key, "print") == 0) {
      printf("SieveEmbForgive parameters: recent-window=%d, forgive-threshold=%.2f\n",
             params->recent_window, params->forgive_threshold);
      continue;
    }

    if (value == NULL) break;

    if (strcasecmp(key, "recent-window") == 0 ||
        strcasecmp(key, "recent_window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "forgive-threshold") == 0 ||
               strcasecmp(key, "forgive_threshold") == 0) {
      params->forgive_threshold = atof(value);
    } else {
      ERROR("SieveEmbForgive does not have parameter %s\n", key);
      abort();
    }
  }

  if (params_str_orig != NULL) {
    free(params_str_orig);
  }
}

cache_t *SieveEmbForgive_init(const common_cache_params_t ccache_params,
                              const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("SieveEmbForgive", ccache_params, cache_specific_params);
  cache->cache_init = SieveEmbForgive_init;
  cache->cache_free = SieveEmbForgive_free;
  cache->get = SieveEmbForgive_get;
  cache->find = SieveEmbForgive_find;
  cache->insert = SieveEmbForgive_insert;
  cache->evict = SieveEmbForgive_evict;
  cache->remove = SieveEmbForgive_remove;
  cache->to_evict = SieveEmbForgive_to_evict;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 1;
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = my_malloc(SieveEmbForgive_params_t);
  memset(cache->eviction_params, 0, sizeof(SieveEmbForgive_params_t));
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  params->pointer = NULL;
  params->q_head = NULL;
  params->q_tail = NULL;
  params->recent_window = DEFAULT_RECENT_WINDOW;
  params->forgive_threshold = DEFAULT_FORGIVE_THRESHOLD;
  params->n_forgive = 0;
  params->n_evict = 0;

  if (cache_specific_params != NULL) {
    SieveEmbForgive_parse_params(cache, cache_specific_params);
  }

  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  params->embedding_manager = static_cast<void *>(emb);

  return cache;
}

static void SieveEmbForgive_free(cache_t *cache) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Print stats
  double forgive_rate = params->n_evict > 0
      ? 100.0 * params->n_forgive / (params->n_forgive + params->n_evict)
      : 0.0;
  printf("SieveEmbForgive stats: evictions=%ld, forgives=%ld, forgive_rate=%.2f%%\n",
         (long)params->n_evict, (long)params->n_forgive, forgive_rate);

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

static bool SieveEmbForgive_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Update embedding on EVERY access (hit or miss)
  emb->on_access(req->obj_id);

  return cache_get_base(cache, req);
}

// ***********************************************************************
// ****                                                               ****
// ****       developer facing APIs (used by cache developer)         ****
// ****                                                               ****
// ***********************************************************************

static cache_obj_t *SieveEmbForgive_find(cache_t *cache, const request_t *req,
                                         const bool update_cache) {
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);
  if (cache_obj != NULL && update_cache) {
    cache_obj->sieve.freq = 1;
  }

  return cache_obj;
}

static cache_obj_t *SieveEmbForgive_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);
  obj->sieve.freq = 0;

  return obj;
}

static cache_obj_t *SieveEmbForgive_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  cache_obj_t *obj = params->pointer == NULL ? params->q_tail : params->pointer;

  // Standard SIEVE scan with embedding-based forgive
  int64_t max_scan = cache->n_obj + 1;
  for (int64_t i = 0; i < max_scan && obj != NULL; i++) {
    if (obj->sieve.freq == 0) {
      // Check embedding similarity - if high, forgive this object
      double sim = emb->max_similarity_to_recent(obj->obj_id);
      if (sim > params->forgive_threshold) {
        // Forgive: set freq=1 and continue
        obj->sieve.freq = 1;
        params->n_forgive++;
      } else {
        // Low similarity: evict this object
        return obj;
      }
    } else {
      // Clear visited bit (standard SIEVE)
      obj->sieve.freq = 0;
    }

    obj = obj->queue.prev == NULL ? params->q_tail : obj->queue.prev;
  }

  // Fallback: return whatever we're pointing at
  return obj != NULL ? obj : params->q_tail;
}

static void SieveEmbForgive_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

  cache_obj_t *obj = SieveEmbForgive_to_evict(cache, req);

  params->n_evict++;
  params->pointer = obj->queue.prev;
  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_evict_base(cache, obj, true);
}

static void SieveEmbForgive_remove_obj(cache_t *cache, cache_obj_t *obj_to_remove) {
  DEBUG_ASSERT(obj_to_remove != NULL);
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  if (obj_to_remove == params->pointer) {
    params->pointer = obj_to_remove->queue.prev;
  }
  remove_obj_from_list(&params->q_head, &params->q_tail, obj_to_remove);
  cache_remove_obj_base(cache, obj_to_remove, true);
}

static bool SieveEmbForgive_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }

  SieveEmbForgive_remove_obj(cache, obj);
  return true;
}

#ifdef __cplusplus
}
#endif
