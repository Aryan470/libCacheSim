//
//  S3-FIFO with Forgiveness: Instead of using embeddings to pick victims,
//  we "forgive" objects that would be evicted if they:
//    1. Have been accessed at least N times (have meaningful embedding)
//    2. Are highly similar to recent accesses
//
//  This is a conservative intervention - only override base algorithm
//  when embedding signal is strong.
//

#include <algorithm>

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

  // Embedding/Forgiveness fields
  void *embedding_manager;
  int min_access_count;      // Min accesses to be eligible for forgiveness (default: 5)
  double forgive_threshold;  // Similarity threshold to forgive (default: 0.7)
  int max_forgives;          // Max forgives per eviction call (default: 3)
  int recent_window;         // Recent window size for embeddings (default: 16)
} S3FIFOForgive_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2,"
    "min-access-count=5,forgive-threshold=0.7,max-forgives=3,recent-window=16";

// Function declarations
static void S3FIFOForgive_free(cache_t *cache);
static bool S3FIFOForgive_get(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFOForgive_find(cache_t *cache, const request_t *req,
                                        const bool update_cache);
static cache_obj_t *S3FIFOForgive_insert(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFOForgive_to_evict(cache_t *cache, const request_t *req);
static void S3FIFOForgive_evict(cache_t *cache, const request_t *req);
static bool S3FIFOForgive_remove(cache_t *cache, const obj_id_t obj_id);
static inline int64_t S3FIFOForgive_get_occupied_byte(const cache_t *cache);
static inline int64_t S3FIFOForgive_get_n_obj(const cache_t *cache);
static inline bool S3FIFOForgive_can_insert(cache_t *cache, const request_t *req);
static void S3FIFOForgive_parse_params(cache_t *cache,
                                        const char *cache_specific_params);

static void S3FIFOForgive_evict_small(cache_t *cache, const request_t *req);
static void S3FIFOForgive_evict_main(cache_t *cache, const request_t *req);

// ***********************************************************************
// ****                   Initialization                              ****
// ***********************************************************************

cache_t *S3FIFOForgive_init(const common_cache_params_t ccache_params,
                             const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("S3FIFOForgive", ccache_params, cache_specific_params);
  cache->cache_init = S3FIFOForgive_init;
  cache->cache_free = S3FIFOForgive_free;
  cache->get = S3FIFOForgive_get;
  cache->find = S3FIFOForgive_find;
  cache->insert = S3FIFOForgive_insert;
  cache->evict = S3FIFOForgive_evict;
  cache->remove = S3FIFOForgive_remove;
  cache->to_evict = S3FIFOForgive_to_evict;
  cache->get_n_obj = S3FIFOForgive_get_n_obj;
  cache->get_occupied_byte = S3FIFOForgive_get_occupied_byte;
  cache->can_insert = S3FIFOForgive_can_insert;

  cache->obj_md_size = 0;

  cache->eviction_params = malloc(sizeof(S3FIFOForgive_params_t));
  memset(cache->eviction_params, 0, sizeof(S3FIFOForgive_params_t));
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  params->req_local = new_request();
  params->hit_on_ghost = false;

  // Set defaults
  params->min_access_count = 5;
  params->forgive_threshold = 0.7;
  params->max_forgives = 3;
  params->recent_window = 16;

  S3FIFOForgive_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    S3FIFOForgive_parse_params(cache, cache_specific_params);
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
    snprintf(params->ghost_fifo->cache_name, CACHE_NAME_ARRAY_LEN, "FIFO-ghost");
  } else {
    params->ghost_fifo = NULL;
  }

  ccache_params_local.cache_size = main_fifo_size;
  params->main_fifo = FIFO_init(ccache_params_local, NULL);

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  params->embedding_manager = static_cast<void *>(emb);

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
           "S3FIFOForgive-%.2lf-%d-%.2lf-%d",
           params->small_size_ratio, params->min_access_count,
           params->forgive_threshold, params->max_forgives);

  return cache;
}

static void S3FIFOForgive_free(cache_t *cache) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
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

// ***********************************************************************
// ****                   Core Operations                             ****
// ***********************************************************************

static bool S3FIFOForgive_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Track access for embeddings on EVERY request
  emb->on_access(req->obj_id);

  return cache_get_base(cache, req);
}

static cache_obj_t *S3FIFOForgive_find(cache_t *cache, const request_t *req,
                                        const bool update_cache) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);

  if (!update_cache) {
    cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, false);
    if (obj != NULL) return obj;
    obj = params->main_fifo->find(params->main_fifo, req, false);
    return obj;
  }

  params->hit_on_ghost = false;
  cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
    return obj;
  }

  if (params->ghost_fifo != NULL &&
      params->ghost_fifo->remove(params->ghost_fifo, req->obj_id)) {
    params->hit_on_ghost = true;
  }

  obj = params->main_fifo->find(params->main_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
  }

  return obj;
}

static cache_obj_t *S3FIFOForgive_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  cache_obj_t *obj = NULL;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (params->hit_on_ghost) {
    params->hit_on_ghost = false;
    obj = main_fifo->insert(main_fifo, req);
  } else {
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

static cache_obj_t *S3FIFOForgive_to_evict(cache_t *cache, const request_t *req) {
  assert(false);
  return NULL;
}

// ***********************************************************************
// ****                   Eviction Logic                              ****
// ***********************************************************************

// Small FIFO eviction - unchanged from base S3FIFO
static void S3FIFOForgive_evict_small(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  cache_t *small_fifo = params->small_fifo;
  cache_t *ghost_fifo = params->ghost_fifo;
  cache_t *main_fifo = params->main_fifo;

  bool has_evicted = false;
  while (!has_evicted && small_fifo->get_occupied_byte(small_fifo) > 0) {
    cache_obj_t *obj_to_evict = small_fifo->to_evict(small_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    if (obj_to_evict->S3FIFO.freq >= params->move_to_main_threshold) {
      main_fifo->insert(main_fifo, params->req_local);
    } else {
      if (ghost_fifo != NULL) {
        ghost_fifo->get(ghost_fifo, params->req_local);
      }
      has_evicted = true;
    }

    bool removed = small_fifo->remove(small_fifo, params->req_local->obj_id);
    DEBUG_ASSERT(removed);
  }
}

// Debug counters
static uint64_t g_main_evict_calls = 0;
static uint64_t g_freq0_candidates = 0;
static uint64_t g_access_count_checked = 0;
static uint64_t g_access_count_passed = 0;
static uint64_t g_similarity_checked = 0;
static uint64_t g_forgives = 0;

// Main FIFO eviction with forgiveness
static void S3FIFOForgive_evict_main(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);
  cache_t *main_fifo = params->main_fifo;

  g_main_evict_calls++;

  int forgives_remaining = params->max_forgives;
  bool has_evicted = false;

  while (!has_evicted && main_fifo->get_occupied_byte(main_fifo) > 0) {
    cache_obj_t *obj_to_evict = main_fifo->to_evict(main_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);

    int freq = obj_to_evict->S3FIFO.freq;
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    if (freq >= 1) {
      // Standard second-chance: reinsert with decremented freq
      main_fifo->remove(main_fifo, obj_to_evict->obj_id);
      cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
      new_obj->S3FIFO.freq = MIN(freq, 3) - 1;
    } else {
      // freq == 0: Candidate for eviction
      g_freq0_candidates++;

      // Check if we should forgive based on embedding similarity
      bool should_forgive = false;

      if (forgives_remaining > 0) {
        g_access_count_checked++;
        int access_count = emb->get_access_count(obj_to_evict->obj_id);

        if (access_count >= params->min_access_count) {
          g_access_count_passed++;
          g_similarity_checked++;
          double similarity = emb->max_similarity_to_recent(obj_to_evict->obj_id);

          // Track similarity distribution
          static uint64_t sim_buckets[10] = {0}; // 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
          int bucket = (int)(similarity * 10);
          if (bucket >= 10) bucket = 9;
          if (bucket < 0) bucket = 0;
          sim_buckets[bucket]++;

          if (g_similarity_checked % 100000 == 0) {
            fprintf(stderr, "[SIM DIST] ");
            for (int i = 0; i < 10; i++) {
              fprintf(stderr, "%.1f-%.1f:%lu ", i*0.1, (i+1)*0.1, sim_buckets[i]);
            }
            fprintf(stderr, "\n");
          }

          if (similarity >= params->forgive_threshold) {
            should_forgive = true;
            forgives_remaining--;
            g_forgives++;
          }
        }
      }

      if (should_forgive) {
        // Forgive: reinsert at head with freq=1 (one more chance)
        main_fifo->remove(main_fifo, obj_to_evict->obj_id);
        cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
        new_obj->S3FIFO.freq = 1;
      } else {
        // Evict
        main_fifo->remove(main_fifo, obj_to_evict->obj_id);
        has_evicted = true;
      }

      // Print debug stats periodically
      if (g_freq0_candidates % 1000 == 0 || g_freq0_candidates == 1) {
        fprintf(stderr, "[Forgive] main_evict=%lu freq0=%lu acc_checked=%lu acc_passed=%lu sim_checked=%lu forgives=%lu\n",
                g_main_evict_calls, g_freq0_candidates, g_access_count_checked,
                g_access_count_passed, g_similarity_checked, g_forgives);
      }
    }
  }
}

static void S3FIFOForgive_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  params->has_evicted = true;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (main_fifo->get_occupied_byte(main_fifo) > main_fifo->cache_size ||
      small_fifo->get_occupied_byte(small_fifo) == 0) {
    S3FIFOForgive_evict_main(cache, req);
  } else {
    S3FIFOForgive_evict_small(cache, req);
  }
}

static bool S3FIFOForgive_remove(cache_t *cache, const obj_id_t obj_id) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  bool removed = false;
  removed = removed || params->small_fifo->remove(params->small_fifo, obj_id);
  removed = removed || (params->ghost_fifo &&
                        params->ghost_fifo->remove(params->ghost_fifo, obj_id));
  removed = removed || params->main_fifo->remove(params->main_fifo, obj_id);
  return removed;
}

static inline int64_t S3FIFOForgive_get_occupied_byte(const cache_t *cache) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  return params->small_fifo->get_occupied_byte(params->small_fifo) +
         params->main_fifo->get_occupied_byte(params->main_fifo);
}

static inline int64_t S3FIFOForgive_get_n_obj(const cache_t *cache) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  return params->small_fifo->get_n_obj(params->small_fifo) +
         params->main_fifo->get_n_obj(params->main_fifo);
}

static inline bool S3FIFOForgive_can_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);
  return req->obj_size <= params->small_fifo->cache_size &&
         cache_can_insert_default(cache, req);
}

// ***********************************************************************
// ****                   Parameter Parsing                           ****
// ***********************************************************************

static void S3FIFOForgive_parse_params(cache_t *cache,
                                        const char *cache_specific_params) {
  auto *params = static_cast<S3FIFOForgive_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') {
      params_str++;
    }

    if (strcasecmp(key, "small-size-ratio") == 0 ||
        strcasecmp(key, "fifo-size-ratio") == 0) {
      params->small_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "ghost-size-ratio") == 0) {
      params->ghost_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "move-to-main-threshold") == 0) {
      params->move_to_main_threshold = atoi(value);
    } else if (strcasecmp(key, "min-access-count") == 0) {
      params->min_access_count = atoi(value);
    } else if (strcasecmp(key, "forgive-threshold") == 0) {
      params->forgive_threshold = strtod(value, NULL);
    } else if (strcasecmp(key, "max-forgives") == 0) {
      params->max_forgives = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "print") == 0) {
      printf("S3FIFOForgive: small-size-ratio=%.2lf, min-access-count=%d, "
             "forgive-threshold=%.2lf, max-forgives=%d\n",
             params->small_size_ratio, params->min_access_count,
             params->forgive_threshold, params->max_forgives);
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
