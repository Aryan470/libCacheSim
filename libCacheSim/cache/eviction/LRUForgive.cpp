//
//  LRU with Forgiveness: A modified LRU that uses embedding-based forgiveness
//  to protect objects with high similarity to recent accesses from eviction.
//
//  How it works:
//    1. Standard LRU: evict tail (least recently used) immediately
//    2. LRUForgive: before evicting tail, check if it should be "forgiven"
//       - If access_count >= min_access_count AND max_similarity >= forgive_threshold
//       - Forgiven: move object to head (MRU position), try next candidate
//       - Not forgiven: evict normally
//

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// Debug tracking for forgiven objects
static std::unordered_set<uint64_t> forgiven_objects;
static std::unordered_map<uint64_t, int> hits_after_forgive;  // obj_id -> hit count since forgiven
static int64_t total_hits_on_forgiven = 0;
static int64_t forgiven_then_evicted = 0;
static int64_t forgiven_with_zero_hits = 0;

// Similarity distribution tracking (10 buckets: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
static int64_t sim_bucket_forgiven[10] = {0};
static int64_t sim_bucket_evicted[10] = {0};

typedef struct {
  cache_obj_t *q_head;           // MRU end
  cache_obj_t *q_tail;           // LRU end (eviction candidate)
  void *embedding_manager;       // EmbeddingManager*
  int min_access_count;          // default: 3
  double forgive_threshold;      // default: 0.325
  int max_forgives;              // default: -1 (unlimited)
  int recent_window;             // default: 16
  double random_forgive_prob;    // if > 0, use random instead of embedding
  double lr;                     // embedding learning rate
  double ctx_speed;              // context perturbation speed
  int64_t n_forgive;             // stats
  int64_t n_evict;               // stats
  int64_t n_candidates;          // stats: total eviction candidates considered
  int64_t n_qualified;           // stats: candidates with access_count >= min
  int64_t n_hit_after_forgive;   // stats: hits on forgiven objects
} LRUForgive_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "min-access-count=3,forgive-threshold=0.325,max-forgives=-1,recent-window=16";

// Function declarations
static void LRUForgive_free(cache_t *cache);
static bool LRUForgive_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgive_find(cache_t *cache, const request_t *req,
                                     const bool update_cache);
static cache_obj_t *LRUForgive_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgive_to_evict(cache_t *cache, const request_t *req);
static void LRUForgive_evict(cache_t *cache, const request_t *req);
static bool LRUForgive_remove(cache_t *cache, const obj_id_t obj_id);
static void LRUForgive_print_cache(const cache_t *cache);
static void LRUForgive_parse_params(cache_t *cache,
                                     const char *cache_specific_params);

// ***********************************************************************
// ****                   Initialization                              ****
// ***********************************************************************

cache_t *LRUForgive_init(const common_cache_params_t ccache_params,
                          const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("LRUForgive", ccache_params, cache_specific_params);
  cache->cache_init = LRUForgive_init;
  cache->cache_free = LRUForgive_free;
  cache->get = LRUForgive_get;
  cache->find = LRUForgive_find;
  cache->insert = LRUForgive_insert;
  cache->evict = LRUForgive_evict;
  cache->remove = LRUForgive_remove;
  cache->to_evict = LRUForgive_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;
  cache->print_cache = LRUForgive_print_cache;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2;  // prev/next pointers
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(LRUForgive_params_t));
  memset(cache->eviction_params, 0, sizeof(LRUForgive_params_t));
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;

  // Set defaults
  params->min_access_count = 3;
  params->forgive_threshold = 0.325;
  params->max_forgives = -1;  // -1 means unlimited
  params->recent_window = 16;
  params->n_forgive = 0;
  params->n_evict = 0;
  params->n_candidates = 0;
  params->n_qualified = 0;
  params->n_hit_after_forgive = 0;
  params->random_forgive_prob = 0.0;  // 0 means use embeddings
  params->lr = 0.2;
  params->ctx_speed = 0.1;

  LRUForgive_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    LRUForgive_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  emb->set_lr(params->lr);
  emb->set_ctx_speed(params->ctx_speed);
  params->embedding_manager = static_cast<void *>(emb);

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
           "LRUForgive-%d-%.2lf-%d",
           params->min_access_count,
           params->forgive_threshold, params->max_forgives);

  return cache;
}

static void LRUForgive_free(cache_t *cache) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Print stats on exit
  fprintf(stderr, "[LRUForgive] Final: evicts=%ld candidates=%ld qualified=%ld(%.2f%%) forgives=%ld(%.2f%% of qualified)\n",
          params->n_evict, params->n_candidates,
          params->n_qualified,
          100.0 * params->n_qualified / params->n_candidates,
          params->n_forgive,
          params->n_qualified > 0 ? 100.0 * params->n_forgive / params->n_qualified : 0.0);

  // Print forgiveness effectiveness stats
  fprintf(stderr, "[LRUForgive] Forgiven objects later evicted: %ld, with zero hits: %ld (%.2f%%)\n",
          forgiven_then_evicted, forgiven_with_zero_hits,
          forgiven_then_evicted > 0 ? 100.0 * forgiven_with_zero_hits / forgiven_then_evicted : 0.0);
  fprintf(stderr, "[LRUForgive] Total hits on forgiven objects: %ld, avg hits per forgiven: %.2f\n",
          total_hits_on_forgiven,
          forgiven_then_evicted > 0 ? (double)total_hits_on_forgiven / forgiven_then_evicted : 0.0);

  // Print similarity distribution
  fprintf(stderr, "[LRUForgive] Similarity distribution (forgiven/evicted):\n");
  for (int i = 0; i < 10; i++) {
    fprintf(stderr, "  [%.1f-%.1f]: %ld / %ld\n",
            i * 0.1, (i + 1) * 0.1, sim_bucket_forgiven[i], sim_bucket_evicted[i]);
  }

  // Reset static vars for next run
  forgiven_objects.clear();
  hits_after_forgive.clear();
  total_hits_on_forgiven = 0;
  forgiven_then_evicted = 0;
  forgiven_with_zero_hits = 0;
  for (int i = 0; i < 10; i++) {
    sim_bucket_forgiven[i] = 0;
    sim_bucket_evicted[i] = 0;
  }

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ***********************************************************************
// ****                   Core Operations                             ****
// ***********************************************************************

static bool LRUForgive_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Track access for embeddings on EVERY request
  emb->on_access(req->obj_id);

  return cache_get_base(cache, req);
}

static cache_obj_t *LRUForgive_find(cache_t *cache, const request_t *req,
                                     const bool update_cache) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    // Track hits on forgiven objects
    if (forgiven_objects.count(req->obj_id)) {
      hits_after_forgive[req->obj_id]++;
      total_hits_on_forgiven++;
    }
    // LRU: move to head on access
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
  }
  return cache_obj;
}

static cache_obj_t *LRUForgive_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);

  return obj;
}

static cache_obj_t *LRUForgive_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);

  DEBUG_ASSERT(params->q_tail != NULL || cache->occupied_byte == 0);

  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return params->q_tail;
}

// ***********************************************************************
// ****                   Eviction Logic                              ****
// ***********************************************************************

static void LRUForgive_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  DEBUG_ASSERT(params->q_tail != NULL);

  int forgives_remaining = params->max_forgives;
  bool evicted = false;

  while (!evicted && params->q_tail != NULL) {
    cache_obj_t *obj = params->q_tail;
    params->n_candidates++;

    // Check if we should forgive based on embedding similarity
    bool should_forgive = false;

    // forgives_remaining < 0 means unlimited forgives
    if (forgives_remaining != 0) {
      int access_count = emb->get_access_count(obj->obj_id);

      if (access_count >= params->min_access_count) {
        params->n_qualified++;

        bool passes_check = false;
        double similarity = 0.0;

        if (params->random_forgive_prob > 0) {
          // Random forgiveness mode
          double r = (double)rand() / RAND_MAX;
          passes_check = (r < params->random_forgive_prob);
        } else {
          // Embedding-based forgiveness (avg of top 3 similarities)
          similarity = emb->avg_top_k_similarity_to_recent(obj->obj_id, 3);
          passes_check = (similarity >= params->forgive_threshold);

          // Track similarity distribution
          int bucket = std::min(9, (int)(similarity * 10));
          if (passes_check) {
            sim_bucket_forgiven[bucket]++;
          } else {
            sim_bucket_evicted[bucket]++;
          }
        }

        if (passes_check) {
          should_forgive = true;
          if (forgives_remaining > 0) forgives_remaining--;  // Don't decrement if unlimited (-1)
          params->n_forgive++;

          // Track this object as forgiven
          forgiven_objects.insert(obj->obj_id);
          hits_after_forgive[obj->obj_id] = 0;
        }
      }
    }

    if (should_forgive) {
      // Forgive: move object to head (MRU position)
      move_obj_to_head(&params->q_head, &params->q_tail, obj);
    } else {
      // Evict: remove from tail
      params->n_evict++;

      // Track if this was a previously forgiven object
      if (forgiven_objects.count(obj->obj_id)) {
        forgiven_then_evicted++;
        int hits = hits_after_forgive[obj->obj_id];
        if (hits == 0) {
          forgiven_with_zero_hits++;
        }
        forgiven_objects.erase(obj->obj_id);
        hits_after_forgive.erase(obj->obj_id);
      }

      params->q_tail = params->q_tail->queue.prev;
      if (likely(params->q_tail != NULL)) {
        params->q_tail->queue.next = NULL;
      } else {
        // Cache is now empty
        DEBUG_ASSERT(cache->n_obj == 1);
        params->q_head = NULL;
      }

      cache_evict_base(cache, obj, true);
      evicted = true;
    }
  }

  // Print debug stats periodically
  if (params->n_evict % 100000 == 0 && params->n_evict > 0) {
    fprintf(stderr, "[LRUForgive] evicts=%ld candidates=%ld qualified=%ld(%.2f%%) forgives=%ld(%.2f%%)\n",
            params->n_evict, params->n_candidates,
            params->n_qualified,
            100.0 * params->n_qualified / params->n_candidates,
            params->n_forgive,
            params->n_qualified > 0 ? 100.0 * params->n_forgive / params->n_qualified : 0.0);
  }
}

static bool LRUForgive_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

static void LRUForgive_print_cache(const cache_t *cache) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);
  cache_obj_t *cur = params->q_head;
  // print from the most recent to the least recent
  if (cur == NULL) {
    printf("empty\n");
    return;
  }
  while (cur != NULL) {
    printf("%lu->", (unsigned long)cur->obj_id);
    cur = cur->queue.next;
  }
  printf("END\n");
}

// ***********************************************************************
// ****                   Parameter Parsing                           ****
// ***********************************************************************

static void LRUForgive_parse_params(cache_t *cache,
                                     const char *cache_specific_params) {
  auto *params = static_cast<LRUForgive_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') {
      params_str++;
    }

    if (strcasecmp(key, "min-access-count") == 0) {
      params->min_access_count = atoi(value);
    } else if (strcasecmp(key, "forgive-threshold") == 0) {
      params->forgive_threshold = strtod(value, NULL);
    } else if (strcasecmp(key, "max-forgives") == 0) {
      params->max_forgives = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "random-forgive-prob") == 0) {
      params->random_forgive_prob = strtod(value, NULL);
    } else if (strcasecmp(key, "lr") == 0) {
      params->lr = strtod(value, NULL);
    } else if (strcasecmp(key, "ctx-speed") == 0) {
      params->ctx_speed = strtod(value, NULL);
    } else if (strcasecmp(key, "print") == 0) {
      printf("LRUForgive: min-access-count=%d, "
             "forgive-threshold=%.2lf, max-forgives=%d, recent-window=%d\n",
             params->min_access_count, params->forgive_threshold,
             params->max_forgives, params->recent_window);
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
