//
//  SieveForgive: A modified SIEVE that uses embedding-based forgiveness
//  to protect objects with high similarity to recent accesses from eviction.
//
//  How it works:
//    1. Standard SIEVE: scan from hand, evict first object with freq=0
//    2. SieveForgive: before evicting a freq=0 object, check if it should be "forgiven"
//       - If access_count >= min_access_count AND similarity >= forgive_threshold
//       - Forgiven: set freq=1, continue scanning
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
static std::unordered_set<uint64_t> sieve_forgiven_objects;
static std::unordered_map<uint64_t, int> sieve_hits_after_forgive;
static int64_t sieve_total_hits_on_forgiven = 0;
static int64_t sieve_forgiven_then_evicted = 0;
static int64_t sieve_forgiven_with_zero_hits = 0;

// Similarity distribution tracking (10 buckets: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
static int64_t sieve_sim_bucket_forgiven[10] = {0};
static int64_t sieve_sim_bucket_evicted[10] = {0};

// Regret tracking: did we evict something that was accessed soon after?
static constexpr int SIEVE_REGRET_BUFFER_SIZE = 10000;
struct SieveEvictionRecord {
  uint64_t evicted_obj_id;
  uint64_t triggering_access_obj_id;
};
static SieveEvictionRecord sieve_regret_buffer[SIEVE_REGRET_BUFFER_SIZE];
static int sieve_regret_buffer_idx = 0;
static int sieve_regret_buffer_count = 0;
static std::unordered_map<uint64_t, uint64_t> sieve_recent_evictions;
static int64_t sieve_n_total_misses = 0;
static int64_t sieve_n_regret_misses = 0;
static int64_t sieve_n_sequential_regret = 0;

typedef struct {
  cache_obj_t *q_head;           // MRU end
  cache_obj_t *q_tail;           // LRU end
  cache_obj_t *pointer;          // SIEVE hand
  void *embedding_manager;       // EmbeddingManager*
  int min_access_count;          // default: 3
  double forgive_threshold;      // default: 0.325
  int max_forgives;              // default: -1 (unlimited)
  int recent_window;             // default: 16
  double random_forgive_prob;    // if > 0, use random instead of embedding
  double lr;                     // embedding learning rate
  double ctx_speed;              // context perturbation speed
  bool use_max_sim;              // if true, use max similarity; else avg_top_k
  int64_t n_forgive;             // stats
  int64_t n_evict;               // stats
  int64_t n_candidates;          // stats: total eviction candidates considered
  int64_t n_qualified;           // stats: candidates with access_count >= min
  int64_t n_hit_after_forgive;   // stats: hits on forgiven objects
} SieveEmbForgive_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "min-access-count=3,forgive-threshold=0.325,max-forgives=-1,recent-window=16";

// Function declarations
static void SieveEmbForgive_free(cache_t *cache);
static bool SieveEmbForgive_get(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmbForgive_find(cache_t *cache, const request_t *req,
                                          const bool update_cache);
static cache_obj_t *SieveEmbForgive_insert(cache_t *cache, const request_t *req);
static cache_obj_t *SieveEmbForgive_to_evict(cache_t *cache, const request_t *req);
static void SieveEmbForgive_evict(cache_t *cache, const request_t *req);
static bool SieveEmbForgive_remove(cache_t *cache, const obj_id_t obj_id);
static void SieveEmbForgive_print_cache(const cache_t *cache);
static void SieveEmbForgive_parse_params(cache_t *cache,
                                          const char *cache_specific_params);

// ***********************************************************************
// ****                   Initialization                              ****
// ***********************************************************************

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
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;
  cache->print_cache = SieveEmbForgive_print_cache;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2 + 1;  // prev/next pointers + sieve.freq
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(SieveEmbForgive_params_t));
  memset(cache->eviction_params, 0, sizeof(SieveEmbForgive_params_t));
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;
  params->pointer = NULL;

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
  params->use_max_sim = false;  // default: use avg_top_k

  SieveEmbForgive_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    SieveEmbForgive_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  emb->set_lr(params->lr);
  emb->set_ctx_speed(params->ctx_speed);
  params->embedding_manager = static_cast<void *>(emb);

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
           "SieveEmbForgive-%d-%.2lf-%d",
           params->min_access_count,
           params->forgive_threshold, params->max_forgives);

  return cache;
}

static void SieveEmbForgive_free(cache_t *cache) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Print stats on exit
  fprintf(stderr, "[SieveEmbForgive] Final: evicts=%ld candidates=%ld qualified=%ld(%.2f%%) forgives=%ld(%.2f%% of qualified)\n",
          params->n_evict, params->n_candidates,
          params->n_qualified,
          params->n_candidates > 0 ? 100.0 * params->n_qualified / params->n_candidates : 0.0,
          params->n_forgive,
          params->n_qualified > 0 ? 100.0 * params->n_forgive / params->n_qualified : 0.0);

  // Print forgiveness effectiveness stats
  fprintf(stderr, "[SieveEmbForgive] Forgiven objects later evicted: %ld, with zero hits: %ld (%.2f%%)\n",
          sieve_forgiven_then_evicted, sieve_forgiven_with_zero_hits,
          sieve_forgiven_then_evicted > 0 ? 100.0 * sieve_forgiven_with_zero_hits / sieve_forgiven_then_evicted : 0.0);
  fprintf(stderr, "[SieveEmbForgive] Total hits on forgiven objects: %ld, avg hits per forgiven: %.2f\n",
          sieve_total_hits_on_forgiven,
          sieve_forgiven_then_evicted > 0 ? (double)sieve_total_hits_on_forgiven / sieve_forgiven_then_evicted : 0.0);

  // Print similarity distribution
  fprintf(stderr, "[SieveEmbForgive] Similarity distribution (forgiven/evicted):\n");
  for (int i = 0; i < 10; i++) {
    fprintf(stderr, "  [%.1f-%.1f]: %ld / %ld\n",
            i * 0.1, (i + 1) * 0.1, sieve_sim_bucket_forgiven[i], sieve_sim_bucket_evicted[i]);
  }

  // Print regret stats
  fprintf(stderr, "\n[Regret Analysis] Evicted objects accessed within next %d requests:\n", SIEVE_REGRET_BUFFER_SIZE);
  fprintf(stderr, "  Total misses: %ld\n", sieve_n_total_misses);
  fprintf(stderr, "  Regret misses (recently evicted): %ld (%.2f%% of misses)\n",
          sieve_n_regret_misses, sieve_n_total_misses > 0 ? 100.0 * sieve_n_regret_misses / sieve_n_total_misses : 0.0);
  fprintf(stderr, "  Sequential regret (sim > 0.2): %ld (%.2f%% of regret)\n",
          sieve_n_sequential_regret, sieve_n_regret_misses > 0 ? 100.0 * sieve_n_sequential_regret / sieve_n_regret_misses : 0.0);

  // Reset static vars for next run
  sieve_forgiven_objects.clear();
  sieve_hits_after_forgive.clear();
  sieve_total_hits_on_forgiven = 0;
  sieve_forgiven_then_evicted = 0;
  sieve_forgiven_with_zero_hits = 0;
  for (int i = 0; i < 10; i++) {
    sieve_sim_bucket_forgiven[i] = 0;
    sieve_sim_bucket_evicted[i] = 0;
  }
  // Reset regret tracking
  sieve_recent_evictions.clear();
  sieve_regret_buffer_idx = 0;
  sieve_regret_buffer_count = 0;
  sieve_n_total_misses = 0;
  sieve_n_regret_misses = 0;
  sieve_n_sequential_regret = 0;

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ***********************************************************************
// ****                   Core Operations                             ****
// ***********************************************************************

static bool SieveEmbForgive_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Track access for embeddings on EVERY request
  emb->on_access(req->obj_id);

  // Check if this is a miss on a recently evicted object (regret tracking)
  cache_obj_t *obj = cache_find_base(cache, req, false);
  if (obj == NULL) {
    // This is a miss
    sieve_n_total_misses++;
    auto it = sieve_recent_evictions.find(req->obj_id);
    if (it != sieve_recent_evictions.end()) {
      // We recently evicted this object - regret!
      sieve_n_regret_misses++;
      // Check if the evicted object was sequential to the triggering access
      double sim = emb->similarity_public(req->obj_id, it->second);
      if (sim > 0.2) {
        sieve_n_sequential_regret++;
      }
    }
  }

  return cache_get_base(cache, req);
}

static cache_obj_t *SieveEmbForgive_find(cache_t *cache, const request_t *req,
                                          const bool update_cache) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    // Track hits on forgiven objects
    if (sieve_forgiven_objects.count(req->obj_id)) {
      sieve_hits_after_forgive[req->obj_id]++;
      sieve_total_hits_on_forgiven++;
    }
    // SIEVE: set visited bit on access
    cache_obj->sieve.freq = 1;
  }
  return cache_obj;
}

static cache_obj_t *SieveEmbForgive_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);
  obj->sieve.freq = 0;  // New objects start unvisited

  return obj;
}

static cache_obj_t *SieveEmbForgive_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

  // Find eviction candidate using SIEVE hand
  cache_obj_t *obj = params->pointer == NULL ? params->q_tail : params->pointer;

  while (obj != NULL && obj->sieve.freq != 0) {
    obj->sieve.freq = 0;  // Clear visited bit
    obj = obj->queue.prev == NULL ? params->q_tail : obj->queue.prev;
  }

  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return obj != NULL ? obj : params->q_tail;
}

// ***********************************************************************
// ****                   Eviction Logic                              ****
// ***********************************************************************

static void SieveEmbForgive_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  int forgives_remaining = params->max_forgives;
  bool evicted = false;

  // Start from hand position or tail
  cache_obj_t *obj = params->pointer == NULL ? params->q_tail : params->pointer;

  while (!evicted && obj != NULL) {
    // If visited, clear and move on (standard SIEVE)
    if (obj->sieve.freq != 0) {
      obj->sieve.freq = 0;
      obj = obj->queue.prev == NULL ? params->q_tail : obj->queue.prev;
      continue;
    }

    // Found an unvisited object - eviction candidate
    params->n_candidates++;

    bool should_forgive = false;

    // Check if we can forgive
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
          // Embedding-based forgiveness
          if (params->use_max_sim) {
            similarity = emb->max_similarity_to_recent(obj->obj_id);
          } else {
            similarity = emb->avg_top_k_similarity_to_recent(obj->obj_id, 3);
          }
          passes_check = (similarity >= params->forgive_threshold);

          // Track similarity distribution
          int bucket = std::min(9, (int)(similarity * 10));
          if (passes_check) {
            sieve_sim_bucket_forgiven[bucket]++;
          } else {
            sieve_sim_bucket_evicted[bucket]++;
          }
        }

        if (passes_check) {
          should_forgive = true;
          if (forgives_remaining > 0) forgives_remaining--;
          params->n_forgive++;

          // Track this object as forgiven
          sieve_forgiven_objects.insert(obj->obj_id);
          sieve_hits_after_forgive[obj->obj_id] = 0;
        }
      }
    }

    if (should_forgive) {
      // Forgive: skip this object (don't change freq), continue scanning
      // This gives exactly one chance - survives only this eviction cycle
      obj = obj->queue.prev == NULL ? params->q_tail : obj->queue.prev;
    } else {
      // Evict this object
      params->n_evict++;

      // Track if this was a previously forgiven object
      if (sieve_forgiven_objects.count(obj->obj_id)) {
        sieve_forgiven_then_evicted++;
        int hits = sieve_hits_after_forgive[obj->obj_id];
        if (hits == 0) {
          sieve_forgiven_with_zero_hits++;
        }
        sieve_forgiven_objects.erase(obj->obj_id);
        sieve_hits_after_forgive.erase(obj->obj_id);
      }

      // Record eviction for regret tracking
      if (sieve_regret_buffer_count == SIEVE_REGRET_BUFFER_SIZE) {
        sieve_recent_evictions.erase(sieve_regret_buffer[sieve_regret_buffer_idx].evicted_obj_id);
      }
      sieve_regret_buffer[sieve_regret_buffer_idx] = {obj->obj_id, req->obj_id};
      sieve_recent_evictions[obj->obj_id] = req->obj_id;
      sieve_regret_buffer_idx = (sieve_regret_buffer_idx + 1) % SIEVE_REGRET_BUFFER_SIZE;
      if (sieve_regret_buffer_count < SIEVE_REGRET_BUFFER_SIZE) sieve_regret_buffer_count++;

      // Update hand position
      params->pointer = obj->queue.prev;

      // Remove from list and evict
      remove_obj_from_list(&params->q_head, &params->q_tail, obj);
      cache_evict_base(cache, obj, true);
      evicted = true;
    }
  }

  // Print debug stats periodically
  if (params->n_evict % 100000 == 0 && params->n_evict > 0) {
    fprintf(stderr, "[SieveEmbForgive] evicts=%ld candidates=%ld qualified=%ld(%.2f%%) forgives=%ld(%.2f%%)\n",
            params->n_evict, params->n_candidates,
            params->n_qualified,
            params->n_candidates > 0 ? 100.0 * params->n_qualified / params->n_candidates : 0.0,
            params->n_forgive,
            params->n_qualified > 0 ? 100.0 * params->n_forgive / params->n_qualified : 0.0);
  }
}

static bool SieveEmbForgive_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

  // Update hand if we're removing the object it points to
  if (obj == params->pointer) {
    params->pointer = obj->queue.prev;
  }

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

static void SieveEmbForgive_print_cache(const cache_t *cache) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);
  cache_obj_t *cur = params->q_head;
  if (cur == NULL) {
    printf("empty\n");
    return;
  }
  while (cur != NULL) {
    printf("%lu(%d)->", (unsigned long)cur->obj_id, cur->sieve.freq);
    cur = cur->queue.next;
  }
  printf("END\n");
}

// ***********************************************************************
// ****                   Parameter Parsing                           ****
// ***********************************************************************

static void SieveEmbForgive_parse_params(cache_t *cache,
                                          const char *cache_specific_params) {
  auto *params = static_cast<SieveEmbForgive_params_t *>(cache->eviction_params);

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
    } else if (strcasecmp(key, "use-max-sim") == 0) {
      params->use_max_sim = (atoi(value) != 0);
    } else if (strcasecmp(key, "print") == 0) {
      printf("SieveEmbForgive: min-access-count=%d, "
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
