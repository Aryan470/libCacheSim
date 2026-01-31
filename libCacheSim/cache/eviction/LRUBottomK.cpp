//
//  LRU with Bottom-K Selection: Modified LRU that examines the bottom K
//  candidates on eviction and evicts the one with worst embedding similarity
//  to recent accesses. The other K-1 candidates are promoted to MRU.
//

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <deque>
#include <cmath>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reuse distance tracking for evicted vs kept objects
struct ReuseTracker {
  std::unordered_map<uint64_t, int64_t> eviction_time;  // obj_id -> vtime when evicted
  std::unordered_map<uint64_t, int64_t> kept_time;      // obj_id -> vtime when kept (moved to MRU)
  std::vector<int64_t> evicted_reuse_distances;
  std::vector<int64_t> kept_reuse_distances;
  int64_t evicted_never_reused = 0;
  int64_t kept_never_reused = 0;
};

static ReuseTracker reuse_tracker;

typedef struct {
  cache_obj_t *q_head;           // MRU end
  cache_obj_t *q_tail;           // LRU end (eviction candidate)
  void *embedding_manager;       // EmbeddingManager*
  int num_candidates;            // K candidates to examine (default: 4)
  int recent_window;             // window for embedding similarity (default: 32)
  double lr;                     // embedding learning rate
  double ctx_speed;              // context perturbation speed
  int min_access_count;          // minimum accesses before using embeddings
  bool use_random;               // use random selection instead of embeddings
  int64_t n_evict;
  int64_t n_kept;                // objects moved to MRU instead of evicted
  int64_t n_no_embedding;        // evictions where embedding wasn't available
} LRUBottomK_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "num-candidates=4,recent-window=32,min-access-count=2";

// Function declarations
static void LRUBottomK_free(cache_t *cache);
static bool LRUBottomK_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUBottomK_find(cache_t *cache, const request_t *req,
                                     const bool update_cache);
static cache_obj_t *LRUBottomK_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUBottomK_to_evict(cache_t *cache, const request_t *req);
static void LRUBottomK_evict(cache_t *cache, const request_t *req);
static bool LRUBottomK_remove(cache_t *cache, const obj_id_t obj_id);
static void LRUBottomK_print_cache(const cache_t *cache);
static void LRUBottomK_parse_params(cache_t *cache,
                                     const char *cache_specific_params);

// Helper to print percentile stats
static void print_reuse_stats(const char* label, std::vector<int64_t>& distances, int64_t never_reused) {
  if (distances.empty()) {
    fprintf(stderr, "  %s: no data\n", label);
    return;
  }
  std::sort(distances.begin(), distances.end());
  size_t n = distances.size();
  fprintf(stderr, "  %s (n=%zu, never_reused=%ld):\n", label, n, never_reused);
  fprintf(stderr, "    p10=%ld p25=%ld p50=%ld p75=%ld p90=%ld p99=%ld\n",
          distances[n * 10 / 100],
          distances[n * 25 / 100],
          distances[n * 50 / 100],
          distances[n * 75 / 100],
          distances[n * 90 / 100],
          distances[n * 99 / 100]);

  // Also compute mean
  double sum = 0;
  for (auto d : distances) sum += d;
  fprintf(stderr, "    mean=%.1f\n", sum / n);
}

// ***********************************************************************
// ****                   Initialization                              ****
// ***********************************************************************

cache_t *LRUBottomK_init(const common_cache_params_t ccache_params,
                          const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("LRUBottomK", ccache_params, cache_specific_params);
  cache->cache_init = LRUBottomK_init;
  cache->cache_free = LRUBottomK_free;
  cache->get = LRUBottomK_get;
  cache->find = LRUBottomK_find;
  cache->insert = LRUBottomK_insert;
  cache->evict = LRUBottomK_evict;
  cache->remove = LRUBottomK_remove;
  cache->to_evict = LRUBottomK_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;
  cache->print_cache = LRUBottomK_print_cache;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2;  // prev/next pointers
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(LRUBottomK_params_t));
  memset(cache->eviction_params, 0, sizeof(LRUBottomK_params_t));
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;

  // Set defaults
  params->num_candidates = 4;
  params->recent_window = 32;
  params->lr = 0.2;
  params->ctx_speed = 0.1;
  params->min_access_count = 2;
  params->use_random = false;
  params->n_evict = 0;
  params->n_kept = 0;
  params->n_no_embedding = 0;

  LRUBottomK_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    LRUBottomK_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  emb->set_lr(params->lr);
  emb->set_ctx_speed(params->ctx_speed);
  params->embedding_manager = static_cast<void *>(emb);

  if (params->use_random) {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
             "LRUBottomK-k%d-rand",
             params->num_candidates);
  } else {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
             "LRUBottomK-k%d-w%d",
             params->num_candidates, params->recent_window);
  }

  // Reset reuse tracker
  reuse_tracker = ReuseTracker();

  return cache;
}

static void LRUBottomK_free(cache_t *cache) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Print stats on exit
  fprintf(stderr, "\n[LRUBottomK] Final stats:\n");
  fprintf(stderr, "  evictions=%ld, kept_and_promoted=%ld, no_embedding=%ld\n",
          params->n_evict, params->n_kept, params->n_no_embedding);
  fprintf(stderr, "  kept/evict ratio: %.2f\n",
          params->n_evict > 0 ? (double)params->n_kept / params->n_evict : 0.0);

  // Print reuse distance stats
  fprintf(stderr, "\n[Reuse Distance Analysis]\n");
  print_reuse_stats("EVICTED", reuse_tracker.evicted_reuse_distances, reuse_tracker.evicted_never_reused);
  print_reuse_stats("KEPT (moved to MRU)", reuse_tracker.kept_reuse_distances, reuse_tracker.kept_never_reused);

  // Cleanup
  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ***********************************************************************
// ****                   Core Operations                             ****
// ***********************************************************************

static bool LRUBottomK_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Track access for embeddings on EVERY request
  emb->on_access(req->obj_id);

  // Track reuse for evicted objects
  auto it = reuse_tracker.eviction_time.find(req->obj_id);
  if (it != reuse_tracker.eviction_time.end()) {
    int64_t reuse_dist = cache->n_req - it->second;
    reuse_tracker.evicted_reuse_distances.push_back(reuse_dist);
    reuse_tracker.eviction_time.erase(it);
  }

  // Track reuse for kept objects
  auto it2 = reuse_tracker.kept_time.find(req->obj_id);
  if (it2 != reuse_tracker.kept_time.end()) {
    int64_t reuse_dist = cache->n_req - it2->second;
    reuse_tracker.kept_reuse_distances.push_back(reuse_dist);
    reuse_tracker.kept_time.erase(it2);
  }

  return cache_get_base(cache, req);
}

static cache_obj_t *LRUBottomK_find(cache_t *cache, const request_t *req,
                                     const bool update_cache) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    // LRU: move to head on access
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
  }
  return cache_obj;
}

static cache_obj_t *LRUBottomK_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);

  return obj;
}

static cache_obj_t *LRUBottomK_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);

  DEBUG_ASSERT(params->q_tail != NULL || cache->occupied_byte == 0);

  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return params->q_tail;
}

// ***********************************************************************
// ****                   Eviction Logic                              ****
// ***********************************************************************

static void LRUBottomK_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  DEBUG_ASSERT(params->q_tail != NULL);

  // Collect bottom K candidates
  std::vector<cache_obj_t*> candidates;
  std::vector<double> similarities;

  cache_obj_t *obj = params->q_tail;
  int collected = 0;
  while (obj != NULL && collected < params->num_candidates) {
    candidates.push_back(obj);

    // Compute max similarity to recent accesses
    double sim = -1.0;  // -1 means no embedding
    if (emb->has_embedding(obj->obj_id) &&
        emb->get_access_count(obj->obj_id) >= params->min_access_count) {
      sim = emb->max_similarity_to_recent(obj->obj_id);
    }
    similarities.push_back(sim);

    obj = obj->queue.prev;
    collected++;
  }

  // Find the candidate to evict
  int worst_idx = 0;

  if (params->use_random) {
    // Random selection, but prioritize freq=1 objects (no embedding)
    std::vector<int> no_emb_indices;
    std::vector<int> has_emb_indices;
    for (int i = 0; i < (int)candidates.size(); i++) {
      if (similarities[i] < 0) {
        no_emb_indices.push_back(i);
      } else {
        has_emb_indices.push_back(i);
      }
    }
    if (!no_emb_indices.empty()) {
      // Evict a random one without embedding (freq < min_access_count)
      worst_idx = no_emb_indices[rand() % no_emb_indices.size()];
    } else {
      // All have embeddings, pick random
      worst_idx = rand() % candidates.size();
    }
  } else {
    // Embedding-based: find candidate with worst (lowest) similarity
    double worst_sim = similarities[0];
    bool all_no_embedding = (similarities[0] < 0);

    for (int i = 1; i < (int)candidates.size(); i++) {
      if (similarities[i] < 0) {
        // No embedding - treat as worst if current worst has embedding
        if (worst_sim >= 0) {
          worst_idx = i;
          worst_sim = similarities[i];
        }
      } else {
        all_no_embedding = false;
        if (worst_sim < 0 || similarities[i] < worst_sim) {
          worst_idx = i;
          worst_sim = similarities[i];
        }
      }
    }

    if (all_no_embedding) {
      params->n_no_embedding++;
    }
  }

  // Evict the worst candidate
  cache_obj_t *to_evict = candidates[worst_idx];

  // First, remove to_evict from its current position in the list
  remove_obj_from_list(&params->q_head, &params->q_tail, to_evict);

  // Track eviction for reuse distance analysis
  reuse_tracker.eviction_time[to_evict->obj_id] = cache->n_req;

  // Evict it
  cache_evict_base(cache, to_evict, true);
  params->n_evict++;

  // Move the other candidates to MRU
  for (int i = 0; i < (int)candidates.size(); i++) {
    if (i != worst_idx) {
      // Track that we kept this object
      reuse_tracker.kept_time[candidates[i]->obj_id] = cache->n_req;
      params->n_kept++;

      // Move to MRU position
      move_obj_to_head(&params->q_head, &params->q_tail, candidates[i]);
    }
  }

  // Print debug stats periodically
  if (params->n_evict % 100000 == 0 && params->n_evict > 0) {
    fprintf(stderr, "[LRUBottomK] evicts=%ld kept=%ld (ratio=%.2f) no_emb=%ld\n",
            params->n_evict, params->n_kept,
            (double)params->n_kept / params->n_evict,
            params->n_no_embedding);
  }
}

static bool LRUBottomK_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

static void LRUBottomK_print_cache(const cache_t *cache) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);
  cache_obj_t *cur = params->q_head;
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

static void LRUBottomK_parse_params(cache_t *cache,
                                     const char *cache_specific_params) {
  auto *params = static_cast<LRUBottomK_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') {
      params_str++;
    }

    if (strcasecmp(key, "num-candidates") == 0) {
      params->num_candidates = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "lr") == 0) {
      params->lr = strtod(value, NULL);
    } else if (strcasecmp(key, "ctx-speed") == 0) {
      params->ctx_speed = strtod(value, NULL);
    } else if (strcasecmp(key, "min-access-count") == 0) {
      params->min_access_count = atoi(value);
    } else if (strcasecmp(key, "random") == 0) {
      params->use_random = (atoi(value) != 0);
    } else if (strcasecmp(key, "print") == 0) {
      printf("LRUBottomK: num-candidates=%d, recent-window=%d\n",
             params->num_candidates, params->recent_window);
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
