//
//  SIEVE with Bottom-K Selection: Modified SIEVE that examines K candidates
//  with freq=0 and evicts the one with worst embedding similarity to recent
//  accesses. The other K-1 candidates are skipped (hand advances past them).
//

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reuse distance tracking
struct SieveReuseTracker {
  std::unordered_map<uint64_t, int64_t> eviction_time;
  std::unordered_map<uint64_t, int64_t> skipped_time;  // objects we skipped (didn't evict)
  std::vector<int64_t> evicted_reuse_distances;
  std::vector<int64_t> skipped_reuse_distances;
};

static SieveReuseTracker reuse_tracker;

typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;
  cache_obj_t *pointer;         // SIEVE hand
  void *embedding_manager;
  int num_candidates;           // K candidates to examine (default: 4)
  int recent_window;            // window for embedding similarity (default: 8)
  int min_access_count;         // minimum accesses before using embeddings
  bool use_random;              // use random selection instead of embeddings
  int64_t n_evict;
  int64_t n_skipped;            // candidates we skipped
  int64_t n_no_embedding;
} SieveBottomK_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "num-candidates=4,recent-window=8,min-access-count=2";

// Function declarations
static void SieveBottomK_free(cache_t *cache);
static bool SieveBottomK_get(cache_t *cache, const request_t *req);
static cache_obj_t *SieveBottomK_find(cache_t *cache, const request_t *req,
                                       const bool update_cache);
static cache_obj_t *SieveBottomK_insert(cache_t *cache, const request_t *req);
static cache_obj_t *SieveBottomK_to_evict(cache_t *cache, const request_t *req);
static void SieveBottomK_evict(cache_t *cache, const request_t *req);
static bool SieveBottomK_remove(cache_t *cache, const obj_id_t obj_id);
static void SieveBottomK_parse_params(cache_t *cache,
                                       const char *cache_specific_params);

static void print_reuse_stats(const char* label, std::vector<int64_t>& distances) {
  if (distances.empty()) {
    fprintf(stderr, "  %s: no data\n", label);
    return;
  }
  std::sort(distances.begin(), distances.end());
  size_t n = distances.size();
  double sum = 0;
  for (auto d : distances) sum += d;
  fprintf(stderr, "  %s (n=%zu):\n", label, n);
  fprintf(stderr, "    p10=%ld p25=%ld p50=%ld p75=%ld p90=%ld\n",
          distances[n * 10 / 100],
          distances[n * 25 / 100],
          distances[n * 50 / 100],
          distances[n * 75 / 100],
          distances[n * 90 / 100]);
  fprintf(stderr, "    mean=%.1f\n", sum / n);
}

cache_t *SieveBottomK_init(const common_cache_params_t ccache_params,
                            const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("SieveBottomK", ccache_params, cache_specific_params);
  cache->cache_init = SieveBottomK_init;
  cache->cache_free = SieveBottomK_free;
  cache->get = SieveBottomK_get;
  cache->find = SieveBottomK_find;
  cache->insert = SieveBottomK_insert;
  cache->evict = SieveBottomK_evict;
  cache->remove = SieveBottomK_remove;
  cache->to_evict = SieveBottomK_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 1;  // visited bit
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(SieveBottomK_params_t));
  memset(cache->eviction_params, 0, sizeof(SieveBottomK_params_t));
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;
  params->pointer = NULL;

  // Set defaults
  params->num_candidates = 4;
  params->recent_window = 8;
  params->min_access_count = 2;
  params->use_random = false;
  params->n_evict = 0;
  params->n_skipped = 0;
  params->n_no_embedding = 0;

  SieveBottomK_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    SieveBottomK_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  params->embedding_manager = static_cast<void *>(emb);

  if (params->use_random) {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
             "SieveBottomK-k%d-rand", params->num_candidates);
  } else {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
             "SieveBottomK-k%d-w%d", params->num_candidates, params->recent_window);
  }

  // Reset reuse tracker
  reuse_tracker = SieveReuseTracker();

  return cache;
}

static void SieveBottomK_free(cache_t *cache) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  fprintf(stderr, "\n[SieveBottomK] Final stats:\n");
  fprintf(stderr, "  evictions=%ld, skipped=%ld, no_embedding=%ld\n",
          params->n_evict, params->n_skipped, params->n_no_embedding);

  fprintf(stderr, "\n[Reuse Distance Analysis]\n");
  print_reuse_stats("EVICTED", reuse_tracker.evicted_reuse_distances);
  print_reuse_stats("SKIPPED (not evicted)", reuse_tracker.skipped_reuse_distances);

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

static bool SieveBottomK_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  emb->on_access(req->obj_id);

  // Track reuse for evicted objects
  auto it = reuse_tracker.eviction_time.find(req->obj_id);
  if (it != reuse_tracker.eviction_time.end()) {
    int64_t reuse_dist = cache->n_req - it->second;
    reuse_tracker.evicted_reuse_distances.push_back(reuse_dist);
    reuse_tracker.eviction_time.erase(it);
  }

  // Track reuse for skipped objects (only on miss - when they were eventually evicted)
  auto it2 = reuse_tracker.skipped_time.find(req->obj_id);
  if (it2 != reuse_tracker.skipped_time.end()) {
    // Only record if this is a miss (object not in cache)
    cache_obj_t *obj = cache_find_base(cache, req, false);
    if (obj == NULL) {
      int64_t reuse_dist = cache->n_req - it2->second;
      reuse_tracker.skipped_reuse_distances.push_back(reuse_dist);
    }
    reuse_tracker.skipped_time.erase(it2);
  }

  return cache_get_base(cache, req);
}

static cache_obj_t *SieveBottomK_find(cache_t *cache, const request_t *req,
                                       const bool update_cache) {
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);
  if (cache_obj != NULL && update_cache) {
    cache_obj->sieve.freq = 1;  // Set visited bit
  }
  return cache_obj;
}

static cache_obj_t *SieveBottomK_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);
  obj->sieve.freq = 0;  // New objects start unvisited

  return obj;
}

static cache_obj_t *SieveBottomK_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);
  if (params->pointer == NULL) {
    params->pointer = params->q_tail;
  }
  return params->pointer;
}

static void SieveBottomK_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  if (params->pointer == NULL) {
    params->pointer = params->q_tail;
  }

  // Collect K candidates with freq=0, clearing freq bits as we go
  std::vector<cache_obj_t*> candidates;
  std::vector<double> similarities;
  cache_obj_t *scan = params->pointer;

  // Scan until we have K candidates or wrap around
  int scanned = 0;
  const int max_scan = cache->n_obj * 2;  // Prevent infinite loop

  while (candidates.size() < (size_t)params->num_candidates && scanned < max_scan) {
    if (scan == NULL) {
      scan = params->q_tail;
    }

    if (scan->sieve.freq == 0) {
      // This is a candidate
      candidates.push_back(scan);

      double sim = -1.0;
      if (emb->has_embedding(scan->obj_id) &&
          emb->get_access_count(scan->obj_id) >= params->min_access_count) {
        sim = emb->max_similarity_to_recent(scan->obj_id);
      }
      similarities.push_back(sim);
    } else {
      // Clear visited bit
      scan->sieve.freq = 0;
    }

    scan = scan->queue.prev;
    scanned++;

    if (scan == params->pointer && scanned > 1) {
      break;  // Wrapped around
    }
  }

  if (candidates.empty()) {
    // Fallback: evict at pointer
    cache_obj_t *obj = params->pointer;
    params->pointer = obj->queue.prev;
    if (params->pointer == NULL) {
      params->pointer = params->q_tail;
    }
    remove_obj_from_list(&params->q_head, &params->q_tail, obj);
    cache_evict_base(cache, obj, true);
    params->n_evict++;
    return;
  }

  // Find the candidate to evict
  int worst_idx = 0;

  if (params->use_random) {
    // Random: prioritize no-embedding, otherwise random
    std::vector<int> no_emb_indices;
    for (int i = 0; i < (int)candidates.size(); i++) {
      if (similarities[i] < 0) {
        no_emb_indices.push_back(i);
      }
    }
    if (!no_emb_indices.empty()) {
      worst_idx = no_emb_indices[rand() % no_emb_indices.size()];
    } else {
      worst_idx = rand() % candidates.size();
    }
  } else {
    // Embedding: find lowest similarity
    double worst_sim = similarities[0];
    bool all_no_embedding = (similarities[0] < 0);

    for (int i = 1; i < (int)candidates.size(); i++) {
      if (similarities[i] < 0) {
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
  reuse_tracker.eviction_time[to_evict->obj_id] = cache->n_req;

  // Track skipped candidates
  for (int i = 0; i < (int)candidates.size(); i++) {
    if (i != worst_idx) {
      reuse_tracker.skipped_time[candidates[i]->obj_id] = cache->n_req;
      params->n_skipped++;
    }
  }

  // Update hand to point past all candidates
  params->pointer = candidates.back()->queue.prev;
  if (params->pointer == NULL) {
    params->pointer = params->q_tail;
  }

  remove_obj_from_list(&params->q_head, &params->q_tail, to_evict);
  cache_evict_base(cache, to_evict, true);
  params->n_evict++;

  if (params->n_evict % 100000 == 0) {
    fprintf(stderr, "[SieveBottomK] evicts=%ld skipped=%ld no_emb=%ld\n",
            params->n_evict, params->n_skipped, params->n_no_embedding);
  }
}

static bool SieveBottomK_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);

  if (obj == params->pointer) {
    params->pointer = obj->queue.prev;
  }
  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

static void SieveBottomK_parse_params(cache_t *cache,
                                       const char *cache_specific_params) {
  auto *params = static_cast<SieveBottomK_params_t *>(cache->eviction_params);

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
    } else if (strcasecmp(key, "min-access-count") == 0) {
      params->min_access_count = atoi(value);
    } else if (strcasecmp(key, "random") == 0) {
      params->use_random = (atoi(value) != 0);
    } else if (strcasecmp(key, "print") == 0) {
      printf("SieveBottomK: num-candidates=%d, recent-window=%d\n",
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
