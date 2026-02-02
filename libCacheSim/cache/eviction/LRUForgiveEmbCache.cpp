//
//  LRUForgiveEmbCache: LRU with embedding-based forgiveness
//  Features:
//    - Self-contained embedding manager with LRU eviction (configurable size)
//    - Variance-preserving update dynamics
//    - Cosine similarity for comparison
//
//  Configurable parameters:
//    min-access-count, forgive-threshold, max-forgives, recent-window,
//    lr, ctx-speed, max-emb-entries
//

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

#include "dataStructure/hashtable/hashtable.h"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Embedding System Constants
// ============================================================================
static constexpr int K = 8;  // Number of context vectors
static constexpr int D = 8;  // Dimensions per vector

using EmbeddingArray = std::array<std::array<double, D>, K>;

// ============================================================================
// Embedding Manager with LRU eviction and variance-preserving dynamics
// (or unlimited if max_entries == -1)
// ============================================================================
class LRUForgiveEmbCacheEmbeddingManager {
public:
  double lr_;
  double ctx_speed_;
  int recent_window_;
  int min_access_count_;
  int64_t max_entries_;  // -1 means unlimited

  LRUForgiveEmbCacheEmbeddingManager(double lr, double ctx_speed, int window, int min_access,
                                      int64_t max_entries = 100000, uint64_t seed = 42)
      : lr_(lr), ctx_speed_(ctx_speed), recent_window_(window), min_access_count_(min_access),
        max_entries_(max_entries), rng_(seed) {
    init_context();
    recent_embeddings_.resize(recent_window_);
    recent_ids_.resize(recent_window_);
    recent_valid_.resize(recent_window_, false);
  }

  void on_access(uint64_t obj_id) {
    // Update context every 10 accesses (variance-preserving with 10x speed)
    if (++perturb_counter_ >= 10) {
      perturb_counter_ = 0;
      update_context_batch();
    }

    int& count = access_count_[obj_id];
    count++;

    auto it = embeddings_.find(obj_id);
    bool has_emb = (it != embeddings_.end());

    if (has_emb) {
      update_embedding(obj_id);
      if (max_entries_ >= 0) {
        move_to_front(obj_id);  // LRU: mark as recently used (only in capped mode)
      }
      update_recent(obj_id);
    } else if (count == min_access_count_) {
      if (max_entries_ >= 0) {
        // Capped mode: evict LRU if at capacity before adding new embedding
        while (static_cast<int64_t>(embeddings_.size()) >= max_entries_ && !lru_list_.empty()) {
          evict_lru();
        }
        init_embedding(obj_id);
        add_to_lru(obj_id);
      } else {
        // Unlimited mode: just add without LRU tracking
        init_embedding(obj_id);
      }
      update_recent(obj_id);
    }
  }

  double avg_top_k_similarity_to_recent(uint64_t obj_id, int top_k = 3) {
    auto it = embeddings_.find(obj_id);
    if (it == embeddings_.end()) return 0.0;

    std::vector<double> sims;
    sims.reserve(recent_count_);

    for (int i = 0; i < recent_count_; i++) {
      if (recent_ids_[i] == obj_id) continue;
      if (!recent_valid_[i]) continue;

      const auto& emb = it->second;
      const auto& recent_emb = recent_embeddings_[i];
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += cosine_similarity(emb[k].data(), recent_emb[k].data());
      }
      sims.push_back(sum / K);
    }

    if (sims.empty()) return 0.0;

    std::sort(sims.begin(), sims.end(), std::greater<double>());
    int count = std::min(top_k, static_cast<int>(sims.size()));
    double total = 0.0;
    for (int i = 0; i < count; i++) total += sims[i];
    return total / count;
  }

  int get_access_count(uint64_t obj_id) const {
    auto it = access_count_.find(obj_id);
    return (it != access_count_.end()) ? it->second : 0;
  }

  bool has_embedding(uint64_t obj_id) const {
    return embeddings_.count(obj_id) > 0;
  }

  size_t get_num_embeddings() const { return embeddings_.size(); }

private:
  // LRU list management
  void add_to_lru(uint64_t obj_id) {
    lru_list_.push_front(obj_id);
    lru_map_[obj_id] = lru_list_.begin();
  }

  void move_to_front(uint64_t obj_id) {
    auto it = lru_map_.find(obj_id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_list_.push_front(obj_id);
      it->second = lru_list_.begin();
    }
  }

  void evict_lru() {
    if (lru_list_.empty()) return;
    uint64_t lru_id = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(lru_id);
    embeddings_.erase(lru_id);
  }

  // Initialize context ~ N(0, I)
  void init_context() {
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] = normal_dist_(rng_);
      }
    }
  }

  // Variance-preserving context update (batched every 10 accesses)
  // C = sqrt(1 - 10*speed) * C + sqrt(10*speed) * N(0, I)
  void update_context_batch() {
    double effective_speed = 10.0 * ctx_speed_;
    double decay = std::sqrt(1.0 - effective_speed);
    double noise_scale = std::sqrt(effective_speed);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] = decay * context_[k][d] + noise_scale * normal_dist_(rng_);
      }
    }
  }

  // Initialize embedding ~ N(0, I)
  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = normal_dist_(rng_);
      }
    }
  }

  // Variance-preserving embedding update
  // E = sqrt(1 - lr) * E + sqrt(lr) * C
  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    double decay = std::sqrt(1.0 - lr_);
    double ctx_scale = std::sqrt(lr_);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = decay * emb[k][d] + ctx_scale * context_[k][d];
      }
    }
  }

  void update_recent(uint64_t obj_id) {
    recent_ids_[recent_idx_] = obj_id;
    auto it = embeddings_.find(obj_id);
    if (it != embeddings_.end()) {
      recent_embeddings_[recent_idx_] = it->second;
      recent_valid_[recent_idx_] = true;
    } else {
      recent_valid_[recent_idx_] = false;
    }
    recent_idx_ = (recent_idx_ + 1) % recent_window_;
    if (recent_count_ < recent_window_) recent_count_++;
  }

  static double cosine_similarity(const double* a, const double* b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < D; i++) {
      dot += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
    }
    double denom = std::sqrt(norm_a * norm_b);
    if (denom < 1e-10) return 0.0;
    return dot / denom;
  }

  double context_[K][D];
  std::unordered_map<uint64_t, EmbeddingArray> embeddings_;
  std::unordered_map<uint64_t, int> access_count_;
  std::vector<EmbeddingArray> recent_embeddings_;
  std::vector<uint64_t> recent_ids_;
  std::vector<bool> recent_valid_;
  int recent_idx_ = 0;
  int recent_count_ = 0;
  int perturb_counter_ = 0;
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};

  // LRU tracking for embedding cache
  std::list<uint64_t> lru_list_;
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map_;
};

// ============================================================================
// Cache Parameters
// ============================================================================
typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;
  void *embedding_manager;

  // Forgiveness parameters
  int min_access_count;
  double forgive_threshold;
  int max_forgives;
  int recent_window;

  // Embedding parameters
  double lr;
  double ctx_speed;
  int64_t max_emb_entries;  // -1 means unlimited

  // Stats
  int64_t n_forgive;
  int64_t n_evict;
  int64_t n_candidates;
  int64_t n_qualified;
} LRUForgiveEmbCache_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "min-access-count=3,forgive-threshold=0.325,max-forgives=5,recent-window=16,"
    "lr=0.2,ctx-speed=0.001,max-emb-entries=100000";

// Function declarations
static void LRUForgiveEmbCache_free(cache_t *cache);
static bool LRUForgiveEmbCache_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgiveEmbCache_find(cache_t *cache, const request_t *req, const bool update_cache);
static cache_obj_t *LRUForgiveEmbCache_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgiveEmbCache_to_evict(cache_t *cache, const request_t *req);
static void LRUForgiveEmbCache_evict(cache_t *cache, const request_t *req);
static bool LRUForgiveEmbCache_remove(cache_t *cache, const obj_id_t obj_id);
static void LRUForgiveEmbCache_parse_params(cache_t *cache, const char *cache_specific_params);

// ============================================================================
// Initialization
// ============================================================================
cache_t *LRUForgiveEmbCache_init(const common_cache_params_t ccache_params,
                                  const char *cache_specific_params) {
  cache_t *cache = cache_struct_init("LRUForgiveEmbCache", ccache_params, cache_specific_params);
  cache->cache_init = LRUForgiveEmbCache_init;
  cache->cache_free = LRUForgiveEmbCache_free;
  cache->get = LRUForgiveEmbCache_get;
  cache->find = LRUForgiveEmbCache_find;
  cache->insert = LRUForgiveEmbCache_insert;
  cache->evict = LRUForgiveEmbCache_evict;
  cache->remove = LRUForgiveEmbCache_remove;
  cache->to_evict = LRUForgiveEmbCache_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2;
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(LRUForgiveEmbCache_params_t));
  memset(cache->eviction_params, 0, sizeof(LRUForgiveEmbCache_params_t));
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;

  // Parse default params first, then user params
  LRUForgiveEmbCache_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    LRUForgiveEmbCache_parse_params(cache, cache_specific_params);
  }

  // Create embedding manager
  params->embedding_manager = static_cast<void *>(
      new LRUForgiveEmbCacheEmbeddingManager(params->lr, params->ctx_speed,
                                              params->recent_window, params->min_access_count,
                                              params->max_emb_entries, 42));

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
           "LRUForgiveEmbCache-th%.2f-lr%.2f", params->forgive_threshold, params->lr);

  return cache;
}

static void LRUForgiveEmbCache_free(cache_t *cache) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  auto *emb = static_cast<LRUForgiveEmbCacheEmbeddingManager *>(params->embedding_manager);

  double forgive_rate = params->n_qualified > 0
      ? 100.0 * params->n_forgive / params->n_qualified : 0.0;
  if (params->max_emb_entries < 0) {
    fprintf(stderr, "[LRUForgiveEmbCache lr=%.2f ctx=%.4f th=%.2f max_emb=unlimited] "
            "evicts=%ld candidates=%ld qualified=%ld forgives=%ld(%.1f%%) final_emb_count=%zu\n",
            params->lr, params->ctx_speed, params->forgive_threshold,
            params->n_evict, params->n_candidates, params->n_qualified,
            params->n_forgive, forgive_rate, emb->get_num_embeddings());
  } else {
    fprintf(stderr, "[LRUForgiveEmbCache lr=%.2f ctx=%.4f th=%.2f max_emb=%ld] "
            "evicts=%ld candidates=%ld qualified=%ld forgives=%ld(%.1f%%) final_emb_count=%zu\n",
            params->lr, params->ctx_speed, params->forgive_threshold, params->max_emb_entries,
            params->n_evict, params->n_candidates, params->n_qualified,
            params->n_forgive, forgive_rate, emb->get_num_embeddings());
  }

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ============================================================================
// Core Operations
// ============================================================================
static bool LRUForgiveEmbCache_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  auto *emb = static_cast<LRUForgiveEmbCacheEmbeddingManager *>(params->embedding_manager);

  emb->on_access(req->obj_id);
  return cache_get_base(cache, req);
}

static cache_obj_t *LRUForgiveEmbCache_find(cache_t *cache, const request_t *req, const bool update_cache) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
  }
  return cache_obj;
}

static cache_obj_t *LRUForgiveEmbCache_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);

  return obj;
}

static cache_obj_t *LRUForgiveEmbCache_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return params->q_tail;
}

// ============================================================================
// Eviction with Embedding-based Forgiveness
// ============================================================================
static void LRUForgiveEmbCache_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);
  auto *emb = static_cast<LRUForgiveEmbCacheEmbeddingManager *>(params->embedding_manager);

  int forgives_remaining = params->max_forgives;
  bool evicted = false;

  while (!evicted && params->q_tail != NULL) {
    cache_obj_t *obj = params->q_tail;
    params->n_candidates++;

    bool should_forgive = false;

    if (forgives_remaining != 0) {
      int access_count = emb->get_access_count(obj->obj_id);

      if (access_count >= params->min_access_count) {
        params->n_qualified++;

        if (emb->has_embedding(obj->obj_id)) {
          double similarity = emb->avg_top_k_similarity_to_recent(obj->obj_id, 3);
          if (similarity >= params->forgive_threshold) {
            should_forgive = true;
            if (forgives_remaining > 0) forgives_remaining--;
            params->n_forgive++;
          }
        }
      }
    }

    if (should_forgive) {
      // Forgive: move object to head (MRU position)
      move_obj_to_head(&params->q_head, &params->q_tail, obj);
    } else {
      // Evict (embedding cache manages its own LRU evictions)
      params->n_evict++;

      params->q_tail = params->q_tail->queue.prev;
      if (likely(params->q_tail != NULL)) {
        params->q_tail->queue.next = NULL;
      } else {
        params->q_head = NULL;
      }

      cache_evict_base(cache, obj, true);
      evicted = true;
    }
  }
}

static bool LRUForgiveEmbCache_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

// ============================================================================
// Parameter Parsing
// ============================================================================
static void LRUForgiveEmbCache_parse_params(cache_t *cache, const char *cache_specific_params) {
  auto *params = static_cast<LRUForgiveEmbCache_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') params_str++;

    if (strcasecmp(key, "min-access-count") == 0) {
      params->min_access_count = atoi(value);
    } else if (strcasecmp(key, "forgive-threshold") == 0 || strcasecmp(key, "threshold") == 0 || strcasecmp(key, "th") == 0) {
      params->forgive_threshold = strtod(value, NULL);
    } else if (strcasecmp(key, "max-forgives") == 0) {
      params->max_forgives = atoi(value);
    } else if (strcasecmp(key, "recent-window") == 0 || strcasecmp(key, "window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "lr") == 0) {
      params->lr = strtod(value, NULL);
    } else if (strcasecmp(key, "ctx-speed") == 0) {
      params->ctx_speed = strtod(value, NULL);
    } else if (strcasecmp(key, "max-emb-entries") == 0) {
      params->max_emb_entries = strtoll(value, NULL, 10);
    } else if (strcasecmp(key, "print") == 0) {
      printf("min-access-count=%d,forgive-threshold=%.2f,max-forgives=%d,recent-window=%d,"
             "lr=%.2f,ctx-speed=%.4f,max-emb-entries=%ld\n",
             params->min_access_count, params->forgive_threshold, params->max_forgives,
             params->recent_window, params->lr, params->ctx_speed, params->max_emb_entries);
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
