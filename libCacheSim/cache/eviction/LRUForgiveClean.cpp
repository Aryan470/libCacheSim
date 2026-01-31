//
//  LRUForgiveClean: Clean implementation of LRU with embedding-based forgiveness
//
//  Key differences from LRUForgive:
//    1. RANDOM embedding initialization (not context-copy)
//    2. Embeddings kept forever (not removed on eviction)
//    3. Hardcoded optimal parameters from standalone validation
//
//  Parameters (validated in standalone simulator):
//    - lr = 0.2
//    - ctx_speed = 0.001 (100x slower than default)
//    - threshold = 0.7
//    - min_access_count = 2
//    - max_forgives = 5
//    - avg_top_k(3) similarity
//

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include "dataStructure/hashtable/hashtable.h"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Embedding System Constants (validated parameters)
// ============================================================================
static constexpr int K = 8;                    // Number of context vectors
static constexpr int D = 8;                    // Dimensions per vector
static constexpr double LR = 0.2;              // Learning rate
static constexpr double CTX_SPEED = 0.001;     // Context speed (100x slower)
static constexpr int RECENT_WINDOW = 16;       // Recent window size
static constexpr int MIN_ACCESS_COUNT = 2;     // Min freq before forgiveness
static constexpr double FORGIVE_THRESHOLD = 0.7; // Similarity threshold
static constexpr int MAX_FORGIVES = 5;         // Max forgives per eviction

using EmbeddingArray = std::array<std::array<double, D>, K>;

// ============================================================================
// Embedded Embedding Manager (with RANDOM initialization)
// ============================================================================
class CleanEmbeddingManager {
public:
  CleanEmbeddingManager(uint64_t seed = 42) : rng_(seed) {
    init_context();
    recent_embeddings_.resize(RECENT_WINDOW);
    recent_ids_.resize(RECENT_WINDOW);
    recent_valid_.resize(RECENT_WINDOW, false);
  }

  void on_access(uint64_t obj_id) {
    // Perturb context every 10 accesses with sqrt(10) scaling
    if (++perturb_counter_ >= 10) {
      perturb_counter_ = 0;
      perturb_context_10x();
    }

    int& count = access_count_[obj_id];
    count++;

    auto it = embeddings_.find(obj_id);
    bool has_emb = (it != embeddings_.end());

    if (has_emb) {
      update_embedding(obj_id);
      update_recent(obj_id);
    } else if (count == 2) {
      init_embedding(obj_id);
      update_recent(obj_id);
    }
  }

  double avg_top_k_similarity_to_recent(uint64_t obj_id, int top_k = 3) {
    auto it = embeddings_.find(obj_id);
    if (it == embeddings_.end()) {
      return 0.0;
    }

    std::vector<double> sims;
    sims.reserve(recent_count_);

    for (int i = 0; i < recent_count_; i++) {
      if (recent_ids_[i] == obj_id) continue;
      if (!recent_valid_[i]) continue;

      const auto& emb = it->second;
      const auto& recent_emb = recent_embeddings_[i];
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += dot_product(emb[k].data(), recent_emb[k].data());
      }
      sims.push_back(sum / K);
    }

    if (sims.empty()) {
      return 0.0;
    }

    std::sort(sims.begin(), sims.end(), std::greater<double>());
    int count = std::min(top_k, static_cast<int>(sims.size()));
    double total = 0.0;
    for (int i = 0; i < count; i++) {
      total += sims[i];
    }
    return total / count;
  }

  int get_access_count(uint64_t obj_id) const {
    auto it = access_count_.find(obj_id);
    return (it != access_count_.end()) ? it->second : 0;
  }

  bool has_embedding(uint64_t obj_id) const {
    return embeddings_.count(obj_id) > 0;
  }

  // NOTE: We do NOT remove embeddings on eviction (key insight from validation)

private:
  void init_context() {
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] = normal_dist_(rng_);
      }
      normalize(context_[k]);
    }
  }

  void perturb_context_10x() {
    double scale = CTX_SPEED * std::sqrt(10.0);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] += normal_dist_(rng_) * scale;
      }
      normalize(context_[k]);
    }
  }

  // CRITICAL: Random initialization (NOT context-copy)
  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = normal_dist_(rng_);  // RANDOM, not context
      }
      normalize(emb[k].data());
    }
  }

  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = LR * context_[k][d] + (1.0 - LR) * emb[k][d];
      }
      normalize(emb[k].data());
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
    recent_idx_ = (recent_idx_ + 1) % RECENT_WINDOW;
    if (recent_count_ < RECENT_WINDOW) recent_count_++;
  }

  static void normalize(double* vec) {
    double norm = 0.0;
    for (int i = 0; i < D; i++) {
      norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-10) {
      for (int i = 0; i < D; i++) {
        vec[i] /= norm;
      }
    }
  }

  static double dot_product(const double* a, const double* b) {
    double result = 0.0;
    for (int i = 0; i < D; i++) {
      result += a[i] * b[i];
    }
    return result;
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

  std::mt19937 rng_;  // 32-bit to match standalone
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

// ============================================================================
// Cache Parameters
// ============================================================================
typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;
  void *embedding_manager;  // CleanEmbeddingManager*
  int64_t n_forgive;
  int64_t n_evict;
  int64_t n_candidates;
  int64_t n_qualified;
} LRUForgiveClean_params_t;

// Function declarations
static void LRUForgiveClean_free(cache_t *cache);
static bool LRUForgiveClean_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgiveClean_find(cache_t *cache, const request_t *req,
                                          const bool update_cache);
static cache_obj_t *LRUForgiveClean_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUForgiveClean_to_evict(cache_t *cache, const request_t *req);
static void LRUForgiveClean_evict(cache_t *cache, const request_t *req);
static bool LRUForgiveClean_remove(cache_t *cache, const obj_id_t obj_id);

// ============================================================================
// Initialization
// ============================================================================
cache_t *LRUForgiveClean_init(const common_cache_params_t ccache_params,
                               const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("LRUForgiveClean", ccache_params, cache_specific_params);
  cache->cache_init = LRUForgiveClean_init;
  cache->cache_free = LRUForgiveClean_free;
  cache->get = LRUForgiveClean_get;
  cache->find = LRUForgiveClean_find;
  cache->insert = LRUForgiveClean_insert;
  cache->evict = LRUForgiveClean_evict;
  cache->remove = LRUForgiveClean_remove;
  cache->to_evict = LRUForgiveClean_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2;  // prev/next pointers
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(LRUForgiveClean_params_t));
  memset(cache->eviction_params, 0, sizeof(LRUForgiveClean_params_t));
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;
  params->n_forgive = 0;
  params->n_evict = 0;
  params->n_candidates = 0;
  params->n_qualified = 0;

  // Create embedding manager with random initialization
  params->embedding_manager = static_cast<void *>(new CleanEmbeddingManager(42));

  return cache;
}

static void LRUForgiveClean_free(cache_t *cache) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<CleanEmbeddingManager *>(params->embedding_manager);

  // Print final stats
  double forgive_rate = params->n_qualified > 0
      ? 100.0 * params->n_forgive / params->n_qualified : 0.0;
  fprintf(stderr, "[LRUForgiveClean] evicts=%ld qualified=%ld forgives=%ld (%.1f%%)\n",
          params->n_evict, params->n_qualified, params->n_forgive, forgive_rate);

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ============================================================================
// Core Operations
// ============================================================================
static bool LRUForgiveClean_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<CleanEmbeddingManager *>(params->embedding_manager);

  // Track access for embeddings on EVERY request
  emb->on_access(req->obj_id);

  return cache_get_base(cache, req);
}

static cache_obj_t *LRUForgiveClean_find(cache_t *cache, const request_t *req,
                                          const bool update_cache) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    // LRU: move to head on access
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
  }
  return cache_obj;
}

static cache_obj_t *LRUForgiveClean_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);

  return obj;
}

static cache_obj_t *LRUForgiveClean_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return params->q_tail;
}

// ============================================================================
// Eviction Logic (with embedding-based forgiveness)
// ============================================================================
static void LRUForgiveClean_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<CleanEmbeddingManager *>(params->embedding_manager);

  DEBUG_ASSERT(params->q_tail != NULL);

  int forgives_remaining = MAX_FORGIVES;
  bool evicted = false;

  while (!evicted && params->q_tail != NULL) {
    cache_obj_t *obj = params->q_tail;
    params->n_candidates++;

    bool should_forgive = false;

    if (forgives_remaining > 0) {
      int access_count = emb->get_access_count(obj->obj_id);

      if (access_count >= MIN_ACCESS_COUNT) {
        params->n_qualified++;

        double similarity = emb->avg_top_k_similarity_to_recent(obj->obj_id, 3);
        if (similarity >= FORGIVE_THRESHOLD) {
          should_forgive = true;
          forgives_remaining--;
          params->n_forgive++;
        }
      }
    }

    if (should_forgive) {
      // Forgive: move object to head (MRU position)
      move_obj_to_head(&params->q_head, &params->q_tail, obj);
    } else {
      // Evict: remove from tail
      // NOTE: We do NOT remove embedding (key insight)
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

static bool LRUForgiveClean_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<LRUForgiveClean_params_t *>(cache->eviction_params);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

#ifdef __cplusplus
}
#endif
