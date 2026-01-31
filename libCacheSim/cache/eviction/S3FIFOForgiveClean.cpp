//
//  S3FIFOForgiveClean: S3-FIFO with embedding-based forgiveness on main queue
//
//  Configurable embedding parameters via -e flag:
//    lr=0.2, ctx-speed=0.001, threshold=0.5, window=16, min-access=2, max-forgives=5
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
// Embedding System Constants (fixed dimensions)
// ============================================================================
static constexpr int K = 8;  // Number of context vectors
static constexpr int D = 8;  // Dimensions per vector

using EmbeddingArray = std::array<std::array<double, D>, K>;

// ============================================================================
// Configurable Embedding Manager
// ============================================================================
class S3FIFOEmbeddingManager {
public:
  double lr_;
  double ctx_speed_;
  int recent_window_;
  int min_access_count_;

  S3FIFOEmbeddingManager(double lr, double ctx_speed, int window, int min_access, uint64_t seed = 42)
      : lr_(lr), ctx_speed_(ctx_speed), recent_window_(window), min_access_count_(min_access), rng_(seed) {
    init_context();
    recent_embeddings_.resize(recent_window_);
    recent_ids_.resize(recent_window_);
    recent_valid_.resize(recent_window_, false);
  }

  void on_access(uint64_t obj_id) {
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
    } else if (count == min_access_count_) {
      init_embedding(obj_id);
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
        sum += dot_product(emb[k].data(), recent_emb[k].data());
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
    double scale = ctx_speed_ * std::sqrt(10.0);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] += normal_dist_(rng_) * scale;
      }
      normalize(context_[k]);
    }
  }

  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = normal_dist_(rng_);
      }
      normalize(emb[k].data());
    }
  }

  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = lr_ * context_[k][d] + (1.0 - lr_) * emb[k][d];
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
    recent_idx_ = (recent_idx_ + 1) % recent_window_;
    if (recent_count_ < recent_window_) recent_count_++;
  }

  static void normalize(double* vec) {
    double norm = 0.0;
    for (int i = 0; i < D; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    if (norm > 1e-10) {
      for (int i = 0; i < D; i++) vec[i] /= norm;
    }
  }

  static double dot_product(const double* a, const double* b) {
    double result = 0.0;
    for (int i = 0; i < D; i++) result += a[i] * b[i];
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
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

// ============================================================================
// Cache Parameters
// ============================================================================
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

  // Embedding parameters (configurable)
  double emb_lr;
  double emb_ctx_speed;
  double emb_threshold;
  int emb_window;
  int emb_min_access;
  int emb_max_forgives;

  // Embedding system
  void *embedding_manager;

  // Stats
  int64_t n_forgive;
  int64_t n_evict_main;
  int64_t n_evict_small;
  int64_t n_candidates;
  int64_t n_qualified;
} S3FIFOForgiveClean_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2,"
    "lr=0.2,ctx-speed=0.001,threshold=0.5,window=16,min-access=2,max-forgives=5";

// Function declarations
static void S3FIFOForgiveClean_free(cache_t *cache);
static bool S3FIFOForgiveClean_get(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFOForgiveClean_find(cache_t *cache, const request_t *req, const bool update_cache);
static cache_obj_t *S3FIFOForgiveClean_insert(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFOForgiveClean_to_evict(cache_t *cache, const request_t *req);
static void S3FIFOForgiveClean_evict(cache_t *cache, const request_t *req);
static bool S3FIFOForgiveClean_remove(cache_t *cache, const obj_id_t obj_id);
static inline int64_t S3FIFOForgiveClean_get_occupied_byte(const cache_t *cache);
static inline int64_t S3FIFOForgiveClean_get_n_obj(const cache_t *cache);
static inline bool S3FIFOForgiveClean_can_insert(cache_t *cache, const request_t *req);
static void S3FIFOForgiveClean_parse_params(cache_t *cache, const char *cache_specific_params);
static void S3FIFOForgiveClean_evict_small(cache_t *cache, const request_t *req);
static void S3FIFOForgiveClean_evict_main(cache_t *cache, const request_t *req);

// ============================================================================
// Initialization
// ============================================================================
cache_t *S3FIFOForgiveClean_init(const common_cache_params_t ccache_params,
                                  const char *cache_specific_params) {
  cache_t *cache = cache_struct_init("S3FIFOForgiveClean", ccache_params, cache_specific_params);
  cache->cache_init = S3FIFOForgiveClean_init;
  cache->cache_free = S3FIFOForgiveClean_free;
  cache->get = S3FIFOForgiveClean_get;
  cache->find = S3FIFOForgiveClean_find;
  cache->insert = S3FIFOForgiveClean_insert;
  cache->evict = S3FIFOForgiveClean_evict;
  cache->remove = S3FIFOForgiveClean_remove;
  cache->to_evict = S3FIFOForgiveClean_to_evict;
  cache->get_n_obj = S3FIFOForgiveClean_get_n_obj;
  cache->get_occupied_byte = S3FIFOForgiveClean_get_occupied_byte;
  cache->can_insert = S3FIFOForgiveClean_can_insert;
  cache->obj_md_size = 0;

  cache->eviction_params = malloc(sizeof(S3FIFOForgiveClean_params_t));
  memset(cache->eviction_params, 0, sizeof(S3FIFOForgiveClean_params_t));
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  params->req_local = new_request();
  params->hit_on_ghost = false;

  // Parse default params first, then user params
  S3FIFOForgiveClean_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    S3FIFOForgiveClean_parse_params(cache, cache_specific_params);
  }

  int64_t small_fifo_size = (int64_t)ccache_params.cache_size * params->small_size_ratio;
  int64_t main_fifo_size = ccache_params.cache_size - small_fifo_size;
  int64_t ghost_fifo_size = (int64_t)(ccache_params.cache_size * params->ghost_size_ratio);

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

  // Initialize embedding manager with parsed params
  params->embedding_manager = static_cast<void *>(
      new S3FIFOEmbeddingManager(params->emb_lr, params->emb_ctx_speed,
                                  params->emb_window, params->emb_min_access, 42));

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
           "S3FIFOForgiveClean-lr%.2f-th%.2f", params->emb_lr, params->emb_threshold);

  return cache;
}

static void S3FIFOForgiveClean_free(cache_t *cache) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<S3FIFOEmbeddingManager *>(params->embedding_manager);

  double forgive_rate = params->n_qualified > 0
      ? 100.0 * params->n_forgive / params->n_qualified : 0.0;
  fprintf(stderr, "[S3FIFOForgiveClean lr=%.2f ctx=%.4f th=%.2f w=%d] "
          "evict_main=%ld evict_small=%ld qualified=%ld forgives=%ld (%.1f%%)\n",
          params->emb_lr, params->emb_ctx_speed, params->emb_threshold, params->emb_window,
          params->n_evict_main, params->n_evict_small,
          params->n_qualified, params->n_forgive, forgive_rate);

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

// ============================================================================
// Core Operations
// ============================================================================
static bool S3FIFOForgiveClean_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<S3FIFOEmbeddingManager *>(params->embedding_manager);

  emb->on_access(req->obj_id);
  return cache_get_base(cache, req);
}

static cache_obj_t *S3FIFOForgiveClean_find(cache_t *cache, const request_t *req, const bool update_cache) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);

  if (!update_cache) {
    cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, false);
    if (obj != NULL) return obj;
    return params->main_fifo->find(params->main_fifo, req, false);
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

static cache_obj_t *S3FIFOForgiveClean_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  cache_obj_t *obj = NULL;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (params->hit_on_ghost) {
    params->hit_on_ghost = false;
    obj = main_fifo->insert(main_fifo, req);
  } else {
    if (req->obj_size >= small_fifo->cache_size) return NULL;
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

static cache_obj_t *S3FIFOForgiveClean_to_evict(cache_t *cache, const request_t *req) {
  assert(false);
  return NULL;
}

// ============================================================================
// Eviction with Embedding-based Forgiveness
// ============================================================================
static void S3FIFOForgiveClean_evict_small(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<S3FIFOEmbeddingManager *>(params->embedding_manager);
  cache_t *small_fifo = params->small_fifo;
  cache_t *ghost_fifo = params->ghost_fifo;
  cache_t *main_fifo = params->main_fifo;

  int forgives_remaining = params->emb_max_forgives;
  bool has_evicted = false;

  while (!has_evicted && small_fifo->get_occupied_byte(small_fifo) > 0) {
    cache_obj_t *obj_to_evict = small_fifo->to_evict(small_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    uint64_t obj_id = obj_to_evict->obj_id;
    int freq = obj_to_evict->S3FIFO.freq;
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    // Embedding-based freq boost: if similar, increment freq
    if (forgives_remaining > 0 && freq < params->move_to_main_threshold) {
      if (emb->has_embedding(obj_id)) {
        params->n_qualified++;
        double similarity = emb->avg_top_k_similarity_to_recent(obj_id, 3);
        if (similarity >= params->emb_threshold) {
          freq++;  // Boost freq
          obj_to_evict->S3FIFO.freq = freq;
          forgives_remaining--;
          params->n_forgive++;
        }
      }
    }

    // Normal S3-FIFO logic with potentially boosted freq
    if (freq >= params->move_to_main_threshold) {
      main_fifo->insert(main_fifo, params->req_local);
    } else {
      if (ghost_fifo != NULL) {
        ghost_fifo->get(ghost_fifo, params->req_local);
      }
      has_evicted = true;
      params->n_evict_small++;
    }

    bool removed = small_fifo->remove(small_fifo, params->req_local->obj_id);
    DEBUG_ASSERT(removed);
  }
}

static void S3FIFOForgiveClean_evict_main(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  auto *emb = static_cast<S3FIFOEmbeddingManager *>(params->embedding_manager);
  cache_t *main_fifo = params->main_fifo;

  int forgives_remaining = params->emb_max_forgives;
  bool has_evicted = false;

  while (!has_evicted && main_fifo->get_occupied_byte(main_fifo) > 0) {
    cache_obj_t *obj_to_evict = main_fifo->to_evict(main_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    int freq = obj_to_evict->S3FIFO.freq;
    uint64_t obj_id = obj_to_evict->obj_id;
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    // Embedding-based freq boost for objects about to be evicted (freq=0)
    if (freq == 0 && forgives_remaining > 0) {
      if (emb->has_embedding(obj_id)) {
        params->n_qualified++;
        double similarity = emb->avg_top_k_similarity_to_recent(obj_id, 3);
        if (similarity >= params->emb_threshold) {
          freq = 1;  // Boost freq, giving it another reinsertion cycle
          forgives_remaining--;
          params->n_forgive++;
        }
      }
    }

    // Normal S3-FIFO main queue logic with potentially boosted freq
    if (freq >= 1) {
      main_fifo->remove(main_fifo, obj_id);
      cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
      new_obj->S3FIFO.freq = MIN(freq, 3) - 1;
    } else {
      main_fifo->remove(main_fifo, obj_id);
      has_evicted = true;
      params->n_evict_main++;
    }
  }
}

static void S3FIFOForgiveClean_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  params->has_evicted = true;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (main_fifo->get_occupied_byte(main_fifo) > main_fifo->cache_size ||
      small_fifo->get_occupied_byte(small_fifo) == 0) {
    S3FIFOForgiveClean_evict_main(cache, req);
  } else {
    S3FIFOForgiveClean_evict_small(cache, req);
  }
}

static bool S3FIFOForgiveClean_remove(cache_t *cache, const obj_id_t obj_id) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  bool removed = false;
  removed = removed || params->small_fifo->remove(params->small_fifo, obj_id);
  removed = removed || (params->ghost_fifo && params->ghost_fifo->remove(params->ghost_fifo, obj_id));
  removed = removed || params->main_fifo->remove(params->main_fifo, obj_id);
  return removed;
}

static inline int64_t S3FIFOForgiveClean_get_occupied_byte(const cache_t *cache) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  return params->small_fifo->get_occupied_byte(params->small_fifo) +
         params->main_fifo->get_occupied_byte(params->main_fifo);
}

static inline int64_t S3FIFOForgiveClean_get_n_obj(const cache_t *cache) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  return params->small_fifo->get_n_obj(params->small_fifo) +
         params->main_fifo->get_n_obj(params->main_fifo);
}

static inline bool S3FIFOForgiveClean_can_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);
  return req->obj_size <= params->small_fifo->cache_size && cache_can_insert_default(cache, req);
}

// ============================================================================
// Parameter Parsing
// ============================================================================
static void S3FIFOForgiveClean_parse_params(cache_t *cache, const char *cache_specific_params) {
  auto *params = static_cast<S3FIFOForgiveClean_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') params_str++;

    if (strcasecmp(key, "small-size-ratio") == 0 || strcasecmp(key, "fifo-size-ratio") == 0) {
      params->small_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "ghost-size-ratio") == 0) {
      params->ghost_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "move-to-main-threshold") == 0) {
      params->move_to_main_threshold = atoi(value);
    } else if (strcasecmp(key, "lr") == 0) {
      params->emb_lr = strtod(value, NULL);
    } else if (strcasecmp(key, "ctx-speed") == 0) {
      params->emb_ctx_speed = strtod(value, NULL);
    } else if (strcasecmp(key, "threshold") == 0 || strcasecmp(key, "th") == 0) {
      params->emb_threshold = strtod(value, NULL);
    } else if (strcasecmp(key, "window") == 0) {
      params->emb_window = atoi(value);
    } else if (strcasecmp(key, "min-access") == 0) {
      params->emb_min_access = atoi(value);
    } else if (strcasecmp(key, "max-forgives") == 0) {
      params->emb_max_forgives = atoi(value);
    } else if (strcasecmp(key, "print") == 0) {
      printf("lr=%.2f,ctx-speed=%.4f,threshold=%.2f,window=%d,min-access=%d,max-forgives=%d\n",
             params->emb_lr, params->emb_ctx_speed, params->emb_threshold,
             params->emb_window, params->emb_min_access, params->emb_max_forgives);
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
