
#include <cstring>
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "dataStructure/hashtable/hashtable.h"
#include "embedding/EmbeddingManager.hpp"
#include "libCacheSim/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

// Diagnostic sample record
struct DiagSample {
  uint64_t obj_id;
  int64_t sample_time;      // when we sampled this object
  int64_t last_access_time; // when it was last accessed (for recency)
  int frequency;            // access count at sample time
  double similarity;        // avg similarity to recent at sample time
  int64_t reuse_distance;   // filled in when object is accessed again (-1 if not yet)
};

// Diagnostic tracker
struct DiagTracker {
  std::vector<DiagSample> samples;
  std::unordered_map<uint64_t, size_t> pending_samples; // obj_id -> index in samples
  std::unordered_map<uint64_t, int64_t> last_access_time; // track last access for all cached objects
  std::mt19937_64 rng{12345};
  int64_t n_samples_taken = 0;
  int64_t n_samples_resolved = 0;
  char output_path[256] = "/tmp/emb_diag_samples.csv";
};

static DiagTracker diag;

typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;
  void *embedding_manager;
  int recent_window;
  int sample_interval;  // sample every N accesses
  char output_path[256];
  double lr;         // learning rate
  double ctx_speed;  // context perturbation speed
} LRUEmbDiag_params_t;

static const int DEFAULT_RECENT_WINDOW = 16;
static const int DEFAULT_SAMPLE_INTERVAL = 1000;

static void LRUEmbDiag_free(cache_t *cache);
static bool LRUEmbDiag_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUEmbDiag_find(cache_t *cache, const request_t *req,
                                    const bool update_cache);
static cache_obj_t *LRUEmbDiag_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUEmbDiag_to_evict(cache_t *cache, const request_t *req);
static void LRUEmbDiag_evict(cache_t *cache, const request_t *req);
static bool LRUEmbDiag_remove(cache_t *cache, const obj_id_t obj_id);
static void LRUEmbDiag_parse_params(cache_t *cache, const char *cache_specific_params);

static void LRUEmbDiag_parse_params(cache_t *cache, const char *cache_specific_params) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
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

    if (key == NULL || value == NULL) break;

    if (strcasecmp(key, "recent-window") == 0) {
      params->recent_window = atoi(value);
    } else if (strcasecmp(key, "sample-interval") == 0) {
      params->sample_interval = atoi(value);
    } else if (strcasecmp(key, "output-path") == 0) {
      snprintf(params->output_path, sizeof(params->output_path), "%s", value);
    } else if (strcasecmp(key, "lr") == 0) {
      params->lr = atof(value);
    } else if (strcasecmp(key, "ctx-speed") == 0) {
      params->ctx_speed = atof(value);
    } else {
      ERROR("LRUEmbDiag does not have parameter %s\n", key);
      abort();
    }
  }

  if (params_str_orig != NULL) {
    free(params_str_orig);
  }
}

cache_t *LRUEmbDiag_init(const common_cache_params_t ccache_params,
                         const char *cache_specific_params) {
  cache_t *cache = cache_struct_init("LRUEmbDiag", ccache_params, cache_specific_params);
  cache->cache_init = LRUEmbDiag_init;
  cache->cache_free = LRUEmbDiag_free;
  cache->get = LRUEmbDiag_get;
  cache->find = LRUEmbDiag_find;
  cache->insert = LRUEmbDiag_insert;
  cache->evict = LRUEmbDiag_evict;
  cache->remove = LRUEmbDiag_remove;
  cache->to_evict = LRUEmbDiag_to_evict;

  cache->obj_md_size = 0;

  cache->eviction_params = my_malloc(LRUEmbDiag_params_t);
  memset(cache->eviction_params, 0, sizeof(LRUEmbDiag_params_t));
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;
  params->recent_window = DEFAULT_RECENT_WINDOW;
  params->sample_interval = DEFAULT_SAMPLE_INTERVAL;
  snprintf(params->output_path, sizeof(params->output_path), "/tmp/emb_diag_samples.csv");
  params->lr = 0.2;         // default
  params->ctx_speed = 0.1;  // default

  if (cache_specific_params != NULL) {
    LRUEmbDiag_parse_params(cache, cache_specific_params);
  }

  // Reset diagnostic tracker FIRST, then copy output path
  diag = DiagTracker();
  snprintf(diag.output_path, sizeof(diag.output_path), "%s", params->output_path);

  auto *emb = new embedding::EmbeddingManager(42);
  emb->set_recent_window(params->recent_window);
  emb->set_lr(params->lr);
  emb->set_ctx_speed(params->ctx_speed);
  params->embedding_manager = static_cast<void *>(emb);

  return cache;
}

static void print_correlation_analysis() {
  if (diag.samples.empty()) {
    fprintf(stderr, "No diagnostic samples collected\n");
    return;
  }

  // Filter to resolved samples only
  std::vector<DiagSample> resolved;
  for (const auto& s : diag.samples) {
    if (s.reuse_distance >= 0) {
      resolved.push_back(s);
    }
  }

  // Output raw data to CSV for plotting (include ALL samples, resolved and unresolved)
  FILE* csv = fopen(diag.output_path, "w");
  if (csv) {
    fprintf(csv, "similarity,reuse_distance,frequency,recency,resolved\n");
    for (const auto& s : diag.samples) {
      int64_t recency = s.sample_time - s.last_access_time;
      int is_resolved = (s.reuse_distance >= 0) ? 1 : 0;
      fprintf(csv, "%.6f,%ld,%d,%ld,%d\n", s.similarity, s.reuse_distance, s.frequency, recency, is_resolved);
    }
    fclose(csv);
    fprintf(stderr, "Wrote %zu samples (%zu resolved, %zu unresolved) to %s\n",
            diag.samples.size(), resolved.size(), diag.samples.size() - resolved.size(), diag.output_path);
  }

  fprintf(stderr, "\n=== Embedding Diagnostic Analysis ===\n");
  fprintf(stderr, "Total samples: %zu, Resolved (reaccessed): %zu\n",
          diag.samples.size(), resolved.size());

  if (resolved.size() < 100) {
    fprintf(stderr, "Not enough resolved samples for analysis\n");
    return;
  }

  // Sort by similarity and compute stats for quartiles
  std::sort(resolved.begin(), resolved.end(),
            [](const DiagSample& a, const DiagSample& b) { return a.similarity < b.similarity; });

  size_t n = resolved.size();
  size_t q_size = n / 4;

  auto compute_stats = [](const std::vector<DiagSample>& samples, size_t start, size_t end) {
    double sum_rd = 0, sum_sim = 0, sum_freq = 0, sum_recency = 0;
    for (size_t i = start; i < end; i++) {
      sum_rd += samples[i].reuse_distance;
      sum_sim += samples[i].similarity;
      sum_freq += samples[i].frequency;
      sum_recency += (samples[i].sample_time - samples[i].last_access_time);
    }
    size_t count = end - start;
    return std::make_tuple(sum_rd / count, sum_sim / count, sum_freq / count, sum_recency / count);
  };

  fprintf(stderr, "\nBy similarity quartile (Q1=lowest similarity, Q4=highest):\n");
  fprintf(stderr, "%-8s %12s %12s %12s %12s\n", "Quartile", "Avg_ReuseD", "Avg_Sim", "Avg_Freq", "Avg_Recency");

  const char* labels[] = {"Q1(low)", "Q2", "Q3", "Q4(high)"};
  for (int q = 0; q < 4; q++) {
    size_t start = q * q_size;
    size_t end = (q == 3) ? n : (q + 1) * q_size;
    auto [avg_rd, avg_sim, avg_freq, avg_recency] = compute_stats(resolved, start, end);
    fprintf(stderr, "%-8s %12.1f %12.4f %12.2f %12.1f\n",
            labels[q], avg_rd, avg_sim, avg_freq, avg_recency);
  }

  // Compute Pearson correlation between similarity and reuse distance
  double sum_sim = 0, sum_rd = 0, sum_sim2 = 0, sum_rd2 = 0, sum_sim_rd = 0;
  for (const auto& s : resolved) {
    sum_sim += s.similarity;
    sum_rd += s.reuse_distance;
    sum_sim2 += s.similarity * s.similarity;
    sum_rd2 += (double)s.reuse_distance * s.reuse_distance;
    sum_sim_rd += s.similarity * s.reuse_distance;
  }
  double n_d = (double)resolved.size();
  double cov = (sum_sim_rd - sum_sim * sum_rd / n_d) / n_d;
  double std_sim = std::sqrt(sum_sim2 / n_d - (sum_sim / n_d) * (sum_sim / n_d));
  double std_rd = std::sqrt(sum_rd2 / n_d - (sum_rd / n_d) * (sum_rd / n_d));
  double correlation = (std_sim > 0 && std_rd > 0) ? cov / (std_sim * std_rd) : 0;

  fprintf(stderr, "\nPearson correlation (similarity vs reuse_distance): %.4f\n", correlation);
  fprintf(stderr, "  (negative = high similarity predicts LOW reuse distance = GOOD)\n");
  fprintf(stderr, "  (positive = high similarity predicts HIGH reuse distance = BAD)\n");

  // Also compute correlation between recency and reuse distance for comparison
  double sum_rec = 0, sum_rec2 = 0, sum_rec_rd = 0;
  for (const auto& s : resolved) {
    double recency = s.sample_time - s.last_access_time;
    sum_rec += recency;
    sum_rec2 += recency * recency;
    sum_rec_rd += recency * s.reuse_distance;
  }
  double cov_rec = (sum_rec_rd - sum_rec * sum_rd / n_d) / n_d;
  double std_rec = std::sqrt(sum_rec2 / n_d - (sum_rec / n_d) * (sum_rec / n_d));
  double corr_recency = (std_rec > 0 && std_rd > 0) ? cov_rec / (std_rec * std_rd) : 0;

  fprintf(stderr, "Pearson correlation (recency vs reuse_distance): %.4f\n", corr_recency);
  fprintf(stderr, "  (positive = older objects have higher reuse distance = expected for LRU)\n");

  fprintf(stderr, "==========================================\n\n");
}

static void LRUEmbDiag_free(cache_t *cache) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  print_correlation_analysis();

  delete emb;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

static bool LRUEmbDiag_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  auto *emb = static_cast<embedding::EmbeddingManager *>(params->embedding_manager);

  // Update embedding on every access
  emb->on_access(req->obj_id);

  // Check if this access resolves any pending samples
  auto it = diag.pending_samples.find(req->obj_id);
  if (it != diag.pending_samples.end()) {
    size_t idx = it->second;
    diag.samples[idx].reuse_distance = cache->n_req - diag.samples[idx].sample_time;
    diag.pending_samples.erase(it);
    diag.n_samples_resolved++;
  }

  // Periodically sample a random object from the cache
  if (cache->n_req % params->sample_interval == 0 && cache->n_obj > 0) {
    // Walk to a random position in the LRU list
    std::uniform_int_distribution<int64_t> dist(0, cache->n_obj - 1);
    int64_t pos = dist(diag.rng);

    cache_obj_t *obj = params->q_head;
    for (int64_t i = 0; i < pos && obj != NULL; i++) {
      obj = obj->queue.next;
    }

    if (obj != NULL && emb->has_embedding(obj->obj_id)) {
      // Don't re-sample if already pending
      if (diag.pending_samples.find(obj->obj_id) == diag.pending_samples.end()) {
        DiagSample sample;
        sample.obj_id = obj->obj_id;
        sample.sample_time = cache->n_req;
        // Get last access time from our tracker
        auto lat_it = diag.last_access_time.find(obj->obj_id);
        sample.last_access_time = (lat_it != diag.last_access_time.end()) ? lat_it->second : 0;
        sample.frequency = emb->get_access_count(obj->obj_id);
        sample.similarity = emb->max_similarity_to_recent(obj->obj_id);
        sample.reuse_distance = -1; // not yet resolved

        diag.pending_samples[obj->obj_id] = diag.samples.size();
        diag.samples.push_back(sample);
        diag.n_samples_taken++;
      }
    }
  }

  return cache_get_base(cache, req);
}

static cache_obj_t *LRUEmbDiag_find(cache_t *cache, const request_t *req,
                                    const bool update_cache) {
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);
  if (cache_obj != NULL && update_cache) {
    auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
    diag.last_access_time[req->obj_id] = cache->n_req;
  }
  return cache_obj;
}

static cache_obj_t *LRUEmbDiag_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);
  diag.last_access_time[req->obj_id] = cache->n_req;

  return obj;
}

static cache_obj_t *LRUEmbDiag_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  return params->q_tail;
}

static void LRUEmbDiag_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  cache_obj_t *obj = params->q_tail;

  // If this object was sampled and pending, mark as evicted (infinite reuse distance)
  auto it = diag.pending_samples.find(obj->obj_id);
  if (it != diag.pending_samples.end()) {
    // Leave reuse_distance as -1 to indicate "never reaccessed before eviction"
    diag.pending_samples.erase(it);
  }

  // Clean up last access time tracking
  diag.last_access_time.erase(obj->obj_id);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_evict_base(cache, obj, true);
}

static void LRUEmbDiag_remove_obj(cache_t *cache, cache_obj_t *obj_to_remove) {
  DEBUG_ASSERT(obj_to_remove != NULL);
  auto *params = static_cast<LRUEmbDiag_params_t *>(cache->eviction_params);
  remove_obj_from_list(&params->q_head, &params->q_tail, obj_to_remove);
  cache_remove_obj_base(cache, obj_to_remove, true);
}

static bool LRUEmbDiag_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  LRUEmbDiag_remove_obj(cache, obj);
  return true;
}

#ifdef __cplusplus
}
#endif
