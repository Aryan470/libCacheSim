//
//  LRUFreq: LRU with frequency-based forgiveness
//
//  Baseline policy to compare against embedding-based forgiveness.
//  Forgives objects at eviction time if their access count >= min-freq threshold.
//
//  Parameters:
//    min-freq: minimum access count to qualify for forgiveness (default: 3)
//    max-forgives: max forgives per eviction cycle (default: 5, -1=unlimited)
//    max-freq-entries: max entries in frequency map (default: 100000)
//

#include <list>
#include <unordered_map>

#include "dataStructure/hashtable/hashtable.h"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// LRU-capped Frequency Tracker (or unlimited if max_entries == -1)
// ============================================================================
class FrequencyTracker {
public:
  int64_t max_entries_;  // -1 means unlimited

  FrequencyTracker(int64_t max_entries = 100000) : max_entries_(max_entries) {}

  void on_access(uint64_t obj_id) {
    auto it = freq_map_.find(obj_id);
    if (it != freq_map_.end()) {
      // Existing entry: increment count and move to front (if capped mode)
      it->second.count++;
      if (max_entries_ >= 0) {
        move_to_front(obj_id);
      }
    } else {
      // New entry
      if (max_entries_ >= 0) {
        // Capped mode: evict LRU if at capacity, then add with LRU tracking
        while (static_cast<int64_t>(freq_map_.size()) >= max_entries_ && !lru_list_.empty()) {
          evict_lru();
        }
        lru_list_.push_front(obj_id);
        freq_map_[obj_id] = {1, lru_list_.begin()};
      } else {
        // Unlimited mode: just add without LRU tracking
        freq_map_[obj_id] = {1, lru_list_.end()};
      }
    }
  }

  int get_count(uint64_t obj_id) const {
    auto it = freq_map_.find(obj_id);
    return (it != freq_map_.end()) ? it->second.count : 0;
  }

  size_t size() const { return freq_map_.size(); }

private:
  struct Entry {
    int count;
    std::list<uint64_t>::iterator lru_iter;
  };

  void move_to_front(uint64_t obj_id) {
    auto it = freq_map_.find(obj_id);
    if (it != freq_map_.end()) {
      lru_list_.erase(it->second.lru_iter);
      lru_list_.push_front(obj_id);
      it->second.lru_iter = lru_list_.begin();
    }
  }

  void evict_lru() {
    if (lru_list_.empty()) return;
    uint64_t lru_id = lru_list_.back();
    lru_list_.pop_back();
    freq_map_.erase(lru_id);
  }

  std::unordered_map<uint64_t, Entry> freq_map_;
  std::list<uint64_t> lru_list_;
};

// ============================================================================
// Cache Parameters
// ============================================================================
typedef struct {
  cache_obj_t *q_head;
  cache_obj_t *q_tail;

  // Frequency tracking (LRU-capped, or unlimited if -1)
  void *freq_tracker;

  // Forgiveness parameters
  int min_freq;
  int max_forgives;
  int64_t max_freq_entries;  // -1 means unlimited

  // Stats
  int64_t n_forgive;
  int64_t n_evict;
  int64_t n_candidates;
} LRUFreq_params_t;

static const char *DEFAULT_CACHE_PARAMS = "min-freq=3,max-forgives=5,max-freq-entries=-1";

// Function declarations
static void LRUFreq_free(cache_t *cache);
static bool LRUFreq_get(cache_t *cache, const request_t *req);
static cache_obj_t *LRUFreq_find(cache_t *cache, const request_t *req, const bool update_cache);
static cache_obj_t *LRUFreq_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LRUFreq_to_evict(cache_t *cache, const request_t *req);
static void LRUFreq_evict(cache_t *cache, const request_t *req);
static bool LRUFreq_remove(cache_t *cache, const obj_id_t obj_id);
static void LRUFreq_parse_params(cache_t *cache, const char *cache_specific_params);

// ============================================================================
// Initialization
// ============================================================================
cache_t *LRUFreq_init(const common_cache_params_t ccache_params,
                       const char *cache_specific_params) {
  cache_t *cache = cache_struct_init("LRUFreq", ccache_params, cache_specific_params);
  cache->cache_init = LRUFreq_init;
  cache->cache_free = LRUFreq_free;
  cache->get = LRUFreq_get;
  cache->find = LRUFreq_find;
  cache->insert = LRUFreq_insert;
  cache->evict = LRUFreq_evict;
  cache->remove = LRUFreq_remove;
  cache->to_evict = LRUFreq_to_evict;
  cache->get_occupied_byte = cache_get_occupied_byte_default;
  cache->can_insert = cache_can_insert_default;
  cache->get_n_obj = cache_get_n_obj_default;

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 2;
  } else {
    cache->obj_md_size = 0;
  }

  cache->eviction_params = malloc(sizeof(LRUFreq_params_t));
  memset(cache->eviction_params, 0, sizeof(LRUFreq_params_t));
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  params->q_head = NULL;
  params->q_tail = NULL;

  // Defaults
  params->min_freq = 3;
  params->max_forgives = 5;
  params->max_freq_entries = 100000;

  // Parse params
  LRUFreq_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    LRUFreq_parse_params(cache, cache_specific_params);
  }

  // Create frequency tracker with parsed max entries
  params->freq_tracker = static_cast<void *>(new FrequencyTracker(params->max_freq_entries));

  snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "LRUFreq-f%d", params->min_freq);

  return cache;
}

static void LRUFreq_free(cache_t *cache) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  auto *tracker = static_cast<FrequencyTracker *>(params->freq_tracker);

  double forgive_rate = params->n_candidates > 0
      ? 100.0 * params->n_forgive / params->n_candidates : 0.0;
  if (params->max_freq_entries < 0) {
    fprintf(stderr, "[LRUFreq min-freq=%d max-freq-entries=unlimited] evicts=%ld candidates=%ld forgives=%ld(%.1f%%) final_freq_entries=%zu\n",
            params->min_freq,
            params->n_evict, params->n_candidates,
            params->n_forgive, forgive_rate, tracker->size());
  } else {
    fprintf(stderr, "[LRUFreq min-freq=%d max-freq-entries=%ld] evicts=%ld candidates=%ld forgives=%ld(%.1f%%) final_freq_entries=%zu\n",
            params->min_freq, params->max_freq_entries,
            params->n_evict, params->n_candidates,
            params->n_forgive, forgive_rate, tracker->size());
  }

  delete tracker;
  free(cache->eviction_params);
  cache_struct_free(cache);
}

// ============================================================================
// Core Operations
// ============================================================================
static bool LRUFreq_get(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  auto *tracker = static_cast<FrequencyTracker *>(params->freq_tracker);

  // Track access count (LRU-capped)
  tracker->on_access(req->obj_id);

  return cache_get_base(cache, req);
}

static cache_obj_t *LRUFreq_find(cache_t *cache, const request_t *req, const bool update_cache) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  cache_obj_t *cache_obj = cache_find_base(cache, req, update_cache);

  if (cache_obj && likely(update_cache)) {
    move_obj_to_head(&params->q_head, &params->q_tail, cache_obj);
  }
  return cache_obj;
}

static cache_obj_t *LRUFreq_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  prepend_obj_to_head(&params->q_head, &params->q_tail, obj);

  return obj;
}

static cache_obj_t *LRUFreq_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return params->q_tail;
}

// ============================================================================
// Eviction with Frequency-based Forgiveness
// ============================================================================
static void LRUFreq_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);
  auto *tracker = static_cast<FrequencyTracker *>(params->freq_tracker);

  int forgives_remaining = params->max_forgives;
  bool evicted = false;

  while (!evicted && params->q_tail != NULL) {
    cache_obj_t *obj = params->q_tail;
    params->n_candidates++;

    bool should_forgive = false;

    // Check frequency-based forgiveness
    if (forgives_remaining != 0) {
      int freq = tracker->get_count(obj->obj_id);

      if (freq >= params->min_freq) {
        should_forgive = true;
        if (forgives_remaining > 0) forgives_remaining--;
        params->n_forgive++;
      }
    }

    if (should_forgive) {
      // Forgive: move to head (MRU position)
      move_obj_to_head(&params->q_head, &params->q_tail, obj);
    } else {
      // Evict
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

static bool LRUFreq_remove(cache_t *cache, const obj_id_t obj_id) {
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == NULL) {
    return false;
  }
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);

  remove_obj_from_list(&params->q_head, &params->q_tail, obj);
  cache_remove_obj_base(cache, obj, true);

  return true;
}

// ============================================================================
// Parameter Parsing
// ============================================================================
// Helper to parse max-freq-entries which can be a number, -1 (unlimited), or Nx (multiplier of cache size)
static int64_t parse_max_freq_entries(const char *value, int64_t cache_size) {
  if (value == NULL) return -1;
  size_t len = strlen(value);
  if (len == 0) return -1;

  // Check for Nx format (e.g., "2x", "0.5x")
  if (value[len - 1] == 'x' || value[len - 1] == 'X') {
    double multiplier = strtod(value, NULL);
    // Estimate number of objects: cache_size / average_object_size
    // Using 1KB as rough average object size estimate
    int64_t estimated_objects = cache_size / 1024;
    return static_cast<int64_t>(multiplier * estimated_objects);
  }

  return strtoll(value, NULL, 10);
}

static void LRUFreq_parse_params(cache_t *cache, const char *cache_specific_params) {
  auto *params = static_cast<LRUFreq_params_t *>(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ') params_str++;

    if (strcasecmp(key, "min-freq") == 0) {
      params->min_freq = atoi(value);
    } else if (strcasecmp(key, "max-forgives") == 0) {
      params->max_forgives = atoi(value);
    } else if (strcasecmp(key, "max-freq-entries") == 0) {
      params->max_freq_entries = parse_max_freq_entries(value, cache->cache_size);
    } else if (strcasecmp(key, "print") == 0) {
      printf("min-freq=%d,max-forgives=%d,max-freq-entries=%ld\n",
             params->min_freq, params->max_forgives, params->max_freq_entries);
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
