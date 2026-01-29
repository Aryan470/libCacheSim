// Two-Tier LRU Simulator with Embedding Sampling
// Simulates RAM (1000) + Disk (100,000) with LRU eviction.
// Samples 1% of accesses with embedding scores and reuse distances.
//
// Compile: g++ -O3 -std=c++17 -o two_tier_sample two_tier_sample.cpp
// Usage: ./two_tier_sample <trace.csv> [max_req=1000000] [ram_size=1000] [disk_size=100000] [sample_rate=0.01] [key_type=0]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ==================== Embedding System ====================

static constexpr int K = 8;   // num context vectors
static constexpr int D = 8;   // dimensions per vector
static constexpr double DEFAULT_LR = 0.2;
static constexpr double DEFAULT_CTX_SPEED = 0.02;

class EmbeddingManager {
 public:
  using EmbeddingArray = std::array<std::array<double, D>, K>;

  EmbeddingManager(uint64_t seed = 42) : rng_(seed) {
    init_context();
  }

  void on_access(uint64_t obj_id) {
    access_count_[obj_id]++;
    int cnt = access_count_[obj_id];

    if (cnt == 2) {
      init_embedding(obj_id);
    } else if (cnt > 2) {
      update_embedding(obj_id);
    }
    perturb_context();
  }

  double max_similarity_to_recent(uint64_t obj_id) {
    if (!has_embedding(obj_id) || recent_count_ == 0) return -1.0;

    const auto& emb = embeddings_[obj_id];
    double max_sim = -2.0;

    int count = std::min(recent_count_, recent_window_);
    for (int i = 0; i < count; i++) {
      if (!recent_valid_[i]) continue;
      if (recent_ids_[i] == obj_id) continue;
      const auto& recent_emb = recent_embeddings_[i];

      // Mean over K vector pairs
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += emb[k][d] * recent_emb[k][d];
        }
        sum += dot;
      }
      double sim = sum / K;
      if (sim > max_sim) max_sim = sim;
    }
    return max_sim;
  }

  void update_recent(uint64_t obj_id) {
    if (recent_embeddings_.size() < (size_t)recent_window_) {
      recent_embeddings_.resize(recent_window_);
      recent_ids_.resize(recent_window_);
      recent_valid_.resize(recent_window_, false);
    }

    recent_ids_[recent_idx_] = obj_id;
    if (has_embedding(obj_id)) {
      recent_embeddings_[recent_idx_] = embeddings_[obj_id];
      recent_valid_[recent_idx_] = true;
    } else {
      recent_valid_[recent_idx_] = false;
    }

    recent_idx_ = (recent_idx_ + 1) % recent_window_;
    recent_count_++;
  }

  bool has_embedding(uint64_t obj_id) const {
    return embeddings_.count(obj_id) > 0;
  }

  int get_access_count(uint64_t obj_id) const {
    auto it = access_count_.find(obj_id);
    return it != access_count_.end() ? it->second : 0;
  }

  void set_ctx_speed(double speed) { ctx_speed_ = speed; }
  void set_lr(double lr) { lr_ = lr; }
  void set_recent_window(int size) { recent_window_ = size; }

 private:
  void init_context() {
    for (int k = 0; k < K; k++) {
      double norm = 0.0;
      for (int d = 0; d < D; d++) {
        context_[k][d] = normal_dist_(rng_);
        norm += context_[k][d] * context_[k][d];
      }
      norm = std::sqrt(norm);
      for (int d = 0; d < D; d++) {
        context_[k][d] /= norm;
      }
    }
  }

  void perturb_context() {
    for (int k = 0; k < K; k++) {
      double norm = 0.0;
      for (int d = 0; d < D; d++) {
        context_[k][d] += ctx_speed_ * normal_dist_(rng_);
        norm += context_[k][d] * context_[k][d];
      }
      norm = std::sqrt(norm);
      for (int d = 0; d < D; d++) {
        context_[k][d] /= norm;
      }
    }
  }

  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = context_[k][d];
      }
    }
  }

  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      double norm = 0.0;
      for (int d = 0; d < D; d++) {
        emb[k][d] = (1.0 - lr_) * emb[k][d] + lr_ * context_[k][d];
        norm += emb[k][d] * emb[k][d];
      }
      norm = std::sqrt(norm);
      for (int d = 0; d < D; d++) {
        emb[k][d] /= norm;
      }
    }
  }

  double context_[K][D];
  std::unordered_map<uint64_t, EmbeddingArray> embeddings_;
  std::unordered_map<uint64_t, int> access_count_;

  std::vector<EmbeddingArray> recent_embeddings_;
  std::vector<uint64_t> recent_ids_;
  std::vector<bool> recent_valid_;
  int recent_idx_ = 0;
  int recent_count_ = 0;
  int recent_window_ = 8;

  double lr_ = DEFAULT_LR;
  double ctx_speed_ = DEFAULT_CTX_SPEED;

  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

// ==================== LRU Tier with Decile Tracking ====================
// Uses 10 balanced segments to get O(1) decile lookup

class LRUTier {
 public:
  static constexpr int NUM_SEGMENTS = 10;

  LRUTier(int capacity) : capacity_(capacity), segment_capacity_(capacity / NUM_SEGMENTS) {
    // Ensure minimum segment size
    if (segment_capacity_ < 1) segment_capacity_ = 1;
  }

  bool contains(uint64_t obj) const {
    return map_.count(obj) > 0;
  }

  // Returns decile (1-10) if found, 0 if not found
  int get_decile(uint64_t obj) const {
    auto it = map_.find(obj);
    if (it == map_.end()) return 0;
    return it->second.segment + 1;  // 1-indexed decile
  }

  bool access(uint64_t obj) {
    auto it = map_.find(obj);
    if (it != map_.end()) {
      int seg = it->second.segment;
      segments_[seg].erase(it->second.iter);
      segments_[0].push_front(obj);
      it->second.iter = segments_[0].begin();
      it->second.segment = 0;
      rebalance();
      return true;
    }
    return false;
  }

  uint64_t insert(uint64_t obj) {
    uint64_t evicted = 0;
    if (total_size() >= capacity_) {
      // Evict from last non-empty segment
      for (int s = NUM_SEGMENTS - 1; s >= 0; s--) {
        if (!segments_[s].empty()) {
          evicted = segments_[s].back();
          map_.erase(evicted);
          segments_[s].pop_back();
          break;
        }
      }
    }
    segments_[0].push_front(obj);
    map_[obj] = {segments_[0].begin(), 0};
    rebalance();
    return evicted;
  }

  void remove(uint64_t obj) {
    auto it = map_.find(obj);
    if (it != map_.end()) {
      int seg = it->second.segment;
      segments_[seg].erase(it->second.iter);
      map_.erase(it);
      rebalance();
    }
  }

  int size() const { return total_size(); }
  int capacity() const { return capacity_; }

  // For compatibility: get_position returns approximate position
  int get_position(uint64_t obj) const {
    int decile = get_decile(obj);
    if (decile == 0) return -1;
    // Return middle of decile range
    int sz = total_size();
    return (decile - 1) * sz / NUM_SEGMENTS + sz / (2 * NUM_SEGMENTS);
  }

 private:
  int total_size() const {
    int total = 0;
    for (int s = 0; s < NUM_SEGMENTS; s++) {
      total += segments_[s].size();
    }
    return total;
  }

  void rebalance() {
    // Move items from overfull segments to underfull ones
    // Keep segments roughly balanced
    int total = total_size();
    int target_per_seg = std::max(1, total / NUM_SEGMENTS);

    // Move from front segments to back if overfull
    for (int s = 0; s < NUM_SEGMENTS - 1; s++) {
      while ((int)segments_[s].size() > target_per_seg + 1 && !segments_[s].empty()) {
        uint64_t obj = segments_[s].back();
        segments_[s].pop_back();
        segments_[s + 1].push_front(obj);
        auto it = map_.find(obj);
        if (it != map_.end()) {
          it->second.iter = segments_[s + 1].begin();
          it->second.segment = s + 1;
        }
      }
    }
  }

  struct Entry {
    std::list<uint64_t>::iterator iter;
    int segment;  // 0-9
  };

  int capacity_;
  int segment_capacity_;
  std::list<uint64_t> segments_[NUM_SEGMENTS];
  std::unordered_map<uint64_t, Entry> map_;
};

// ==================== Stack Distance Tracker ====================
// Uses a simple approximation: track position in access order list
// This is O(1) per access instead of O(n)

class StackDistanceTracker {
 public:
  // Record access and return stack distance (approximated as time since last access)
  // Returns -1 for first access to this object
  int64_t on_access(uint64_t obj_id) {
    auto it = last_access_time_.find(obj_id);
    int64_t stack_distance = -1;

    if (it != last_access_time_.end()) {
      // Approximate stack distance as number of accesses since last access
      // This is an upper bound on true stack distance
      stack_distance = current_time_ - it->second - 1;
    }

    last_access_time_[obj_id] = current_time_++;
    return stack_distance;
  }

 private:
  std::unordered_map<uint64_t, int64_t> last_access_time_;
  int64_t current_time_ = 0;
};

// ==================== Two-Tier Cache ====================

class TwoTierCache {
 public:
  TwoTierCache(int ram_capacity, int disk_capacity)
      : ram_(ram_capacity), disk_(disk_capacity) {}

  // Returns location string: ram_d1-ram_d10, disk_d1-disk_d10, or miss
  std::string access(uint64_t obj) {
    // Check RAM first - get_decile is O(1)
    int ram_decile = ram_.get_decile(obj);
    if (ram_decile > 0) {
      ram_hits_++;
      std::string location = "ram_d" + std::to_string(ram_decile);
      ram_.access(obj);  // move to MRU
      return location;
    }

    // Check disk - get_decile is O(1)
    int disk_decile = disk_.get_decile(obj);
    if (disk_decile > 0) {
      disk_hits_++;
      std::string location = "disk_d" + std::to_string(disk_decile);
      // Promote to RAM
      disk_.remove(obj);
      uint64_t evicted = ram_.insert(obj);
      if (evicted != 0) {
        // RAM eviction goes to disk
        disk_.insert(evicted);
      }
      return location;
    }

    // Miss - insert to disk
    misses_++;
    disk_.insert(obj);
    return "miss";
  }

  int64_t ram_hits() const { return ram_hits_; }
  int64_t disk_hits() const { return disk_hits_; }
  int64_t misses() const { return misses_; }

 private:
  LRUTier ram_;
  LRUTier disk_;
  int64_t ram_hits_ = 0;
  int64_t disk_hits_ = 0;
  int64_t misses_ = 0;
};

// ==================== Key Parsing ====================

int g_key_type = 0;  // 0 = numeric, 1 = hash string

uint64_t parse_key_numeric(const char* line, bool* valid) {
  // Format: timestamp,obj_id,...
  const char* p = line;
  while (*p && *p != ',') p++;
  if (*p == ',') p++;
  char* end;
  uint64_t val = strtoull(p, &end, 10);
  *valid = (end != p);
  return val;
}

uint64_t parse_key_hash(const char* line, bool* valid) {
  const char* p = line;
  while (*p && *p != ',') p++;
  if (*p == ',') p++;
  const char* key_start = p;
  while (*p && *p != ',') p++;
  int key_len = p - key_start;
  if (key_len < 1) {
    *valid = false;
    return 0;
  }
  uint64_t hash = 14695981039346656037ULL;
  for (int i = 0; i < key_len; i++) {
    hash ^= (uint64_t)key_start[i];
    hash *= 1099511628211ULL;
  }
  *valid = true;
  return hash;
}

uint64_t parse_key(const char* line, bool* valid) {
  return (g_key_type == 1) ? parse_key_hash(line, valid)
                           : parse_key_numeric(line, valid);
}

// ==================== Main ====================

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Two-Tier LRU Simulator with Embedding Sampling\n"
            "Usage: %s <trace.csv> [max_req] [ram_size] [disk_size] [sample_rate] [key_type]\n",
            argv[0]);
    fprintf(stderr, "  max_req: max requests (default: 1000000)\n");
    fprintf(stderr, "  ram_size: RAM tier capacity (default: 1000)\n");
    fprintf(stderr, "  disk_size: Disk tier capacity (default: 100000)\n");
    fprintf(stderr, "  sample_rate: fraction of accesses to sample (default: 0.01)\n");
    fprintf(stderr, "  key_type: 0=numeric, 1=hash (default: 0)\n");
    return 1;
  }

  const char* path = argv[1];
  int64_t max_req = (argc > 2) ? atoll(argv[2]) : 1000000;
  int ram_size = (argc > 3) ? atoi(argv[3]) : 1000;
  int disk_size = (argc > 4) ? atoi(argv[4]) : 100000;
  double sample_rate = (argc > 5) ? atof(argv[5]) : 0.01;
  if (argc > 6) g_key_type = atoi(argv[6]);

  fprintf(stderr, "=== TWO-TIER LRU SIMULATOR WITH EMBEDDING SAMPLING ===\n");
  fprintf(stderr, "Trace: %s\n", path);
  fprintf(stderr, "Config: RAM=%d, Disk=%d, sample_rate=%.3f\n\n", ram_size, disk_size, sample_rate);

  // ==================== Pass 1: Build reuse distance map ====================
  fprintf(stderr, "Pass 1: Building reuse distance map...\n");

  std::vector<std::pair<int64_t, uint64_t>> accesses;  // (access_idx, obj_id)

  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    return 1;
  }

  char line[4096];
  fgets(line, sizeof(line), f);  // skip header

  int64_t idx = 0;
  while (fgets(line, sizeof(line), f) && idx < max_req) {
    bool valid;
    uint64_t obj = parse_key(line, &valid);
    if (!valid) continue;
    accesses.push_back({idx, obj});
    idx++;
  }
  fclose(f);

  int64_t total_requests = idx;
  fprintf(stderr, "  Read %ld requests\n", total_requests);

  // Build next_access map: for each access index, when is the next access to same object?
  std::unordered_map<uint64_t, int64_t> last_seen;  // obj_id -> last access index
  std::unordered_map<int64_t, int64_t> reuse_distance;  // access_idx -> reuse_distance

  // Scan backwards to build reuse distances
  for (int64_t i = total_requests - 1; i >= 0; i--) {
    uint64_t obj = accesses[i].second;
    auto it = last_seen.find(obj);
    if (it != last_seen.end()) {
      reuse_distance[i] = it->second - i;
    } else {
      reuse_distance[i] = -1;  // never reused
    }
    last_seen[obj] = i;
  }
  fprintf(stderr, "  Built reuse distance map\n");

  // ==================== Pass 2: Simulation with sampling ====================
  fprintf(stderr, "Pass 2: Running simulation...\n");

  EmbeddingManager emb_mgr(42);
  emb_mgr.set_ctx_speed(DEFAULT_CTX_SPEED);
  emb_mgr.set_lr(DEFAULT_LR);
  emb_mgr.set_recent_window(8);

  TwoTierCache cache(ram_size, disk_size);
  StackDistanceTracker stack_tracker;

  // RNG for sampling
  std::mt19937_64 sample_rng(12345);
  std::uniform_real_distribution<double> sample_dist(0.0, 1.0);

  // Output CSV header
  printf("access_idx,obj_id,location,emb_score,recency,frequency,reuse_distance\n");

  int64_t sampled_count = 0;

  for (int64_t i = 0; i < total_requests; i++) {
    uint64_t obj = accesses[i].second;

    // Compute stack distance (recency)
    int64_t recency = stack_tracker.on_access(obj);

    // Update embedding and get frequency
    emb_mgr.on_access(obj);
    int frequency = emb_mgr.get_access_count(obj);

    // Get embedding score before updating recent (so obj is not in recent)
    double emb_score = emb_mgr.max_similarity_to_recent(obj);

    // Access cache and get location
    std::string location = cache.access(obj);

    // Update recent window after cache access
    emb_mgr.update_recent(obj);

    // Sample with probability sample_rate
    if (sample_dist(sample_rng) < sample_rate) {
      int64_t rd = reuse_distance[i];
      printf("%ld,%lu,%s,%.6f,%ld,%d,%ld\n",
             i, obj, location.c_str(), emb_score, recency, frequency, rd);
      sampled_count++;
    }

    if ((i + 1) % 100000 == 0) {
      fprintf(stderr, "  %ldK requests...\n", (i + 1) / 1000);
    }
  }

  // Print summary stats
  int64_t total = cache.ram_hits() + cache.disk_hits() + cache.misses();
  double ram_rate = 100.0 * cache.ram_hits() / total;
  double disk_rate = 100.0 * cache.disk_hits() / total;
  double miss_rate = 100.0 * cache.misses() / total;

  fprintf(stderr, "\n=== RESULTS ===\n");
  fprintf(stderr, "Requests: %ld\n", total);
  fprintf(stderr, "RAM hits:  %6.2f%% (%ld)\n", ram_rate, cache.ram_hits());
  fprintf(stderr, "Disk hits: %6.2f%% (%ld)\n", disk_rate, cache.disk_hits());
  fprintf(stderr, "Misses:    %6.2f%% (%ld)\n", miss_rate, cache.misses());
  fprintf(stderr, "\nSampled: %ld (%.2f%%)\n", sampled_count, 100.0 * sampled_count / total);

  return 0;
}
