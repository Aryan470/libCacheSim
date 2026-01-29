// Two-Tier Cache with Embedding-Based Promotion
// Tests whether embedding similarity can improve multi-tier cache hit rates.
// When object A is accessed, check SSD for objects with high similarity to recent context
// and promote them to RAM proactively.
//
// Compile: g++ -O3 -std=c++17 -o two_tier_emb_sim two_tier_emb_sim.cpp
// Usage: ./two_tier_emb_sim <trace.csv> [max_req=1000000] [ram_size=1000] [ssd_size=100000] [sim_thresh=0.3] [ctx_speed=0.02]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ==================== Embedding System ====================

static constexpr int K = 8;   // num context vectors
static constexpr int D = 8;   // dimensions per vector
static constexpr double DEFAULT_LR = 0.2;
static constexpr double DEFAULT_CTX_SPEED = 0.02;  // slower context = better signal

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
      const auto& recent_emb = recent_embeddings_[i];

      // Max over K vector pairs
      double pair_max = -2.0;
      for (int k = 0; k < K; k++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += emb[k][d] * recent_emb[k][d];
        }
        pair_max = std::max(pair_max, dot);
      }
      max_sim = std::max(max_sim, pair_max);
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
  int recent_window_ = 16;

  double lr_ = DEFAULT_LR;
  double ctx_speed_ = DEFAULT_CTX_SPEED;

  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

// ==================== LRU Tier ====================

class LRUTier {
 public:
  LRUTier(int capacity) : capacity_(capacity) {}

  bool contains(uint64_t obj) const {
    return map_.count(obj) > 0;
  }

  bool access(uint64_t obj) {
    auto it = map_.find(obj);
    if (it != map_.end()) {
      lru_.erase(it->second);
      lru_.push_front(obj);
      map_[obj] = lru_.begin();
      return true;
    }
    return false;
  }

  uint64_t insert(uint64_t obj) {
    uint64_t evicted = 0;
    if ((int)lru_.size() >= capacity_) {
      evicted = lru_.back();
      map_.erase(evicted);
      lru_.pop_back();
    }
    lru_.push_front(obj);
    map_[obj] = lru_.begin();
    return evicted;
  }

  void remove(uint64_t obj) {
    auto it = map_.find(obj);
    if (it != map_.end()) {
      lru_.erase(it->second);
      map_.erase(it);
    }
  }

  int size() const { return (int)lru_.size(); }
  int capacity() const { return capacity_; }

  // Get position of object in LRU (0 = MRU, size-1 = LRU)
  // Returns -1 if not found
  int get_position(uint64_t obj) const {
    auto it = map_.find(obj);
    if (it == map_.end()) return -1;
    int pos = 0;
    for (auto lit = lru_.begin(); lit != lru_.end(); ++lit, ++pos) {
      if (lit == it->second) return pos;
    }
    return -1;
  }

  // Get all objects in cache (for scanning SSD)
  const std::list<uint64_t>& objects() const { return lru_; }

 private:
  int capacity_;
  std::list<uint64_t> lru_;
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> map_;
};

// ==================== Two-Tier Cache with Embedding Promotion ====================

class TwoTierEmbCache {
 public:
  TwoTierEmbCache(int ram_capacity, int ssd_capacity, double sim_threshold,
                  int max_promote_per_access, EmbeddingManager* emb_mgr)
      : ram_(ram_capacity),
        ssd_(ssd_capacity),
        sim_threshold_(sim_threshold),
        max_promote_(max_promote_per_access),
        emb_mgr_(emb_mgr) {}

  void access(uint64_t obj) {
    // Check RAM first - track position before moving to MRU
    int ram_pos = ram_.get_position(obj);
    if (ram_pos >= 0) {
      ram_hits_++;
      track_ram_position(ram_pos);
      ram_.access(obj);  // move to MRU
      return;
    }

    // Check SSD - track position before promoting
    int ssd_pos = ssd_.get_position(obj);
    if (ssd_pos >= 0) {
      ssd_hits_++;
      track_ssd_position(ssd_pos);
      ssd_.remove(obj);
      promote_to_ram(obj);
      // No proactive promotions for diagnostic mode
      if (enable_promotion_) check_and_promote_high_sim();
      return;
    }

    // Miss
    misses_++;
    insert_to_ssd(obj);
    // No proactive promotions for diagnostic mode
    if (enable_promotion_) check_and_promote_high_sim();
  }

  void set_enable_promotion(bool enable) { enable_promotion_ = enable; }

  void print_position_stats() const {
    if (ram_hits_ == 0 && ssd_hits_ == 0) return;

    printf("\n--- ACCESS POSITION DISTRIBUTION (deciles) ---\n");

    if (ram_hits_ > 0) {
      printf("RAM hits by position (total: %ld):\n", ram_hits_);
      for (int i = 0; i < 10; i++) {
        printf("  D%d (%d-%d%%):  %5.1f%%\n", i+1, i*10, (i+1)*10, 100.0 * ram_pos_[i] / ram_hits_);
      }
    }

    if (ssd_hits_ > 0) {
      printf("SSD hits by position (total: %ld):\n", ssd_hits_);
      for (int i = 0; i < 10; i++) {
        printf("  D%d (%d-%d%%):  %5.1f%%\n", i+1, i*10, (i+1)*10, 100.0 * ssd_pos_[i] / ssd_hits_);
      }
    }
  }

 private:
  void track_ram_position(int pos) {
    int size = ram_.size();
    if (size == 0) return;
    int decile = std::min(9, (pos * 10) / size);
    ram_pos_[decile]++;
  }

  void track_ssd_position(int pos) {
    int size = ssd_.size();
    if (size == 0) return;
    int decile = std::min(9, (pos * 10) / size);
    ssd_pos_[decile]++;
  }

 public:

  // Stats
  int64_t ram_hits() const { return ram_hits_; }
  int64_t ssd_hits() const { return ssd_hits_; }
  int64_t misses() const { return misses_; }
  int64_t promotions() const { return promotions_; }
  int64_t promotions_useful() const { return promotions_useful_; }

 private:
  void promote_to_ram(uint64_t obj) {
    uint64_t evicted = ram_.insert(obj);
    if (evicted != 0) {
      // Evicted from RAM goes to SSD
      ssd_.insert(evicted);
    }
  }

  void insert_to_ssd(uint64_t obj) {
    // Insert directly to SSD (for first access / miss)
    ssd_.insert(obj);
  }

  void check_and_promote_high_sim() {
    if (max_promote_ <= 0) return;

    const auto& ssd_objects = ssd_.objects();
    if (ssd_objects.empty()) return;

    // Sample from SSD HEAD (recently evicted from RAM)
    // Use very high threshold to be selective
    double best_sim = -2.0;
    uint64_t best_obj = 0;

    int sample_count = 0;
    const int MAX_SAMPLE = 100;

    for (auto it = ssd_objects.begin();
         it != ssd_objects.end() && sample_count < MAX_SAMPLE;
         ++it, ++sample_count) {
      uint64_t obj = *it;
      double sim = emb_mgr_->max_similarity_to_recent(obj);
      if (sim > best_sim) {
        best_sim = sim;
        best_obj = obj;
      }
    }

    // Track similarity distribution
    sim_samples_++;
    sim_sum_ += best_sim;
    if (best_sim > 0.9) sim_above_90_++;
    else if (best_sim > 0.8) sim_above_80_++;
    else if (best_sim > 0.7) sim_above_70_++;
    else if (best_sim > 0.5) sim_above_50_++;

    // Only promote if the best object exceeds threshold
    if (best_sim >= sim_threshold_ && best_obj != 0) {
      ssd_.remove(best_obj);
      promote_to_ram(best_obj);
      promotions_++;
      promoted_objs_.insert(best_obj);
    }
  }

 public:
  void print_sim_stats() const {
    if (sim_samples_ == 0) return;
    fprintf(stderr, "\nSimilarity distribution (best of 100 SSD head samples):\n");
    fprintf(stderr, "  Mean: %.3f\n", sim_sum_ / sim_samples_);
    fprintf(stderr, "  >0.9: %.1f%%\n", 100.0 * sim_above_90_ / sim_samples_);
    fprintf(stderr, "  0.8-0.9: %.1f%%\n", 100.0 * sim_above_80_ / sim_samples_);
    fprintf(stderr, "  0.7-0.8: %.1f%%\n", 100.0 * sim_above_70_ / sim_samples_);
    fprintf(stderr, "  0.5-0.7: %.1f%%\n", 100.0 * sim_above_50_ / sim_samples_);
    fprintf(stderr, "  <0.5: %.1f%%\n", 100.0 * (sim_samples_ - sim_above_90_ - sim_above_80_ - sim_above_70_ - sim_above_50_) / sim_samples_);
  }

 private:
  // Similarity stats
  int64_t sim_samples_ = 0;
  double sim_sum_ = 0;
  int64_t sim_above_90_ = 0;
  int64_t sim_above_80_ = 0;
  int64_t sim_above_70_ = 0;
  int64_t sim_above_50_ = 0;

 public:
  // Track if promoted objects are accessed
  void track_promoted_access(uint64_t obj) {
    if (promoted_objs_.count(obj)) {
      promotions_useful_++;
      promoted_objs_.erase(obj);
    }
  }

 private:
  LRUTier ram_;
  LRUTier ssd_;
  double sim_threshold_;
  int max_promote_;
  EmbeddingManager* emb_mgr_;

  int64_t ram_hits_ = 0;
  int64_t ssd_hits_ = 0;
  int64_t misses_ = 0;
  int64_t promotions_ = 0;
  int64_t promotions_useful_ = 0;
  int64_t promote_counter_ = 0;
  bool enable_promotion_ = true;

  // Position tracking (deciles)
  int64_t ram_pos_[10] = {0};
  int64_t ssd_pos_[10] = {0};

  std::unordered_set<uint64_t> promoted_objs_;
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
            "Usage: %s <trace.csv> [max_req] [ram_size] [ssd_size] "
            "[sim_thresh] [ctx_speed] [max_promote] [key_type]\n",
            argv[0]);
    fprintf(stderr, "  max_req: max requests (default: 1000000)\n");
    fprintf(stderr, "  ram_size: RAM tier capacity (default: 1000)\n");
    fprintf(stderr, "  ssd_size: SSD tier capacity (default: 100000)\n");
    fprintf(stderr, "  sim_thresh: similarity threshold for promotion (default: 0.3)\n");
    fprintf(stderr, "  ctx_speed: context perturbation speed (default: 0.02)\n");
    fprintf(stderr, "  max_promote: max proactive promotions per access (default: 3)\n");
    fprintf(stderr, "  key_type: 0=numeric, 1=hash (default: 0)\n");
    return 1;
  }

  const char* path = argv[1];
  int64_t max_req = (argc > 2) ? atoll(argv[2]) : 1000000;
  int ram_size = (argc > 3) ? atoi(argv[3]) : 1000;
  int ssd_size = (argc > 4) ? atoi(argv[4]) : 100000;
  double sim_threshold = (argc > 5) ? atof(argv[5]) : 0.3;
  double ctx_speed = (argc > 6) ? atof(argv[6]) : 0.02;
  int max_promote = (argc > 7) ? atoi(argv[7]) : 3;
  if (argc > 8) g_key_type = atoi(argv[8]);

  fprintf(stderr, "=== TWO-TIER EMBEDDING PROMOTION SIMULATION ===\n");
  fprintf(stderr, "Trace: %s\n", path);
  fprintf(stderr, "Config: RAM=%d, SSD=%d, sim_thresh=%.2f, ctx_speed=%.3f, max_promote=%d\n\n",
          ram_size, ssd_size, sim_threshold, ctx_speed, max_promote);

  // Create embedding managers (separate for each cache to be fair)
  double lr = 0.1;  // learning rate for embedding updates
  EmbeddingManager emb_with_promote(42);
  emb_with_promote.set_ctx_speed(ctx_speed);
  emb_with_promote.set_lr(lr);
  emb_with_promote.set_recent_window(16);

  EmbeddingManager emb_baseline(42);
  emb_baseline.set_ctx_speed(ctx_speed);
  emb_baseline.set_lr(lr);
  emb_baseline.set_recent_window(16);

  // Create caches
  TwoTierEmbCache with_promote(ram_size, ssd_size, sim_threshold, max_promote, &emb_with_promote);
  TwoTierEmbCache baseline(ram_size, ssd_size, sim_threshold, 0, &emb_baseline);  // no promotion

  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    return 1;
  }

  char line[4096];
  fgets(line, sizeof(line), f);  // skip header

  int64_t total = 0;
  while (fgets(line, sizeof(line), f) && total < max_req) {
    bool valid;
    uint64_t obj = parse_key(line, &valid);
    if (!valid) continue;
    total++;

    // Update embeddings for both
    emb_with_promote.on_access(obj);
    emb_baseline.on_access(obj);

    // Track promoted object access before cache access
    with_promote.track_promoted_access(obj);

    // Access caches
    with_promote.access(obj);
    baseline.access(obj);

    // Update recent embeddings
    emb_with_promote.update_recent(obj);
    emb_baseline.update_recent(obj);

    if (total % 100000 == 0) {
      fprintf(stderr, "  %ldK requests...\n", total / 1000);
    }
  }
  fclose(f);

  // Results
  double pf_ram_rate = 100.0 * with_promote.ram_hits() / total;
  double pf_ssd_rate = 100.0 * with_promote.ssd_hits() / total;
  double pf_miss_rate = 100.0 * with_promote.misses() / total;

  double bl_ram_rate = 100.0 * baseline.ram_hits() / total;
  double bl_ssd_rate = 100.0 * baseline.ssd_hits() / total;
  double bl_miss_rate = 100.0 * baseline.misses() / total;

  printf("\n");
  printf("========================================\n");
  printf("   TWO-TIER EMBEDDING PROMOTION RESULTS\n");
  printf("========================================\n\n");
  printf("Config: RAM=%d, SSD=%d, sim_thresh=%.2f, ctx_speed=%.3f, max_promote=%d\n",
         ram_size, ssd_size, sim_threshold, ctx_speed, max_promote);
  printf("Requests: %ld\n\n", total);

  printf("--- WITH EMBEDDING PROMOTION ---\n");
  printf("RAM hits:  %6.2f%% (%ld)\n", pf_ram_rate, with_promote.ram_hits());
  printf("SSD hits:  %6.2f%% (%ld)\n", pf_ssd_rate, with_promote.ssd_hits());
  printf("Misses:    %6.2f%% (%ld)\n", pf_miss_rate, with_promote.misses());
  printf("Proactive promotions: %ld\n", with_promote.promotions());
  if (with_promote.promotions() > 0) {
    printf("  -> Useful (later accessed): %ld (%.1f%%)\n",
           with_promote.promotions_useful(),
           100.0 * with_promote.promotions_useful() / with_promote.promotions());
  }
  printf("\n");

  printf("--- BASELINE (no embedding promotion) ---\n");
  printf("RAM hits:  %6.2f%% (%ld)\n", bl_ram_rate, baseline.ram_hits());
  printf("SSD hits:  %6.2f%% (%ld)\n", bl_ssd_rate, baseline.ssd_hits());
  printf("Misses:    %6.2f%% (%ld)\n\n", bl_miss_rate, baseline.misses());

  printf("--- IMPROVEMENT ---\n");
  printf("RAM hit rate: %+.2f pp\n", pf_ram_rate - bl_ram_rate);
  printf("SSD hit rate: %+.2f pp\n", pf_ssd_rate - bl_ssd_rate);
  printf("Miss rate:    %+.2f pp\n", pf_miss_rate - bl_miss_rate);

  // Print similarity distribution
  with_promote.print_sim_stats();

  // Print position stats (from baseline which has no promotions)
  baseline.print_position_stats();

  return 0;
}
