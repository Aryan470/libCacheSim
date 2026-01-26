#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>

namespace embedding {

// Embedding system constants
static constexpr int K = 8;               // num context vectors
static constexpr int D = 8;               // dimensions per vector
static constexpr double LR = 0.2;         // learning rate
static constexpr double CTX_SPEED = 0.1;  // context perturbation speed

class EmbeddingManager {
 public:
  EmbeddingManager(uint64_t seed = 42);
  ~EmbeddingManager() = default;

  // Update context and embedding when an object is accessed
  void on_access(uint64_t obj_id);

  // Compute maximum similarity between obj and any object in the recent window
  double max_similarity_to_recent(uint64_t obj_id);

  // Track recent accesses (call after insert)
  void update_recent(uint64_t obj_id);

  // Check if an object has an embedding
  bool has_embedding(uint64_t obj_id) const;

  // Get access count for an object
  int get_access_count(uint64_t obj_id) const {
    auto it = access_count_.find(obj_id);
    return it != access_count_.end() ? it->second : 0;
  }

  // Get/set recent window size
  int get_recent_window() const { return recent_window_; }
  void set_recent_window(int size) { recent_window_ = size; }

  // Memory usage estimate
  size_t memory_bytes() const;

 private:
  using EmbeddingArray = std::array<std::array<double, D>, K>;

  void init_context();
  void perturb_context();
  void init_embedding(uint64_t obj_id);
  void update_embedding(uint64_t obj_id);
  double similarity(uint64_t a, uint64_t b);

  static void normalize(double* vec, int dim);
  static double dot_product(const double* a, const double* b, int dim);

  // Context vectors (K vectors of D dimensions each)
  double context_[K][D];

  // Per-object embeddings
  std::unordered_map<uint64_t, EmbeddingArray> embeddings_;

  // Access count per object (embedding initialized on 2nd access)
  std::unordered_map<uint64_t, int> access_count_;

  // Recent embeddings stored contiguously for cache efficiency
  std::vector<EmbeddingArray> recent_embeddings_;
  std::vector<uint64_t> recent_ids_;
  std::vector<bool> recent_valid_;  // true if embedding exists
  int recent_idx_ = 0;  // circular buffer index
  int recent_count_ = 0;
  int recent_window_ = 16;

  // Random number generator
  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

}  // namespace embedding
