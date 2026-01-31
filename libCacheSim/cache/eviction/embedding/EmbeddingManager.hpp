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
static constexpr double DEFAULT_LR = 0.2;         // learning rate
static constexpr double DEFAULT_CTX_SPEED = 0.1;  // context perturbation speed

class EmbeddingManager {
 public:
  EmbeddingManager(uint64_t seed = 42);
  ~EmbeddingManager();

  // Update context and embedding when an object is accessed
  void on_access(uint64_t obj_id);

  // Compute maximum similarity between obj and any object in the recent window
  double max_similarity_to_recent(uint64_t obj_id);

  // Compute similarity between two specific objects (public for diagnostics)
  double similarity_public(uint64_t a, uint64_t b) { return similarity(a, b); }

  // Compute average of top-k similarities to recent objects
  double avg_top_k_similarity_to_recent(uint64_t obj_id, int k);

  // Compute average similarity to ALL recent objects
  double avg_similarity_to_recent(uint64_t obj_id);

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

  // Get/set learning rate
  double get_lr() const { return lr_; }
  void set_lr(double lr) { lr_ = lr; }

  // Get/set context speed
  double get_ctx_speed() const { return ctx_speed_; }
  void set_ctx_speed(double speed) { ctx_speed_ = speed; }

  // Memory usage estimate
  size_t memory_bytes() const;

  // Get embedding for an object (returns nullptr if not found)
  const std::array<std::array<double, D>, K>* get_embedding(uint64_t obj_id) const {
    auto it = embeddings_.find(obj_id);
    return it != embeddings_.end() ? &it->second : nullptr;
  }

 private:
  using EmbeddingArray = std::array<std::array<double, D>, K>;

  void init_context();
  void perturb_context();
  void perturb_context_10x();
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
  int perturb_counter_ = 0;  // for batched perturbation
  double lr_ = DEFAULT_LR;
  double ctx_speed_ = DEFAULT_CTX_SPEED;

  // Random number generator
  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

}  // namespace embedding
