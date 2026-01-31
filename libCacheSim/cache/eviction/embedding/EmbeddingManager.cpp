#include "EmbeddingManager.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>

namespace embedding {

// Timing instrumentation (single-threaded)
static int64_t t_perturb_context_ns = 0;
static int64_t t_update_embedding_ns = 0;
static int64_t t_init_embedding_ns = 0;
static int64_t t_update_recent_ns = 0;
static int64_t t_avg_top_k_sim_ns = 0;
static int64_t t_on_access_ns = 0;
static int64_t t_get_access_count_ns = 0;
static int64_t n_on_access_calls = 0;
static int64_t n_avg_top_k_calls = 0;
static int64_t n_perturb_calls = 0;
static int64_t n_update_emb_calls = 0;
static bool timing_printed = false;

// Sequential similarity diagnostics
static double sum_sequential_sim = 0.0;
static int64_t n_sequential_pairs = 0;
static double sum_random_sim = 0.0;
static int64_t n_random_pairs = 0;
static uint64_t last_obj_id = 0;
static bool last_had_embedding = false;
static std::vector<uint64_t> sampled_obj_ids;  // for random pair comparison
static std::mt19937_64 diag_rng(12345);

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void print_embedding_timing() {
  if (timing_printed) return;
  timing_printed = true;

  auto to_ms = [](int64_t ns) { return ns / 1000000.0; };
  auto per_call_us = [](int64_t ns, int64_t calls) {
    return calls > 0 ? (ns / 1000.0) / calls : 0.0;
  };

  fprintf(stderr, "\n=== EmbeddingManager Timing Breakdown ===\n");
  fprintf(stderr, "on_access total:       %10.2f ms  (%ld calls, %.3f us/call)\n",
          to_ms(t_on_access_ns), n_on_access_calls, per_call_us(t_on_access_ns, n_on_access_calls));
  fprintf(stderr, "  perturb_context:     %10.2f ms  (%ld calls, %.3f us/call)\n",
          to_ms(t_perturb_context_ns), n_perturb_calls, per_call_us(t_perturb_context_ns, n_perturb_calls));
  fprintf(stderr, "  update_embedding:    %10.2f ms  (%ld calls, %.3f us/call)\n",
          to_ms(t_update_embedding_ns), n_update_emb_calls, per_call_us(t_update_embedding_ns, n_update_emb_calls));
  fprintf(stderr, "  init_embedding:      %10.2f ms\n", to_ms(t_init_embedding_ns));
  fprintf(stderr, "  update_recent:       %10.2f ms\n", to_ms(t_update_recent_ns));
  fprintf(stderr, "avg_top_k_similarity:  %10.2f ms  (%ld calls, %.3f us/call)\n",
          to_ms(t_avg_top_k_sim_ns), n_avg_top_k_calls, per_call_us(t_avg_top_k_sim_ns, n_avg_top_k_calls));
  fprintf(stderr, "get_access_count:      %10.2f ms\n", to_ms(t_get_access_count_ns));
  fprintf(stderr, "=========================================\n\n");

  // Sequential vs random similarity diagnostics
  fprintf(stderr, "=== Sequential vs Random Similarity ===\n");
  double avg_seq = n_sequential_pairs > 0 ? sum_sequential_sim / n_sequential_pairs : 0.0;
  double avg_rand = n_random_pairs > 0 ? sum_random_sim / n_random_pairs : 0.0;
  fprintf(stderr, "Consecutive pairs:  avg_sim = %.4f  (n = %ld)\n", avg_seq, n_sequential_pairs);
  fprintf(stderr, "Random pairs:       avg_sim = %.4f  (n = %ld)\n", avg_rand, n_random_pairs);
  fprintf(stderr, "Difference:         %.4f\n", avg_seq - avg_rand);
  fprintf(stderr, "========================================\n\n");
}

EmbeddingManager::EmbeddingManager(uint64_t seed) : rng_(seed) {
  init_context();
}

EmbeddingManager::~EmbeddingManager() {
  print_embedding_timing();
}

void EmbeddingManager::init_context() {
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      context_[k][d] = normal_dist_(rng_);
    }
    normalize(context_[k], D);
  }
}

void EmbeddingManager::perturb_context() {
  int64_t start = now_ns();
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      context_[k][d] += normal_dist_(rng_) * ctx_speed_;
    }
    normalize(context_[k], D);
  }
  t_perturb_context_ns += now_ns() - start;
  n_perturb_calls++;
}

void EmbeddingManager::perturb_context_10x() {
  int64_t start = now_ns();
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      context_[k][d] += normal_dist_(rng_) * ctx_speed_ * std::sqrt(10.0);
    }
    normalize(context_[k], D);
  }
  t_perturb_context_ns += now_ns() - start;
  n_perturb_calls++;
}

void EmbeddingManager::init_embedding(uint64_t obj_id) {
  int64_t start = now_ns();
  auto& emb = embeddings_[obj_id];
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      emb[k][d] = context_[k][d] + normal_dist_(rng_) * 0.01;
    }
    normalize(emb[k].data(), D);
  }
  t_init_embedding_ns += now_ns() - start;
}

void EmbeddingManager::update_embedding(uint64_t obj_id) {
  int64_t start = now_ns();
  auto& emb = embeddings_[obj_id];
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      emb[k][d] = lr_ * context_[k][d] + (1.0 - lr_) * emb[k][d];
    }
    normalize(emb[k].data(), D);
  }
  t_update_embedding_ns += now_ns() - start;
  n_update_emb_calls++;
}

void EmbeddingManager::on_access(uint64_t obj_id) {
  int64_t start = now_ns();

  // Perturb context every 10 accesses with 10x larger jump
  if (++perturb_counter_ >= 10) {
    perturb_counter_ = 0;
    perturb_context_10x();
  }

  int& count = access_count_[obj_id];
  count++;

  // If object already has embedding, update it
  // If object just reached freq 2, give it an embedding
  // Otherwise just track freq
  auto it = embeddings_.find(obj_id);
  bool has_emb = (it != embeddings_.end());

  if (has_emb) {
    update_embedding(obj_id);
    // Track in recent window (has valid embedding)
    update_recent(obj_id);
  } else if (count == 2) {
    init_embedding(obj_id);
    has_emb = true;
    // Track in recent window (just got embedding)
    update_recent(obj_id);
  }

  // Diagnostic: measure similarity between consecutive accesses
  if (last_had_embedding && has_emb && last_obj_id != obj_id) {
    double sim = similarity(last_obj_id, obj_id);
    sum_sequential_sim += sim;
    n_sequential_pairs++;
  }

  // Diagnostic: sample objects for random pair comparison (1 in 100)
  if (has_emb && (n_on_access_calls % 100 == 0)) {
    sampled_obj_ids.push_back(obj_id);
    // Compare with a random previously sampled object
    if (sampled_obj_ids.size() > 20) {
      std::uniform_int_distribution<size_t> dist(0, sampled_obj_ids.size() - 2);
      size_t idx = dist(diag_rng);
      uint64_t other = sampled_obj_ids[idx];
      if (other != obj_id && embeddings_.find(other) != embeddings_.end()) {
        double sim = similarity(other, obj_id);
        sum_random_sim += sim;
        n_random_pairs++;
      }
    }
  }

  last_obj_id = obj_id;
  last_had_embedding = has_emb;

  t_on_access_ns += now_ns() - start;
  n_on_access_calls++;

  // Progress output every 10k accesses
  if (n_on_access_calls % 10000 == 0) {
    fprintf(stderr, "[EmbeddingManager] %ld accesses, %ld embeddings, %ld seq_pairs (avg_sim=%.3f)\n",
            n_on_access_calls, (int64_t)embeddings_.size(), n_sequential_pairs,
            n_sequential_pairs > 0 ? sum_sequential_sim / n_sequential_pairs : 0.0);
  }
}

double EmbeddingManager::similarity(uint64_t a, uint64_t b) {
  auto it_a = embeddings_.find(a);
  auto it_b = embeddings_.find(b);

  // If either doesn't have an embedding yet, return 0
  if (it_a == embeddings_.end() || it_b == embeddings_.end()) {
    return 0.0;
  }

  const auto& emb_a = it_a->second;
  const auto& emb_b = it_b->second;

  double sum = 0;
  for (int k = 0; k < K; k++) {
    sum += dot_product(emb_a[k].data(), emb_b[k].data(), D);
  }
  return sum / K;  // Mean of k dot products
}

double EmbeddingManager::max_similarity_to_recent(uint64_t obj_id) {
  auto it = embeddings_.find(obj_id);

  double max_sim = -1.0;

  for (int i = 0; i < recent_count_; i++) {
    if (recent_ids_[i] == obj_id) continue;

    // No embedding for either: similarity = 0
    if (it == embeddings_.end() || !recent_valid_[i]) {
      if (max_sim < 0.0) max_sim = 0.0;
      continue;
    }

    const auto& emb = it->second;
    const auto& recent_emb = recent_embeddings_[i];
    double sum = 0;
    for (int k = 0; k < K; k++) {
      sum += dot_product(emb[k].data(), recent_emb[k].data(), D);
    }
    double sim = sum / K;
    if (sim > max_sim) max_sim = sim;
  }

  return max_sim;
}

double EmbeddingManager::avg_top_k_similarity_to_recent(uint64_t obj_id, int k) {
  int64_t start = now_ns();

  auto it = embeddings_.find(obj_id);
  if (it == embeddings_.end()) {
    t_avg_top_k_sim_ns += now_ns() - start;
    n_avg_top_k_calls++;
    return 0.0;  // No embedding for this object
  }

  // Collect all similarities
  std::vector<double> sims;
  sims.reserve(recent_count_);

  for (int i = 0; i < recent_count_; i++) {
    if (recent_ids_[i] == obj_id) continue;
    if (!recent_valid_[i]) continue;

    const auto& emb = it->second;
    const auto& recent_emb = recent_embeddings_[i];
    double sum = 0;
    for (int j = 0; j < K; j++) {
      sum += dot_product(emb[j].data(), recent_emb[j].data(), D);
    }
    sims.push_back(sum / K);
  }

  if (sims.empty()) {
    t_avg_top_k_sim_ns += now_ns() - start;
    n_avg_top_k_calls++;
    return 0.0;
  }

  // Sort descending and take average of top k
  std::sort(sims.begin(), sims.end(), std::greater<double>());
  int count = std::min(k, static_cast<int>(sims.size()));
  double sum = 0;
  for (int i = 0; i < count; i++) {
    sum += sims[i];
  }

  t_avg_top_k_sim_ns += now_ns() - start;
  n_avg_top_k_calls++;
  return sum / count;
}

double EmbeddingManager::avg_similarity_to_recent(uint64_t obj_id) {
  auto it = embeddings_.find(obj_id);
  if (it == embeddings_.end()) {
    return 0.0;  // No embedding for this object
  }

  double total_sim = 0.0;
  int count = 0;

  for (int i = 0; i < recent_count_; i++) {
    if (recent_ids_[i] == obj_id) continue;
    if (!recent_valid_[i]) continue;

    const auto& emb = it->second;
    const auto& recent_emb = recent_embeddings_[i];
    double sum = 0;
    for (int j = 0; j < K; j++) {
      sum += dot_product(emb[j].data(), recent_emb[j].data(), D);
    }
    total_sim += sum / K;
    count++;
  }

  return count > 0 ? total_sim / count : 0.0;
}

void EmbeddingManager::update_recent(uint64_t obj_id) {
  int64_t start = now_ns();

  // Ensure vectors are sized
  if (recent_embeddings_.size() < static_cast<size_t>(recent_window_)) {
    recent_embeddings_.resize(recent_window_);
    recent_ids_.resize(recent_window_);
    recent_valid_.resize(recent_window_, false);
  }

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

  t_update_recent_ns += now_ns() - start;
}

bool EmbeddingManager::has_embedding(uint64_t obj_id) const {
  return embeddings_.find(obj_id) != embeddings_.end();
}

size_t EmbeddingManager::memory_bytes() const {
  // embeddings: each entry is K*D doubles
  size_t emb_bytes = embeddings_.size() * K * D * sizeof(double);
  // access_count: each entry is ~40 bytes (key + value + hash table overhead)
  size_t count_bytes = access_count_.size() * 40;
  // context: fixed K*D doubles
  size_t ctx_bytes = K * D * sizeof(double);
  return emb_bytes + count_bytes + ctx_bytes;
}

void EmbeddingManager::normalize(double* vec, int dim) {
  double norm = 0;
  for (int i = 0; i < dim; i++) {
    norm += vec[i] * vec[i];
  }
  norm = std::sqrt(norm);
  if (norm > 1e-10) {
    for (int i = 0; i < dim; i++) {
      vec[i] /= norm;
    }
  }
}

double EmbeddingManager::dot_product(const double* a, const double* b,
                                     int dim) {
  double sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace embedding
