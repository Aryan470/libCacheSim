#include "EmbeddingManager.hpp"

#include <algorithm>
#include <cmath>

namespace embedding {

EmbeddingManager::EmbeddingManager(uint64_t seed) : rng_(seed) {
  init_context();
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
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      context_[k][d] += normal_dist_(rng_) * CTX_SPEED;
    }
    normalize(context_[k], D);
  }
}

void EmbeddingManager::init_embedding(uint64_t obj_id) {
  auto& emb = embeddings_[obj_id];
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      emb[k][d] = context_[k][d] + normal_dist_(rng_) * 0.01;
    }
    normalize(emb[k].data(), D);
  }
}

void EmbeddingManager::update_embedding(uint64_t obj_id) {
  auto& emb = embeddings_[obj_id];
  for (int k = 0; k < K; k++) {
    for (int d = 0; d < D; d++) {
      emb[k][d] = LR * context_[k][d] + (1.0 - LR) * emb[k][d];
    }
    normalize(emb[k].data(), D);
  }
}

void EmbeddingManager::on_access(uint64_t obj_id) {
  int& count = access_count_[obj_id];
  count++;

  if (count == 2) {
    init_embedding(obj_id);
  } else if (count > 2) {
    update_embedding(obj_id);
  }

  // Perturb context after every access
  perturb_context();
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

void EmbeddingManager::update_recent(uint64_t obj_id) {
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
