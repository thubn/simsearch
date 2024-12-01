#pragma once
#include "aligned_types.h"
#include "embedding_search_base.h"
#include <immintrin.h>

class OptimizedEmbeddingSearchAVX2
    : public OptimizedEmbeddingSearchBase<avx2_vector, float, float> {
public:
  // Constructor can stay in header since it's simple
  OptimizedEmbeddingSearchAVX2() = default;

  // Virtual function declarations
  bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;
  std::vector<std::pair<float, size_t>>
  similarity_search(const avx2_vector &query, size_t k) override;
  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<float> &query, size_t k);
  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<float> &query, size_t k,
                    std::vector<std::pair<int, size_t>> &searchIndexes);
  std::vector<float> getEmbedding(size_t index) const;

protected:
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

private:
  // std::vector<float> norms;
  // Private method declarations
  float compute_norm_avx2(const float *vec) const;
  float cosine_similarity_optimized(const float *vec_a, const float *vec_b) const override;
  void cosine_similarity_optimized(const int j, const float *vec_a, float *sim);
};