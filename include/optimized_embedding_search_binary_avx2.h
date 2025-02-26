#pragma once
#include "aligned_types.h"         // for avx2i_vector
// #include "avx2_popcount.h"         // for AVX2Popcount
#include "embedding_search_base.h" // for OptimizedEmbeddingSearchBase
#include <arm_neon.h>              // for __m256i
#include <stddef.h>                // for size_t
#include <stdint.h>                // for int32_t
#include <string>                  // for string
#include <utility>                 // for pair
#include <vector>                  // for vector
#include <omp.h>                   // for OpenMP support

class OptimizedEmbeddingSearchBinaryAVX2
    : public OptimizedEmbeddingSearchBase<avx2i_vector, int32_t, uint32x4_t> {
public:
  OptimizedEmbeddingSearchBinaryAVX2() = default;

  bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;
  std::vector<std::pair<int32_t, size_t>>
  similarity_search(const avx2i_vector &query, size_t k, bool use_multithreading);
  avx2i_vector getEmbeddingAVX2(size_t index) const;

  std::vector<std::pair<int32_t, size_t>>
  similarity_search(const avx2i_vector &query, size_t k) override {
    return similarity_search(query, k, true); // Default to using multithreading
  }

protected:
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

private:
  // AVX2Popcount counter;
  int32_t cosine_similarity_optimized(const uint32x4_t *vec_a,
                                      const uint32x4_t *vec_b) const override;
  int32_t cosine_similarity_optimized_dynamic(const uint32x4_t *vec_a,
                                              const uint32x4_t *vec_b) const;
  void convert_float_to_binary_avx2(const std::vector<float> &input,
                                    uint32x4_t *output) const;
};