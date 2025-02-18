#pragma once
#include "aligned_types.h"         // for avx2i_vector
#include "embedding_search_base.h" // for OptimizedEmbeddingSearchBase
#include <arm_neon.h>              // for __m256i
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <utility>                 // for pair
#include <vector>                  // for vector

class OptimizedEmbeddingSearchUint8AVX2
    : public OptimizedEmbeddingSearchBase<avx2i_vector8, int, int8x16_t> {
public:
  OptimizedEmbeddingSearchUint8AVX2() = default;

  bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;
  std::vector<std::pair<int, size_t>>
  similarity_search(const avx2i_vector8 &query, size_t k) override;
  avx2i_vector8 getEmbeddingAVX2(size_t index) const;

protected:
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

private:
  int cosine_similarity_optimized(const int8x16_t *vec_a,
                                  const int8x16_t *vec_b) const override;
  void convert_float_to_uint8_avx2(const std::vector<float> &input,
                                   int8x16_t *output) const;
};