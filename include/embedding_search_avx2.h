#pragma once
#include "aligned_types.h"         // for avx2_vector
#include "embedding_search_base.h" // for EmbeddingSearchBase
#include <immintrin.h>             // for __m256
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <utility>                 // for pair
#include <vector>                  // for vector

class EmbeddingSearchAVX2 : public EmbeddingSearchBase<avx2_vector, float> {
public:
  EmbeddingSearchAVX2() = default;

  std::vector<std::pair<float, size_t>>
  similarity_search(const avx2_vector &query, size_t k) override;
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

  std::vector<std::pair<float, size_t>>
  similarity_search(const avx2_vector &query, size_t k,
                    std::vector<std::pair<int, size_t>> &searchIndexes);

  bool setEmbeddings(const std::vector<std::vector<float>> &m) override;

private:
  float cosine_similarity(const avx2_vector &a, const avx2_vector &b);
  float dot_product_avx2(__m256 a, __m256 b);
};