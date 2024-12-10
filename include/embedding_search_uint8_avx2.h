#pragma once
#include "aligned_types.h"         // for avx2i_vector
#include "embedding_search_base.h" // for EmbeddingSearchBase
#include <immintrin.h>             // for __m256i
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <sys/types.h>             // for uint
#include <utility>                 // for pair
#include <vector>                  // for vector

class EmbeddingSearchUint8AVX2
    : public EmbeddingSearchBase<avx2i_vector, uint> {
public:
  EmbeddingSearchUint8AVX2() = default;

  // Override virtual functions from base class
  std::vector<std::pair<uint, size_t>>
  similarity_search(const avx2i_vector &query, size_t k) override;
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

  // Additional functions
  std::vector<std::pair<uint, size_t>>
  similarity_search(const avx2i_vector &query, size_t k,
                    std::vector<std::pair<int, size_t>> &searchIndexes);
  bool setEmbeddings(const std::vector<std::vector<float>> &m) override;

private:
  uint cosine_similarity(const avx2i_vector &a, const avx2i_vector &b);
  float dot_product_avx2(__m256i a, __m256i b);
};