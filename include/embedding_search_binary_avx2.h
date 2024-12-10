#pragma once
#include "aligned_types.h"         // for avx2i_vector
#include "embedding_search_base.h" // for EmbeddingSearchBase
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <utility>                 // for pair
#include <vector>                  // for vector

class EmbeddingSearchBinaryAVX2
    : public EmbeddingSearchBase<avx2i_vector, int> {
public:
  EmbeddingSearchBinaryAVX2() = default;
  std::vector<std::pair<int, size_t>>
  similarity_search(const avx2i_vector &query, size_t k) override;
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

  bool setEmbeddings(
      const std::vector<std::vector<float>> &float_embeddings) override;

private:
  int cosine_similarity(const avx2i_vector &a, const avx2i_vector &b);
};