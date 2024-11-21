#pragma once
#include "aligned_types.h"
#include "embedding_search_base.h"
#include <immintrin.h>

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
  int binary_cosine_similarity(const avx2i_vector &a, const avx2i_vector &b);
};