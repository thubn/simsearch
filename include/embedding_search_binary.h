#pragma once
#include "embedding_search_base.h" // for EmbeddingSearchBase
#include <cstdint>                 // for uint64_t
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <utility>                 // for pair
#include <vector>                  // for vector

class EmbeddingSearchBinary
    : public EmbeddingSearchBase<std::vector<uint64_t>, int> {
public:
  EmbeddingSearchBinary() = default;

  std::vector<std::pair<int, size_t>>
  similarity_search(const std::vector<uint64_t> &query, size_t k) override;
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

  bool setEmbeddings(
      const std::vector<std::vector<float>> &float_embeddings) override;

private:
  int cosine_similarity(const std::vector<uint64_t> &a,
                        const std::vector<uint64_t> &b);
};