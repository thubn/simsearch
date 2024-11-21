#pragma once
#include "embedding_search_base.h"
#include "simplified_float2.h"

class EmbeddingSearchFloatInt8
    : public EmbeddingSearchBase<std::vector<SimplifiedFloat>, float> {
public:
  EmbeddingSearchFloatInt8() = default;

  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<SimplifiedFloat> &query,
                    size_t k) override;

  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override {
    if (input.empty()) {
      error_message = "Input vector is empty";
      return false;
    }
    return true;
  }

  bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;

private:
  float cosine_similarity(const std::vector<SimplifiedFloat> &a,
                          const std::vector<SimplifiedFloat> &b);
};