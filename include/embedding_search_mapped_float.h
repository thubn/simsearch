#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchMappedFloat
    : public EmbeddingSearchBase<std::vector<uint8_t>, float> {
public:
  EmbeddingSearchMappedFloat() = default;

  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<uint8_t> &query, size_t k) override;

  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override {
    if (input.empty()) {
      error_message = "Input vector is empty";
      return false;
    }
    return true;
  }

  bool setEmbeddings(
      const std::vector<std::vector<float>> &float_embeddings) override;

private:
  // static const u_int8_t MUL_RESULTS_SIZE = 64;
  float mapped_floats[256];
  // float mapped_floats_mul_result[MUL_RESULTS_SIZE][MUL_RESULTS_SIZE];
  float cosine_similarity(const std::vector<uint8_t> &a,
                          const std::vector<uint8_t> &b);
};