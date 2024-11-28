#pragma once
#include "embedding_search_base.h"
#include <stdexcept>

class EmbeddingSearchMappedFloat
    : public EmbeddingSearchBase<std::vector<uint8_t>, float> {
public:
  EmbeddingSearchMappedFloat() = default;

  struct PartitionInfo {
    float start;
    float end;
    float average;
  };

  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<uint8_t> &query, size_t k) override;
  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<float> &query, size_t k);

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
  bool setEmbeddings(const std::vector<std::vector<float>> &float_embeddings,
                     double distib_factor);

private:
  // static const u_int8_t MUL_RESULTS_SIZE = 64;
  alignas(32) float mapped_floats[256];
  // float mapped_floats_mul_result[MUL_RESULTS_SIZE][MUL_RESULTS_SIZE];
  float cosine_similarity(const std::vector<uint8_t> &a,
                          const std::vector<uint8_t> &b) {
    throw std::runtime_error("not implemented");
  };
  float cosine_similarity(const float *a, const std::vector<uint8_t> &b);
  std::vector<PartitionInfo> partitions;
};