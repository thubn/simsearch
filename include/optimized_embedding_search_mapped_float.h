// optimized_embedding_search_mapped_float.h
#pragma once
#include "aligned_types.h"
#include "embedding_search_base.h"
#include <immintrin.h>

enum class DistributionType {
  UNIFORM,
  LINEAR,
  QUADRATIC,
  CUBIC,
  GAUSSIAN,
  COSINE
};

class OptimizedEmbeddingSearchMappedFloat
    : public OptimizedEmbeddingSearchBase<avx2_vector, float, uint8_t> {
public:
  OptimizedEmbeddingSearchMappedFloat() = default;

  bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;
  bool setEmbeddings(const std::vector<std::vector<float>> &input_vectors,
                     double distrib_factor, DistributionType dist_type);
  std::vector<std::pair<float, size_t>>
  similarity_search(const avx2_vector &query, size_t k) override;
  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<float> &query, size_t k);
  std::vector<float> getEmbedding(size_t index) const;
  void convertToMappedFormat(const std::vector<float> &input,
                             uint8_t *output) const;

protected:
  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override;

private:
  alignas(32) float mapped_floats[256];

  /*struct PartitionInfo {
    float start;
    float end;
    float average;
    size_t count;
  };*/

  /*double calculateWeight(double relative_pos, DistributionType dist_type,
                         double factor) const;
  void computePartitionBoundaries(const std::vector<float> &sorted_values,
                                  std::vector<PartitionInfo> &partitions,
                                  DistributionType dist_type, double factor);
  void computeMappedFloats(const std::vector<std::vector<float>> &input_vectors,
                           double distrib_factor, DistributionType dist_type);*/
  float cosine_similarity_optimized(const uint8_t *vec_a,
                                    const uint8_t *vec_b) const override;
  // uint8_t findPartitionIndex(float value) const;
};