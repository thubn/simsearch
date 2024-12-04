// embedding_utils.h
#pragma once
#include "aligned_types.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace EmbeddingUtils {
// Convert a single float vector to AVX2 format
void convertSingleEmbeddingToAVX2(const std::vector<float> &input,
                                  avx2_vector &output, size_t vector_dim);

// Convert multiple float vectors to AVX2 format in parallel
void convertEmbeddingsToAVX2(const std::vector<std::vector<float>> &input,
                             std::vector<avx2_vector> &output,
                             size_t vector_dim);

// Validate input dimensions for AVX2 conversion
bool validateAVX2Dimensions(const std::vector<std::vector<float>> &input,
                            std::string &error_message);

size_t calculateBinaryAVX2VectorSize(size_t float_vector_size);

void convertSingleFloatToBinaryAVX2(const std::vector<float> &input,
                                    avx2i_vector &output, size_t vector_dim);

bool validateBinaryAVX2Dimensions(const std::vector<std::vector<float>> &input,
                                  std::string &error_message);

void convertSingleFloatToUint8AVX2(const std::vector<float> &input,
                                   avx2i_vector &output, size_t vector_dim);

bool validateUint8AVX2Dimensions(const std::vector<std::vector<float>> &input,
                                 std::string &error_message);

size_t calculateUint8AVX2VectorSize(size_t float_vector_size);

std::string sanitize_utf8(const std::string &input);
bool pca_dimension_reduction(
    const int factor, const std::vector<std::vector<float>> &input_embeddings,
    std::vector<std::vector<float>> &result_embeddings,
    std::vector<std::vector<float>> &result_pca_matrix,
    std::vector<float> &result_mean);
std::vector<float> apply_pca_dimension_reduction_to_query(
    const std::vector<std::vector<float>> &pca_matrix, const std::vector<float> &mean,
    const std::vector<float> &query);

template <typename T1, typename T2>
double calculateJaccardIndex(const std::vector<std::pair<T1, size_t>> &set1,
                             const std::vector<std::pair<T2, size_t>> &set2) {
  std::vector<size_t> vec1, vec2;

  for (const auto &pair : set1)
    vec1.push_back(pair.second);
  for (const auto &pair : set2)
    vec2.push_back(pair.second);

  std::sort(vec1.begin(), vec1.end());
  std::sort(vec2.begin(), vec2.end());

  vec1.erase(std::unique(vec1.begin(), vec1.end()), vec1.end());
  vec2.erase(std::unique(vec2.begin(), vec2.end()), vec2.end());

  std::vector<size_t> intersection;
  std::set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(),
                        std::back_inserter(intersection));

  std::vector<size_t> union_set;
  std::set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(),
                 std::back_inserter(union_set));

  return union_set.empty()
             ? 0.0
             : static_cast<double>(intersection.size()) / union_set.size();
}

template <typename T1, typename T2>
double calculateNDCG(const std::vector<std::pair<T1, size_t>> &groundTruth,
                     const std::vector<std::pair<T2, size_t>> &prediction) {
  if (groundTruth.empty() || prediction.empty()) {
    return 0.0;
  }

  const size_t k = std::min(groundTruth.size(), prediction.size());

  // Create position lookup for ground truth
  std::unordered_map<size_t, size_t> truthPositions;
  for (size_t i = 0; i < k; ++i) {
    truthPositions[groundTruth[i].second] = i;
  }

  // Calculate DCG
  double dcg = 0.0;
  for (size_t i = 0; i < k; ++i) {
    auto it = truthPositions.find(prediction[i].second);
    if (it != truthPositions.end()) {
      // Calculate relevance score based on position difference
      // Perfect match (same position) gets relevance of 1.0
      // Relevance decreases based on position difference
      double positionDiff = std::abs(static_cast<double>(it->second) - i);
      double relevance = std::exp(-positionDiff / k); // Exponential decay

      // DCG formula: rel_i / log2(i + 2)
      dcg += relevance / std::log2(i + 2);
    }
  }

  // Calculate IDCG (ideal DCG - when order is perfect)
  double idcg = 0.0;
  for (size_t i = 0; i < k; ++i) {
    idcg += 1.0 / std::log2(i + 2); // Perfect relevance of 1.0
  }

  return idcg > 0 ? dcg / idcg : 0.0;
}

inline float calcNorm(const std::vector<float> &input) {
  float norm = 0.0f;
  for (const float &val : input) {
    norm += val * val;
  }
  return std::sqrt(norm);
}
} // namespace EmbeddingUtils