#include "embedding_search_avx2.h"
#include "embedding_utils.h" // for convertEmbeddingsToAVX2, validateAVX2Di...
#include <algorithm>         // for partial_sort
#include <cmath>             // for sqrt
#include <pmmintrin.h>       // for _mm_hadd_ps
#include <stdexcept>         // for runtime_error
#include <xmmintrin.h>       // for __m128, _mm_add_ps, _mm_cvtss_f32

std::vector<std::pair<float, size_t>>
EmbeddingSearchAVX2::similarity_search(const avx2_vector &query, size_t k) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<float, size_t>> similarities;
  similarities.reserve(embeddings.size());

  for (size_t i = 0; i < embeddings.size(); ++i) {
    float sim = cosine_similarity(query, embeddings[i]);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<float, size_t>>(similarities.begin(),
                                               similarities.begin() + k);
}

std::vector<std::pair<float, size_t>> EmbeddingSearchAVX2::similarity_search(
    const avx2_vector &query, size_t k,
    std::vector<std::pair<int, size_t>> &searchIndexes) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<float, size_t>> similarities;
  similarities.reserve(searchIndexes.size());

  for (size_t i = 0; i < searchIndexes.size(); i++) {
    float sim = cosine_similarity(query, embeddings[searchIndexes[i].second]);
    similarities.emplace_back(sim, searchIndexes[i].second);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<float, size_t>>(similarities.begin(),
                                               similarities.begin() + k);
}

bool EmbeddingSearchAVX2::setEmbeddings(
    const std::vector<std::vector<float>> &m) {
  std::string error_message;
  if (!validateDimensions(m, error_message)) {
    throw std::runtime_error(error_message);
  }

  vector_dim = m[0].size() / 8;
  num_vectors = m.size();
  embeddings.clear();
  EmbeddingUtils::convertEmbeddingsToAVX2(m, embeddings, vector_dim);
  return true;
}

bool EmbeddingSearchAVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateAVX2Dimensions(input, error_message);
}

// Helper function to horizontally sum all 8 float elements in a 256-bit AVX
// vector
inline float _mm256_reduce_add_ps(__m256 x) {
  // Extract the high 128 bits into a new 128-bit vector
  __m128 high128 = _mm256_extractf128_ps(x, 1);

  // Cast the low 128 bits into a new 128-bit vector (no actual data movement)
  __m128 low128 = _mm256_castps256_ps128(x);

  // Add corresponding elements of high and low vectors
  // Result: (high[0]+low[0], high[1]+low[1], high[2]+low[2], high[3]+low[3])
  __m128 sum = _mm_add_ps(high128, low128);

  // Horizontally add adjacent pairs of floats
  // Before: (a, b, c, d)
  // After:  (a+b, c+d, a+b, c+d)
  sum = _mm_hadd_ps(sum, sum);

  // Horizontally add adjacent pairs again
  // Before: (a+b, c+d, a+b, c+d)
  // After:  (a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d)
  sum = _mm_hadd_ps(sum, sum);

  // Extract the lowest 32 bits (first float) from the vector
  return _mm_cvtss_f32(sum);
}

// Calculate cosine similarity between two vectors using AVX2 instructions
// Cosine similarity = (a·b)/(|a|·|b|) where a·b is dot product and |a|,|b| are
// magnitudes
float EmbeddingSearchAVX2::cosine_similarity(const avx2_vector &a,
                                             const avx2_vector &b) {
  // Initialize 256-bit vectors to store:
  // - running sum of dot product (a·b)
  // - running sum of squared magnitudes (|a|² and |b|²)
  __m256 dot_product = _mm256_setzero_ps();
  __m256 mag_a = _mm256_setzero_ps();
  __m256 mag_b = _mm256_setzero_ps();

  // Process 8 floats at a time using AVX2
  for (size_t i = 0; i < a.size(); ++i) {
    // Multiply corresponding elements: a[i] * b[i]
    __m256 prod = _mm256_mul_ps(a[i], b[i]);

    // Add to running dot product sum
    dot_product = _mm256_add_ps(dot_product, prod);

    // Add to running magnitude sums: a[i]² and b[i]²
    mag_a = _mm256_add_ps(mag_a, _mm256_mul_ps(a[i], a[i]));
    mag_b = _mm256_add_ps(mag_b, _mm256_mul_ps(b[i], b[i]));
  }

  // Horizontally sum all elements in each vector
  float dot_product_sum = _mm256_reduce_add_ps(dot_product);
  float mag_a_sum = _mm256_reduce_add_ps(mag_a);
  float mag_b_sum = _mm256_reduce_add_ps(mag_b);

  // Return final cosine similarity:
  // dot_product / (sqrt(|a|²) * sqrt(|b|²))
  return dot_product_sum / (std::sqrt(mag_a_sum) * std::sqrt(mag_b_sum));
}

float EmbeddingSearchAVX2::dot_product_avx2(__m256 a, __m256 b) {
  __m256 mul = _mm256_mul_ps(a, b);
  return _mm256_reduce_add_ps(mul);
}