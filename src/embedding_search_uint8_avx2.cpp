#include "embedding_search_uint8_avx2.h"
#include "embedding_utils.h" // for validateUint8AVX2Dimensions, calculateU...
#include <algorithm>         // for partial_sort
#include <emmintrin.h>       // for _mm_add_epi32, _mm_cvtsi128_si32, _mm_s...
#include <stdexcept>         // for runtime_error
#include <xmmintrin.h>       // for _MM_SHUFFLE

std::vector<std::pair<uint, size_t>>
EmbeddingSearchUint8AVX2::similarity_search(const avx2i_vector &query,
                                            size_t k) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<uint, size_t>> similarities;
  similarities.reserve(embeddings.size());

  for (size_t i = 0; i < embeddings.size(); ++i) {
    uint sim = cosine_similarity(query, embeddings[i]);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<uint, size_t>>(similarities.begin(),
                                              similarities.begin() + k);
}

std::vector<std::pair<uint, size_t>>
EmbeddingSearchUint8AVX2::similarity_search(
    const avx2i_vector &query, size_t k,
    std::vector<std::pair<int, size_t>> &searchIndexes) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<uint, size_t>> similarities;
  similarities.reserve(searchIndexes.size());

  for (const auto &[_, idx] : searchIndexes) {
    uint sim = cosine_similarity(query, embeddings[idx]);
    similarities.emplace_back(sim, idx);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<uint, size_t>>(similarities.begin(),
                                              similarities.begin() + k);
}

bool EmbeddingSearchUint8AVX2::setEmbeddings(
    const std::vector<std::vector<float>> &m) {
  std::string error_message;
  if (!EmbeddingUtils::validateUint8AVX2Dimensions(m, error_message)) {
    throw std::runtime_error(error_message);
  }

  num_vectors = m.size();
  size_t float_vector_size = m[0].size();
  vector_dim = EmbeddingUtils::calculateUint8AVX2VectorSize(float_vector_size);

  embeddings.clear();
  embeddings.resize(num_vectors, avx2i_vector(vector_dim));

  for (size_t i = 0; i < num_vectors; ++i) {
    EmbeddingUtils::convertSingleFloatToUint8AVX2(m[i], embeddings[i],
                                                  vector_dim);
  }

  return true;
}

bool EmbeddingSearchUint8AVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateUint8AVX2Dimensions(input, error_message);
}

uint EmbeddingSearchUint8AVX2::cosine_similarity(const avx2i_vector &a,
                                                 const avx2i_vector &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("Vectors must have the same size");
  }

  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();

  for (size_t i = 0; i < a.size(); ++i) {
    __m256i mul_lo = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[i], 0)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[i], 0)));
    __m256i mul_hi = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[i], 1)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[i], 1)));

    sum_lo = _mm256_add_epi32(sum_lo,
                              _mm256_madd_epi16(mul_lo, _mm256_set1_epi16(1)));
    sum_hi = _mm256_add_epi32(sum_hi,
                              _mm256_madd_epi16(mul_hi, _mm256_set1_epi16(1)));
  }

  __m256i sum = _mm256_add_epi32(sum_lo, sum_hi);
  __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                  _mm256_extracti128_si256(sum, 1));

  sum_128 = _mm_add_epi32(sum_128,
                          _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(1, 0, 3, 2)));
  sum_128 = _mm_add_epi32(sum_128,
                          _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(2, 3, 0, 1)));

  return _mm_cvtsi128_si32(sum_128);
}