#include "optimized_embedding_search_avx2.h"
#include "embedding_io.h"
#include "embedding_utils.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

constexpr int_fast8_t NUM_READ_POSITIONS = 6;

bool OptimizedEmbeddingSearchAVX2::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  if (input_vectors.empty() || input_vectors[0].empty()) {
    return false;
  }

  try {
    std::string error_message;
    if (!validateDimensions(input_vectors, error_message))
      throw std::runtime_error(error_message);

    if (!initializeDimensions(input_vectors))
      return false;
    padded_dim = ((vector_dim + 7) / 8) * 8;
    vectors_per_embedding = padded_dim;

    // Allocate aligned memory for embeddings
    size_t total_size = num_vectors * padded_dim;
    if (!allocateAlignedMemory(total_size))
      return false;

    // Initialize norms
    // norms.resize(num_vectors);

    // Copy and process embeddings
    for (size_t i = 0; i < num_vectors; i++) {
      float *dest = get_embedding_ptr(i);

      // Copy embedding data
      std::memcpy(dest, input_vectors[i].data(), vector_dim * sizeof(float));

      // Zero-pad if necessary
      if (padded_dim > vector_dim) {
        std::memset(dest + vector_dim, 0,
                    (padded_dim - vector_dim) * sizeof(float));
      }

      // Compute and store norm
      // norms[i] = compute_norm_avx2(dest);
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Failed to load vectors: " << e.what() << std::endl;
    return false;
  }
}

std::vector<float>
OptimizedEmbeddingSearchAVX2::getEmbedding(size_t index) const {
  if (index >= num_vectors) {
    throw std::out_of_range("Embedding index out of range");
  }

  std::vector<float> result(vector_dim);
  const float *embedding = get_embedding_ptr(index);
  std::copy(embedding, embedding + vector_dim, result.begin());
  return result;
}

std::vector<std::pair<float, size_t>>
OptimizedEmbeddingSearchAVX2::similarity_search(const avx2_vector &query,
                                                size_t k) {
  throw std::runtime_error(
      "AVX2 vector input not supported in optimized version");
}

std::vector<std::pair<float, size_t>>
OptimizedEmbeddingSearchAVX2::similarity_search(const std::vector<float> &query,
                                                size_t k) {
  if (query.size() != vector_dim) {
    throw std::invalid_argument("Invalid query dimension");
  }

  // Prepare aligned query vector
  auto query_aligned = aligned_vector<float>(padded_dim);
  std::memcpy(query_aligned.data(), query.data(), vector_dim * sizeof(float));
  if (padded_dim > vector_dim) {
    std::memset(query_aligned.data() + vector_dim, 0,
                (padded_dim - vector_dim) * sizeof(float));
  }
  // float query_norm = compute_norm_avx2(query_aligned.data());

  // Calculate similarities
  // std::vector<std::pair<float, size_t>> results;
  // results.reserve(num_vectors);
  std::vector<std::pair<float, size_t>> results(num_vectors);

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; i + NUM_READ_POSITIONS - 1 < num_vectors;
       i += NUM_READ_POSITIONS) {
    float sim[NUM_READ_POSITIONS] = {};
    cosine_similarity_optimized(i, query_aligned.data(), sim);
    results[i] = std::make_pair(sim[0], i);
    results[i + 1] = std::make_pair(sim[1], i + 1);
    results[i + 2] = std::make_pair(sim[2], i + 2);
    results[i + 3] = std::make_pair(sim[3], i + 3);
    results[i + 4] = std::make_pair(sim[4], i + 4);
    results[i + 5] = std::make_pair(sim[5], i + 5);
    // results[i * NUM_READ_POSITIONS + 6] = std::make_pair(sim[6], i + 6 *
    // rpo); results[i * NUM_READ_POSITIONS + 7] = std::make_pair(sim[7], i + 7
    // * rpo);
  }

  // Partial sort to get top-k results
  if (results.size() > k) {
    std::partial_sort(
        results.begin(), results.begin() + k, results.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });
    results.resize(k);
  }

  return results;
}

std::vector<std::pair<float, size_t>>
OptimizedEmbeddingSearchAVX2::similarity_search(
    const std::vector<float> &query, size_t k,
    std::vector<std::pair<int, size_t>> &searchIndexes) {
  if (query.size() != vector_dim) {
    throw std::invalid_argument("Invalid query dimension");
  }

  // Prepare aligned query vector
  auto query_aligned = aligned_vector<float>(padded_dim);
  std::memcpy(query_aligned.data(), query.data(), vector_dim * sizeof(float));
  if (padded_dim > vector_dim) {
    std::memset(query_aligned.data() + vector_dim, 0,
                (padded_dim - vector_dim) * sizeof(float));
  }
  float query_norm = compute_norm_avx2(query_aligned.data());

  // Calculate similarities
  std::vector<std::pair<float, size_t>> results;
  results.reserve(searchIndexes.size());

  for (size_t i = 0; i < searchIndexes.size(); i++) {
    float similarity = cosine_similarity_optimized(
        get_embedding_ptr(searchIndexes[i].second), query_aligned.data());
    results.emplace_back(similarity, searchIndexes[i].second);
  }

  // Partial sort to get top-k results
  if (results.size() > k) {
    std::partial_sort(
        results.begin(), results.begin() + k, results.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });
    results.resize(k);
  }

  return results;
}

bool OptimizedEmbeddingSearchAVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateAVX2Dimensions(input, error_message);
}

float OptimizedEmbeddingSearchAVX2::compute_norm_avx2(const float *vec) const {
  __m256 sum = _mm256_setzero_ps();

  for (size_t i = 0; i < padded_dim; i += 8) {
    __m256 v = _mm256_load_ps(vec + i);
    sum = _mm256_fmadd_ps(v, v, sum);
  }

  // Horizontal sum and square root
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum_128 = _mm_add_ps(hi, lo);
  sum_128 = _mm_hadd_ps(sum_128, sum_128);
  sum_128 = _mm_hadd_ps(sum_128, sum_128);

  return std::sqrt(_mm_cvtss_f32(sum_128));
}
float OptimizedEmbeddingSearchAVX2::cosine_similarity_optimized(
    const float *vec_a, const float *vec_b) const {
  return 0.0f;
}

inline void OptimizedEmbeddingSearchAVX2::cosine_similarity_optimized(
    const int j, const float *vec_a, float *sim) {
  // Check if we have enough vectors
  __m256 sum[NUM_READ_POSITIONS] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                                    _mm256_setzero_ps(), _mm256_setzero_ps(),
                                    _mm256_setzero_ps(), _mm256_setzero_ps()};
  //_mm256_setzero_ps(), _mm256_setzero_ps()
  //_mm256_setzero_ps(), _mm256_setzero_ps()};
  //  _mm256_setzero_ps(), _mm256_setzero_ps()};
  __m256 a;
  __m256 b[NUM_READ_POSITIONS];
  // Load embedding pointers
  const float *emb_ptr[NUM_READ_POSITIONS] = {
      get_embedding_ptr(j),     get_embedding_ptr(j + 1),
      get_embedding_ptr(j + 2), get_embedding_ptr(j + 3),
      get_embedding_ptr(j + 4), get_embedding_ptr(j + 5)};
  // get_embedding_ptr(j + (6 * rpo)),
  // get_embedding_ptr(j + (7 * rpo))

  for (int i = 0; i < padded_dim; i += 8) {
    a = _mm256_load_ps(vec_a + i);

    b[0] = _mm256_load_ps(emb_ptr[0] + i);
    b[1] = _mm256_load_ps(emb_ptr[1] + i);
    b[2] = _mm256_load_ps(emb_ptr[2] + i);
    b[3] = _mm256_load_ps(emb_ptr[3] + i);
    b[4] = _mm256_load_ps(emb_ptr[4] + i);
    b[5] = _mm256_load_ps(emb_ptr[5] + i);
    //[6] = _mm256_load_ps(emb_ptr[6] + i);
    // b[7] = _mm256_load_ps(emb_ptr[7] + i);

    sum[0] = _mm256_fmadd_ps(a, b[0], sum[0]);
    sum[1] = _mm256_fmadd_ps(a, b[1], sum[1]);
    sum[2] = _mm256_fmadd_ps(a, b[2], sum[2]);
    sum[3] = _mm256_fmadd_ps(a, b[3], sum[3]);
    sum[4] = _mm256_fmadd_ps(a, b[4], sum[4]);
    sum[5] = _mm256_fmadd_ps(a, b[5], sum[5]);
    // sum[6] = _mm256_fmadd_ps(a, b[6], sum[6]);
    // sum[7] = _mm256_fmadd_ps(a, b[7], sum[7]);
  }

  for (int i = 0; i < NUM_READ_POSITIONS; i++) {
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum[i], 1);
    __m128 lo = _mm256_castps256_ps128(sum[i]);
    __m128 sum_128 = _mm_add_ps(hi, lo);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sim[i] = _mm_cvtss_f32(sum_128);
  }
}