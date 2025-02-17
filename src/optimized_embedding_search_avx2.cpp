#include "optimized_embedding_search_avx2.h"
#include "embedding_utils.h" // for apply_pca_dimension_reduction_to_query
#include <algorithm>         // for partial_sort, copy
#include <arm_neon.h>        // for _mm256_load_ps, _mm256_setzero_ps, _mm2...
#include <cmath>             // for sqrt
#include <cstring>           // for size_t, memcpy, memset
#include <exception>         // for exception
#include <iostream>          // for basic_ostream, char_traits, operator<<
#include <stdexcept>         // for invalid_argument, runtime_error, out_of...
#include <stdint.h>          // for int32_t, int_fast8_t

constexpr int_fast8_t NUM_STRIDES = 1;
// constexpr int_fast64_t STRIDE_DIST = 1;

const int32_t calcStrideDist(const size_t &vector_dim,
                             const size_t size_of_datatype,
                             const size_t num_vectors) {
  int32_t size_per_embedding = vector_dim * size_of_datatype;
  int32_t i = 1;
  int32_t max = num_vectors / 32;
  while (i < max && i <= 2048) {
    if (i * size_per_embedding % 4096 == 0) {
      std::cout << "vector_dim: " << vector_dim << " STRIDE_DIST: " << i
                << " STRIDE_DIST in bytes: " << i * size_per_embedding
                << std::endl;
      return i;
    } else {
      i++;
    }
  }
  std::cout << "No optimal stride distance found :(\n"
            << "vector_dim: " << vector_dim << " STRIDE_DIST: " << 1
            << " STRIDE_DIST in bytes: " << size_per_embedding << std::endl;
  return 1;
}

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
    padded_dim = ((vector_dim + 3) / 4) * 4;
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

    stride_dist = calcStrideDist(vector_dim, sizeof(float), num_vectors);

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
  std::vector<float> temp_query;
  if (is_pca && query.size() != vector_dim) {
    temp_query = EmbeddingUtils::apply_pca_dimension_reduction_to_query(
        pca_matrix, mean, query);
  } else {
    temp_query = query;
  }
  if (temp_query.size() != vector_dim) {
    std::cout << "temp_query.size(): " << temp_query.size()
              << " vector_dim: " << vector_dim << std::endl;
    throw std::invalid_argument("Invalid query dimension");
  }

  // Prepare aligned query vector
  auto query_aligned = aligned_vector<float>(padded_dim);
  std::memcpy(query_aligned.data(), temp_query.data(),
              vector_dim * sizeof(float));
  if (padded_dim > vector_dim) {
    std::memset(query_aligned.data() + vector_dim, 0,
                (padded_dim - vector_dim) * sizeof(float));
  }

  std::vector<std::pair<float, size_t>> results(num_vectors);
  const size_t STRIDE_DIST = stride_dist;

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; (i + (NUM_STRIDES * STRIDE_DIST) - 1) < num_vectors;
       i += (NUM_STRIDES * STRIDE_DIST)) {
    for (size_t j = i; j < (i + STRIDE_DIST); j++) {
      float sim[NUM_STRIDES] = {};
      float *emb_ptr[NUM_STRIDES];
      for (int k = 0; k < NUM_STRIDES; k++) {
        emb_ptr[k] = get_embedding_ptr(j + k * STRIDE_DIST);
      }
      cosine_similarity_optimized(query_aligned.data(), sim, emb_ptr);
      for (int k = 0; k < NUM_STRIDES; k++) {
        results[j + k * STRIDE_DIST] =
            std::make_pair(sim[k], j + k * STRIDE_DIST);
      }
    }
  }
  // calc remaining similarities when strides dont fit with num_embeddings
  if (num_vectors % (NUM_STRIDES * STRIDE_DIST) != 0) {
    const size_t start =
        num_vectors - (num_vectors % (NUM_STRIDES * STRIDE_DIST));
    for (int i = start; i < num_vectors; i++) {
      float sim = cosine_similarity_optimized(get_embedding_ptr(i),
                                              query_aligned.data());
      results[i] = std::make_pair(sim, i);
    }
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
  float32x4_t sum = vdupq_n_f32(0.0f);

  for (size_t i = 0; i < padded_dim; i += 4) {
    float32x4_t v = vld1q_f32(vec + i);
    sum = vmlaq_f32(sum, v, v); // Fused multiply-add
  }

  // Horizontal sum and square root
  float32x2_t sum2 = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
  sum2 = vpadd_f32(sum2, sum2);

  return std::sqrt(vget_lane_f32(sum2, 0));
}
float OptimizedEmbeddingSearchAVX2::cosine_similarity_optimized(
    const float *vec_a, const float *vec_b) const {
  float32x4_t sum = vdupq_n_f32(0.0f);

  // Process 4 floats at a time
  for (size_t i = 0; i < padded_dim; i += 4) {
    float32x4_t a = vld1q_f32(vec_a + i);
    float32x4_t b = vld1q_f32(vec_b + i);
    sum = vmlaq_f32(sum, a, b); // Fused multiply-add
  }

  // Horizontal sum
  float32x2_t sum2 = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
  sum2 = vpadd_f32(sum2, sum2);

  return vget_lane_f32(sum2, 0);
}

inline void OptimizedEmbeddingSearchAVX2::cosine_similarity_optimized(
    const float *vec_a, float *sim, float *emb_ptr[]) {
  float32x4_t sum[NUM_STRIDES];
  float32x4_t a;
  float32x4_t b[NUM_STRIDES];

  // Initialize sums to zero
  for (int i = 0; i < NUM_STRIDES; i++) {
    sum[i] = vdupq_n_f32(0.0f);
  }

  // Process 4 floats at a time for each stride
  for (size_t i = 0; i < padded_dim; i += 4) {
    a = vld1q_f32(vec_a + i);

    for (int j = 0; j < NUM_STRIDES; j++) {
      b[j] = vld1q_f32(emb_ptr[j] + i);
      sum[j] = vmlaq_f32(sum[j], a, b[j]); // Fused multiply-add
    }
  }

  // Calculate final similarities for each stride
  for (int i = 0; i < NUM_STRIDES; i++) {
    float32x2_t sum2 = vadd_f32(vget_high_f32(sum[i]), vget_low_f32(sum[i]));
    sum2 = vpadd_f32(sum2, sum2);
    sim[i] = vget_lane_f32(sum2, 0);
  }
}
