#include "optimized_embedding_search_uint8_avx2.h"
#include "embedding_utils.h" // for validateUint8AVX2Dimensions
#include <algorithm>         // for clamp, partial_sort
#include <arm_neon.h>        // for _MM_SHUFFLE, _MM_HINT_T0, _mm_prefetch
#include <iostream>          // for basic_ostream, operator<<, cerr, endl
#include <stdexcept>         // for runtime_error, out_of_range
#include <stdint.h>          // for int8_t

bool OptimizedEmbeddingSearchUint8AVX2::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  std::string error_message;
  if (!validateDimensions(input_vectors, error_message))
    throw std::runtime_error(error_message);

  if (!initializeDimensions(input_vectors))
    return false;

  // Calculate padding for int8 values (16 values per NEON vector)
  padded_dim = ((vector_dim + 15) / 16) * 16;
  vectors_per_embedding =
      padded_dim / 16; // Number of int8x16_t vectors needed per embedding

  // Allocate aligned memory
  size_t total_vectors = num_vectors * vectors_per_embedding;
  if (!allocateAlignedMemory(total_vectors))
    return false;

  // Convert and store each vector
  for (size_t i = 0; i < num_vectors; i++) {
    int8x16_t *dest = get_embedding_ptr(i);
    convert_float_to_uint8_avx2(input_vectors[i], dest);
  }

  return true;
}

avx2i_vector8
OptimizedEmbeddingSearchUint8AVX2::getEmbeddingAVX2(size_t index) const {
  if (index >= num_vectors) {
    throw std::out_of_range("Embedding index out of range");
  }

  avx2i_vector8 result(vectors_per_embedding);
  const int8x16_t *src = get_embedding_ptr(index);

  // Each int8x16_t contains 16 int8 values
  for (size_t i = 0; i < vectors_per_embedding; ++i) {
    result[i] = vld1q_s8(reinterpret_cast<const int8_t *>(&src[i]));
  }

  return result;
}

std::vector<std::pair<int, size_t>>
OptimizedEmbeddingSearchUint8AVX2::similarity_search(const avx2i_vector8 &query,
                                                     size_t k) {
  if (query.size() != vectors_per_embedding) {
    std::cerr << "expected dimension: " << vectors_per_embedding
              << "\ngot dimension: " << query.size() << std::endl;
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<int, size_t>> similarities;
  similarities.reserve(num_vectors);

  const int8x16_t *query_data =
      reinterpret_cast<const int8x16_t *>(query.data());

  for (size_t i = 0; i < num_vectors; i++) {
    int sim = cosine_similarity_optimized(get_embedding_ptr(i), query_data);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<int, size_t>>(similarities.begin(),
                                             similarities.begin() + k);
}

bool OptimizedEmbeddingSearchUint8AVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateUint8AVX2Dimensions(input, error_message);
}

void OptimizedEmbeddingSearchUint8AVX2::convert_float_to_uint8_avx2(
    const std::vector<float> &input, int8x16_t *output) const {
  for (size_t i = 0; i < vectors_per_embedding; i++) {
    std::vector<int8_t> temp(16, 0);

    for (size_t j = 0; j < 16 && (i * 16 + j) < input.size(); j++) {
      float val = input[i * 16 + j];
      temp[j] = static_cast<int8_t>(std::clamp(val * 127.0f, -127.0f, 127.0f));
    }

    output[i] = vld1q_s8(temp.data());
  }
}

int OptimizedEmbeddingSearchUint8AVX2::cosine_similarity_optimized(
    const int8x16_t *vec_a, const int8x16_t *vec_b) const {
  // Initialize 16-bit accumulators
  int16x8_t acc16_low = vdupq_n_s16(0);
  int16x8_t acc16_high = vdupq_n_s16(0);

  for (size_t i = 0; i < vectors_per_embedding; i++) {
    // Load vectors
    int8x16_t a = vld1q_s8(reinterpret_cast<const int8_t *>(&vec_a[i]));
    int8x16_t b = vld1q_s8(reinterpret_cast<const int8_t *>(&vec_b[i]));

    // Use vmlal_s8 correctly - it needs an existing 16-bit accumulator
    // Process lower half of vectors
    acc16_low = vmlal_s8(acc16_low, vget_low_s8(a), vget_low_s8(b));

    // Process upper half of vectors
    acc16_high = vmlal_s8(acc16_high, vget_high_s8(a), vget_high_s8(b));
  }

  // Now perform the horizontal sum after the loop
  // First convert 16-bit to 32-bit
  int32x4_t sum_low = vpaddlq_s16(acc16_low);
  int32x4_t sum_high = vpaddlq_s16(acc16_high);

  // Combine the two 32-bit vectors
  int32x4_t acc = vaddq_s32(sum_low, sum_high);

  // Horizontal sum of the final 32-bit accumulator
  int32x2_t sum = vadd_s32(vget_high_s32(acc), vget_low_s32(acc));
  sum = vpadd_s32(sum, sum);

  return vget_lane_s32(sum, 0);
}
