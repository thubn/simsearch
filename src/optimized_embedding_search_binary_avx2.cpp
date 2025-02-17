#include "optimized_embedding_search_binary_avx2.h"
#include "embedding_utils.h" // for validateBinaryAVX2Dimensions
#include <algorithm>         // for partial_sort
#include <arm_neon.h>
#include <iostream>  // for basic_ostream, operator<<, cerr, endl
#include <stdexcept> // for runtime_error, out_of_range

constexpr int_fast8_t NUM_STRIDES = 1;
constexpr int_fast64_t STRIDE_DIST = 1;

bool OptimizedEmbeddingSearchBinaryAVX2::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  std::string error_message;
  if (!validateDimensions(input_vectors, error_message))
    throw std::runtime_error(error_message);

  if (!initializeDimensions(input_vectors))
    return false;

  // Calculate padding for binary values (128 bits per NEON vector)

  padded_dim = ((vector_dim + 127) / 128) * 128;
  vectors_per_embedding =
      padded_dim / 128; // Number of uint32x4_t vectors needed per embedding

  // Allocate aligned memory
  size_t total_vectors = num_vectors * vectors_per_embedding;
  if (!allocateAlignedMemory(total_vectors))
    return false;

  // Convert and store each vector
  for (size_t i = 0; i < num_vectors; i++) {
    uint32x4_t *dest = get_embedding_ptr(i);
    convert_float_to_binary_neon(input_vectors[i], dest);
  }

  return true;
}

avx2i_vector
OptimizedEmbeddingSearchBinaryAVX2::getEmbeddingAVX2(size_t index) const {
  if (index >= num_vectors) {
    throw std::out_of_range("Embedding index out of range");
  }

  avx2i_vector result(vectors_per_embedding);
  const uint32x4_t *src = get_embedding_ptr(index);

  // Each uint32x4_t contains 128 bits
  for (size_t i = 0; i < vectors_per_embedding; ++i) {
    result[i] = vld1q_u32(reinterpret_cast<const uint32_t *>(&src[i]));
  }

  return result;
}

std::vector<std::pair<int32_t, size_t>>
OptimizedEmbeddingSearchBinaryAVX2::similarity_search(const avx2i_vector &query,
                                                      size_t k) {
  if (query.size() != vectors_per_embedding) {
    std::cerr << "expected dimension: " << vectors_per_embedding
              << "\ngot dimension: " << query.size() << std::endl;
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<int32_t, size_t>> similarities;
  similarities.reserve(num_vectors);

  const __m256i *query_data = reinterpret_cast<const __m256i *>(query.data());
  // counter = AVX2Popcount();
  //  AVX2PopcountHarleySeal counter;

  if (vectors_per_embedding == 4) {
    for (size_t i = 0; i < num_vectors; i++) {
      int32_t sim =
          cosine_similarity_optimized(get_embedding_ptr(i), query_data);
      similarities.emplace_back(sim, i);
    }
  } else {
    for (size_t i = 0; i < num_vectors; i++) {
      int32_t sim =
          cosine_similarity_optimized_dynamic(get_embedding_ptr(i), query_data);
      similarities.emplace_back(sim, i);
    }
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<int32_t, size_t>>(similarities.begin(),
                                                 similarities.begin() + k);
}

bool OptimizedEmbeddingSearchBinaryAVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateBinaryAVX2Dimensions(input, error_message);
}

void OptimizedEmbeddingSearchBinaryAVX2::convert_float_to_binary_avx2(
    const std::vector<float> &input, uint32x4_t *output) const {
  for (size_t i = 0; i < vectors_per_embedding; i++) {
    uint32_t bits[4] = {0, 0, 0, 0}; // 128 bits total

    // Process each 32-bit chunk
    for (size_t chunk = 0; chunk < 4; chunk++) {
      // Process each bit within the chunk
      for (size_t bit = 0; bit < 32; bit++) {
        size_t input_idx = i * 128 + chunk * 32 + bit;
        if (input_idx < input.size() && input[input_idx] >= 0) {
          bits[chunk] |= (1U << (31 - bit));
        }
      }
    }

    // Combine into single NEON vector
    output[i] = vld1q_u32(bits);
  }
}

// currently hardcoded for 1024bit vectors
int32_t OptimizedEmbeddingSearchBinaryAVX2::cosine_similarity_optimized(
    const uint32x4_t *vec_a, const uint32x4_t *vec_b) const {

  // For 1024-bit vectors (8 x 128-bit NEON vectors)
  uint32x4_t all_ones = vdupq_n_u32(0xFFFFFFFF);
  int32_t = total_popcount = 0;

  // Process 8 vectors (1024 bits total)
  for (int i = 0; i < 8; i++) {
    // Compute XOR and NOT for each pair of vectors
    uint32x4_t xor_result = veorq_u32(vec_a[i], vec_b[i]);
    xor_result = veorq_u32(xor_result, all_ones);

    // Convert to bytes for VCNT (which operates on uint8x16_t)
    uint8x16_t bytes = vreinterpretq_u8_u32(xor_result);

    // Count 1s in each byte
    uint8x16_t popcnt = vcntq_u8(bytes);

    // Pairwise add to sum up counts
    uint16x8_t sum16 = vpaddlq_u8(popcnt);
    uint32x4_t sum32 = vpaddlq_u16(sum16);

    // Final horizontal sum of the 4 32-bit values
    uint64x2_t sum64 = vpaddlq_u32(sum32);
    total_popcount += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
  }

  return total_popcount;
}

int32_t OptimizedEmbeddingSearchBinaryAVX2::cosine_similarity_optimized_dynamic(
    const uint32x4_t *vec_a, const uint32x4_t *vec_b) const {

  uint32x4_t all_ones = vdupq_n_u32(0xFFFFFFFF);
  int32_t total_popcount = 0;

  // Process vectors_per_embedding number of 128-bit vectors
  for (int i = 0; i < vectors_per_embedding; i++) {
    // Compute XOR and NOT for each pair of vectors
    uint32x4_t xor_result = veorq_u32(vec_a[i], vec_b[i]);
    xor_result = veorq_u32(xor_result, all_ones);

    // Convert to bytes for VCNT (which operates on uint8x16_t)
    uint8x16_t bytes = vreinterpretq_u8_u32(xor_result);

    // Count 1s in each byte using VCNT
    uint8x16_t popcnt = vcntq_u8(bytes);

    // Pairwise add to sum up counts
    uint16x8_t sum16 = vpaddlq_u8(popcnt);
    uint32x4_t sum32 = vpaddlq_u16(sum16);

    // Final horizontal sum of the 4 32-bit values
    uint64x2_t sum64 = vpaddlq_u32(sum32);
    total_popcount += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
  }

  return total_popcount;
}
