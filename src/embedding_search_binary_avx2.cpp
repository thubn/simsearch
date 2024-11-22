#include "embedding_search_binary_avx2.h"
#include "embedding_io.h"
#include "embedding_utils.h"
#include <algorithm>
#include <bit>
#include <omp.h>
#include <stdexcept>

std::vector<std::pair<int, size_t>>
EmbeddingSearchBinaryAVX2::similarity_search(const avx2i_vector &query,
                                             size_t k) {
  if (query.size() != vector_dim) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<int, size_t>> similarities;
  similarities.reserve(embeddings.size());

  for (size_t i = 0; i < embeddings.size(); ++i) {
    int sim = cosine_similarity(query, embeddings[i]);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<int, size_t>>(similarities.begin(),
                                             similarities.begin() + k);
}

bool EmbeddingSearchBinaryAVX2::setEmbeddings(
    const std::vector<std::vector<float>> &float_data) {
  std::string error_message;
  if (!validateDimensions(float_data, error_message)) {
    throw std::runtime_error(error_message);
  }

  num_vectors = float_data.size();
  size_t float_vector_size = float_data[0].size();

  vector_dim = (float_vector_size + 255) /
                256; // Round up to nearest multiple of 256 bits
  embeddings.resize(num_vectors, avx2i_vector(vector_dim));

  for (size_t i = 0; i < num_vectors; ++i) {
    for (size_t j = 0; j < float_vector_size; ++j) {
      if (float_data[i][j] >= 0) {
        size_t vector_idx = j / 256;
        size_t bit_pos = j % 256;
        size_t chunk_idx = bit_pos / 64;
        size_t local_bit_pos = bit_pos % 64;

        uint64_t *ptr =
            reinterpret_cast<uint64_t *>(&embeddings[i][vector_idx]);
        ptr[chunk_idx] |= (1ULL << (63 - local_bit_pos));
      }
    }
  }

  return true;
}

bool EmbeddingSearchBinaryAVX2::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
  return EmbeddingUtils::validateBinaryAVX2Dimensions(input, error_message);
}

int EmbeddingSearchBinaryAVX2::cosine_similarity(const avx2i_vector &a,
                                                 const avx2i_vector &b) {
  int dot_product = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    __m256i result = _mm256_xor_si256(a[i], b[i]);
    __m256i all_ones = _mm256_set1_epi32(-1);
    result = _mm256_xor_si256(result, all_ones);

    uint64_t *result_ptr = reinterpret_cast<uint64_t *>(&result);
    dot_product += __builtin_popcountll(result_ptr[0]);
    dot_product += __builtin_popcountll(result_ptr[1]);
    dot_product += __builtin_popcountll(result_ptr[2]);
    dot_product += __builtin_popcountll(result_ptr[3]);
  }
  return dot_product;
}