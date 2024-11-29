#include "optimized_embedding_search_uint8_avx2.h"
#include "embedding_utils.h"
#include <cstring>
#include <stdexcept>

bool OptimizedEmbeddingSearchUint8AVX2::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  std::string error_message;
  if (!validateDimensions(input_vectors, error_message))
    throw std::runtime_error(error_message);

  if (!initializeDimensions(input_vectors))
    return false;

  // Calculate padding for int8 values (32 values per AVX2 vector)
  padded_dim = ((vector_dim + 31) / 32) * 32;
  vectors_per_embedding =
      padded_dim / 32; // Number of __m256i vectors needed per embedding

  // Allocate aligned memory
  size_t total_vectors = num_vectors * vectors_per_embedding;
  if (!allocateAlignedMemory(total_vectors))
    return false;

  // Convert and store each vector
  for (size_t i = 0; i < num_vectors; i++) {
    __m256i *dest = get_embedding_ptr(i);
    convert_float_to_uint8_avx2(input_vectors[i], dest);
  }

  return true;
}

avx2i_vector
OptimizedEmbeddingSearchUint8AVX2::getEmbeddingAVX2(size_t index) const {
  if (index >= num_vectors) {
    throw std::out_of_range("Embedding index out of range");
  }

  avx2i_vector result(vectors_per_embedding);
  const __m256i *src = get_embedding_ptr(index);

  // Each __m256i contains 32 int8 values
  for (size_t i = 0; i < vectors_per_embedding; ++i) {
    result[i] = _mm256_load_si256(&src[i]);
  }

  return result;
}

std::vector<std::pair<int, size_t>>
OptimizedEmbeddingSearchUint8AVX2::similarity_search(const avx2i_vector &query,
                                                     size_t k) {
  if (query.size() != vectors_per_embedding) {
    std::cerr << "expected dimension: " << vectors_per_embedding
              << "\ngot dimension: " << query.size() << std::endl;
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<int, size_t>> similarities;
  similarities.reserve(num_vectors);

  const __m256i *query_data = reinterpret_cast<const __m256i *>(query.data());

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
    const std::vector<float> &input, __m256i *output) const {
  for (size_t i = 0; i < vectors_per_embedding; i++) {
    std::vector<int8_t> temp(32, 0);

    for (size_t j = 0; j < 32 && (i * 32 + j) < input.size(); j++) {
      float val = input[i * 32 + j];
      temp[j] = static_cast<int8_t>(std::clamp(val * 127.0f, -127.0f, 127.0f));
    }

    output[i] =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(temp.data()));
  }
}

int OptimizedEmbeddingSearchUint8AVX2::cosine_similarity_optimized(
    const __m256i *vec_a, const __m256i *vec_b) const {
  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();
  __m256i ones = _mm256_set1_epi16(1);

  for (size_t i = 0; i < vectors_per_embedding; i += 2) {
    __m256i a[2], b[2];
    a[0] = _mm256_load_si256(vec_a + i);
    a[1] = _mm256_load_si256(vec_a + i + 1);
    b[0] = _mm256_load_si256(vec_b + i);
    b[1] = _mm256_load_si256(vec_b + i + 1);

    _mm_prefetch(vec_a + i + 2 * 10, _MM_HINT_T0);

    // Process pairs
    __m256i mul_lo[2], mul_hi[2];
    mul_lo[0] =
        _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a[0])),
                           _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b[0])));
    mul_hi[0] = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[0], 1)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[0], 1)));

    mul_lo[1] =
        _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a[1])),
                           _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b[1])));
    mul_hi[1] = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[1], 1)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[1], 1)));

    // Accumulate results
    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(mul_lo[0], ones));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(mul_hi[0], ones));
    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(mul_lo[1], ones));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(mul_hi[1], ones));
  }

  // Handle remaining odd vector
  if (vectors_per_embedding % 2) {
    size_t i = vectors_per_embedding - 1;
    __m256i a = _mm256_load_si256(vec_a + i);
    __m256i b = _mm256_load_si256(vec_b + i);

    __m256i mul_lo =
        _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
                           _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));
    __m256i mul_hi = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)));

    sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(mul_lo, ones));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(mul_hi, ones));
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