#include "optimized_embedding_search_binary_avx2.h"
#include "embedding_utils.h"
#include <stdexcept>
#include <cstring>

bool OptimizedEmbeddingSearchBinaryAVX2::setEmbeddings(const std::vector<std::vector<float>> &input_vectors)
{
    std::string error_message;
    if (!validateDimensions(input_vectors, error_message))
        throw std::runtime_error(error_message);

    if (!initializeDimensions(input_vectors))
        return false;

    // Calculate padding for binary values (256 bits per AVX2 vector)
    padded_dim = ((vector_dim + 255) / 256) * 256;
    vectors_per_embedding = padded_dim / 256; // Number of __m256i vectors needed per embedding

    // Allocate aligned memory
    size_t total_vectors = num_vectors * vectors_per_embedding;
    if (!allocateAlignedMemory(total_vectors))
        return false;

    // Convert and store each vector
    for (size_t i = 0; i < num_vectors; i++)
    {
        __m256i *dest = get_embedding_ptr(i);
        convert_float_to_binary_avx2(input_vectors[i], dest);
    }

    return true;
}

avx2i_vector OptimizedEmbeddingSearchBinaryAVX2::getEmbeddingAVX2(size_t index) const
{
    if (index >= num_vectors)
    {
        throw std::out_of_range("Embedding index out of range");
    }

    avx2i_vector result(vectors_per_embedding);
    const __m256i *src = get_embedding_ptr(index);

    // Each __m256i contains 256 bits
    for (size_t i = 0; i < vectors_per_embedding; ++i)
    {
        result[i] = _mm256_load_si256(&src[i]);
    }

    return result;
}

std::vector<std::pair<int32_t, size_t>> OptimizedEmbeddingSearchBinaryAVX2::similarity_search(const avx2i_vector &query, size_t k)
{
    if (query.size() != vectors_per_embedding)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<int32_t, size_t>> similarities;
    similarities.reserve(num_vectors);

    const __m256i *query_data = reinterpret_cast<const __m256i *>(query.data());
    AVX2Popcount counter;
    // AVX2PopcountHarleySeal counter;

    for (size_t i = 0; i < num_vectors; i++)
    {
        int32_t sim = compute_similarity_avx2(get_embedding_ptr(i), query_data, counter);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(
        similarities.begin(),
        similarities.begin() + k,
        similarities.end(),
        [](const auto &a, const auto &b)
        { return a.first > b.first; });

    return std::vector<std::pair<int32_t, size_t>>(
        similarities.begin(),
        similarities.begin() + k);
}

bool OptimizedEmbeddingSearchBinaryAVX2::validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message)
{
    return EmbeddingUtils::validateBinaryAVX2Dimensions(input, error_message);
}

void OptimizedEmbeddingSearchBinaryAVX2::convert_float_to_binary_avx2(const std::vector<float> &input, __m256i *output) const
{
    for (size_t i = 0; i < vectors_per_embedding; i++)
    {
        uint64_t bits[4] = {0, 0, 0, 0}; // 256 bits total

        // Process each 64-bit chunk
        for (size_t chunk = 0; chunk < 4; chunk++)
        {
            // Process each bit within the chunk
            for (size_t bit = 0; bit < 64; bit++)
            {
                size_t input_idx = i * 256 + chunk * 64 + bit;
                if (input_idx < input.size() && input[input_idx] >= 0)
                {
                    bits[chunk] |= (1ULL << (63 - bit));
                }
            }
        }

        // Combine into single AVX2 vector
        output[i] = _mm256_set_epi64x(
            bits[3],
            bits[2],
            bits[1],
            bits[0]);
    }
}

// currently hardcoded for 1024bit vectors
int32_t OptimizedEmbeddingSearchBinaryAVX2::compute_similarity_avx2(const __m256i *vec_a, const __m256i *vec_b, AVX2Popcount &counter) const
{
    int32_t total_popcount = 0;

   // prefetch 2 cache lines for 4 vectors 10 loops ahead
    _mm_prefetch(vec_a + 4 * 10, _MM_HINT_T0);
    _mm_prefetch(vec_a + 4 * 10 + 2, _MM_HINT_T0);
    __m256i all_ones = _mm256_set1_epi32(-1);
    __m256i xor_result[4];
    xor_result[0] = _mm256_xor_si256(vec_a[0], vec_b[0]);
    xor_result[0] = _mm256_xor_si256(xor_result[0], all_ones);
    xor_result[1] = _mm256_xor_si256(vec_a[1], vec_b[1]);
    xor_result[1] = _mm256_xor_si256(xor_result[1], all_ones);
    xor_result[2] = _mm256_xor_si256(vec_a[2], vec_b[2]);
    xor_result[2] = _mm256_xor_si256(xor_result[2], all_ones);
    xor_result[3] = _mm256_xor_si256(vec_a[3], vec_b[3]);
    xor_result[3] = _mm256_xor_si256(xor_result[3], all_ones);

    // popcnt lookup is faster than harley seal
    return counter.popcnt_AVX2_lookup(reinterpret_cast<const uint8_t *>(xor_result), 4 * sizeof(__m256i));
}