// embedding_search_binary.cpp
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
#include "embedding_utils.h"
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <immintrin.h>
#include <omp.h>

bool EmbeddingSearchBinaryAVX2::load(const std::string &filename)
{
    // Binary embeddings are typically created from float embeddings
    // This method could be implemented to directly load binary data if needed
    throw std::runtime_error("Direct loading of binary embeddings not implemented");
}

std::vector<std::pair<int, size_t>> EmbeddingSearchBinaryAVX2::similarity_search(const avx2i_vector &query, size_t k)
{
    if (query.size() != vector_size)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<int, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        int sim = binary_cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<int, size_t>>(similarities.begin(), similarities.begin() + k);
}

bool EmbeddingSearchBinaryAVX2::create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_data)
{
    std::string error_message;
    if (!EmbeddingUtils::validateBinaryAVX2Dimensions(float_data, error_message)) {
        throw std::runtime_error(error_message);
    }

    size_t num_vectors = float_data.size();
    size_t float_vector_size = float_data[0].size();

    // Calculate required vector size with proper rounding
    vector_size = EmbeddingUtils::calculateBinaryAVX2VectorSize(float_vector_size);

    // Resize embeddings vector
    embeddings.clear(); // Clear first to ensure clean state
    embeddings.resize(num_vectors, avx2i_vector(vector_size));

    // Convert all vectors in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_vectors; ++i) {
        EmbeddingUtils::convertSingleFloatToBinaryAVX2(
            float_data[i],
            embeddings[i],
            vector_size
        );
    }

    return true;
}

int popcount_avx2(const __m256i &v)
{
    int result = __builtin_popcountll(_mm256_extract_epi64(v, 0));
    result += __builtin_popcountll(_mm256_extract_epi64(v, 1));
    result += __builtin_popcountll(_mm256_extract_epi64(v, 2));
    result += __builtin_popcountll(_mm256_extract_epi64(v, 3));

    return result;
}

int EmbeddingSearchBinaryAVX2::binary_cosine_similarity(const avx2i_vector &a, const avx2i_vector &b)
{
    int dot_product = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        __m256i result = _mm256_xor_si256(a[i], b[i]);
        __m256i all_ones = _mm256_set1_epi32(-1);
        result = _mm256_xor_si256(result, all_ones);
        dot_product += popcount_avx2(result);
    }
    return dot_product;
}