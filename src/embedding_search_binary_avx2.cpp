// embedding_search_binary.cpp
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <immintrin.h>

bool EmbeddingSearchBinaryAVX2::load(const std::string &filename)
{
    // Binary embeddings are typically created from float embeddings
    // This method could be implemented to directly load binary data if needed
    throw std::runtime_error("Direct loading of binary embeddings not implemented");
}

std::vector<std::pair<int, size_t>> EmbeddingSearchBinaryAVX2::similarity_search(const std::vector<__m256i> &query, size_t k)
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
    size_t num_vectors = float_data.size();
    size_t float_vector_size = float_data[0].size();

    vector_size = ((float_vector_size + 63) / 64) / 4; // Round up to nearest multiple of 64
    embeddings.resize(num_vectors, std::vector<__m256i>(vector_size));

    std::vector<std::vector<uint64_t>> embeddingsU64;
    embeddingsU64.resize(num_vectors, std::vector<uint64_t>(vector_size * 4));

    for (size_t i = 0; i < num_vectors; ++i)
    {
        for (size_t j = 0; j < float_vector_size; j++)
        {
            if (float_data[i][j] >= 0)
            {
                embeddingsU64[i][j / 64] |= (1ULL << (63 - (j % 64)));
            }
        }
    }
    for (size_t i = 0; i < num_vectors; i++)
    {
        for (size_t j = 0; j < vector_size; j++)
        {
            int k = j * 4;
            embeddings[i][j] = _mm256_setr_epi64x(embeddingsU64[i][k], embeddingsU64[i][k + 1], embeddingsU64[i][k + 2], embeddingsU64[i][k + 3]);
        }
    }

    // sentences = float_embeddings.getSentences();
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

int EmbeddingSearchBinaryAVX2::binary_cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b)
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