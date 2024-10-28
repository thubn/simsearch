// embedding_search_binary.cpp
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
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

std::vector<__m256i> EmbeddingSearchBinaryAVX2::floatToBinaryAvx2(const std::vector<float> &v)
{
    // Calculate sizes
    size_t u64Size = (v.size() + 63) / 64; // Round up division to handle partial groups
    size_t avxSize = (u64Size + 3) / 4;    // Round up for AVX2 vectors (4 uint64_t per __m256i)

    // Initialize vectors with proper size
    std::vector<uint64_t> vU64(u64Size, 0); // Initialize with zeros
    std::vector<__m256i> result(avxSize);

    for (size_t j = 0; j < v.size(); j++)
    {
        if (v[j] >= 0)
        {
            vU64[j / 64] |= (1ULL << (63 - (j % 64)));
        }
    }
    for (size_t j = 0; j < avxSize; j++)
    {
        int k = j * 4;
        result[j] = _mm256_setr_epi64x(vU64[k], vU64[k + 1], vU64[k + 2], vU64[k + 3]);
    }
    return result;
}

bool EmbeddingSearchBinaryAVX2::create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_data)
{
    size_t num_vectors = float_data.size();
    size_t float_vector_size = float_data[0].size();

    vector_size = ((float_vector_size + 63) / 64) / 4; // Round up to nearest multiple of 64
    embeddings.resize(num_vectors, std::vector<__m256i>(vector_size));

    std::vector<std::vector<uint64_t>> embeddingsU64;
    embeddingsU64.resize(num_vectors, std::vector<uint64_t>(vector_size * 4));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_vectors; ++i)
    {
        embeddings[i] = floatToBinaryAvx2(float_data[i]);
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