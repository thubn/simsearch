#include "embedding_search.h"
#include "embedding_search_avx2.h"
#include "util.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <vector>
#include <bitset>
#include <immintrin.h>
#include <nmmintrin.h>
#include <x86gprintrin.h>

int popcount_avx2(const __m256i &v)
{
    int result = _popcnt64(_mm256_extract_epi64(v, 0));
    result += _popcnt64(_mm256_extract_epi64(v, 1));
    result += _popcnt64(_mm256_extract_epi64(v, 2));
    result += _popcnt64(_mm256_extract_epi64(v, 3));

    return result;
}

float dot_product_avx2(__m256 a, __m256 b)
{
    // Multiply the vectors
    __m256 mul = _mm256_mul_ps(a, b);

    // Horizontally add the products
    __m256 sum = _mm256_hadd_ps(mul, mul);
    sum = _mm256_hadd_ps(sum, sum);

    // At this point, the first and fifth float in sum contain the sum of all products
    // Extract these values and add them
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 result = _mm_add_ss(sum_low, sum_high);

    // Extract the final scalar result
    return _mm_cvtss_f32(result);
}

void print_m256i_binary(__m256i v)
{
    alignas(32) uint64_t values[4];
    _mm256_store_si256((__m256i *)values, v);
    for (int i = 0; i < 4; ++i)
    {
        std::cout << std::bitset<64>(values[i]) << "\n";
    }
    std::cout << std::endl;
}

EmbeddingSearchAvx2::EmbeddingSearchAvx2() : vector_size(0) {}

const std::vector<std::vector<__m256>> &EmbeddingSearchAvx2::getEmbeddings()
{
    return embeddings;
}

const std::vector<std::vector<__m256i>> &EmbeddingSearchAvx2::getBinaryEmbeddings()
{
    return binary_embeddings;
}

const std::vector<std::string> &EmbeddingSearchAvx2::getSentences()
{
    return sentences;
}

const size_t &EmbeddingSearchAvx2::getVectorSize()
{
    return vector_size;
}

bool EmbeddingSearchAvx2::setEbeddings(const std::vector<std::vector<float>> &m)
{
    uint org_vector_size = m[0].size();
    vector_size = org_vector_size / 8;
    uint num_embeddings = m.size();

    embeddings.resize(num_embeddings, std::vector<__m256>(vector_size));
    for (int i = 0; i < num_embeddings; i++)
    {
        for (int j = 0; j < vector_size; j++)
        {
            int k = j * 8;
            embeddings[i][j] = _mm256_set_ps(m[i][k], m[i][k + 1], m[i][k + 2], m[i][k + 3], m[i][k + 4], m[i][k + 5], m[i][k + 6], m[i][k + 7]);
        }
    }

    return true;
}

bool EmbeddingSearchAvx2::setBinaryEbeddings(const std::vector<std::vector<uint64_t>> &m)
{
    uint org_vector_size = m[0].size();
    uint binary_vector_size = org_vector_size / 4;
    uint num_embeddings = m.size();

    binary_embeddings.resize(num_embeddings, std::vector<__m256i>(binary_vector_size));
    for (int i = 0; i < num_embeddings; i++)
    {
        for (int j = 0; j < binary_vector_size; j++)
        {
            int k = j * 4;
            binary_embeddings[i][j] = _mm256_setr_epi64x(m[i][k], m[i][k + 1], m[i][k + 2], m[i][k + 3]);
        }
    }

    return true;
}

std::vector<std::pair<float, size_t>> EmbeddingSearchAvx2::similarity_search(const std::vector<__m256> &query, size_t k)
{
    if (query.size() != vector_size)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        float sim = cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    // std::vector<size_t> result;
    std::vector<std::pair<float, size_t>> result;
    result.reserve(k);
    for (size_t i = 0; i < k && i < similarities.size(); ++i)
    {
        result.push_back(similarities[i]);
    }

    return result;
}

float EmbeddingSearchAvx2::cosine_similarity(const std::vector<__m256> &a, const std::vector<__m256> &b)
{
    float dot_product = 0.0f;
    // float mag_a = 0.0f;
    // float mag_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += dot_product_avx2(a[i], b[i]);
        // mag_a += dot_product_avx2(a[i], a[i]);
        // mag_b += dot_product_avx2(b[i], b[i]);
    }

    // return dot_product / (std::sqrt(mag_a) * std::sqrt(mag_b));
    return dot_product;
}

/*
bool EmbeddingSearchAvx2::create_binary_embedding_from_float()
{
    int intsize = sizeof(uint64_t) * 8;
    binary_embeddings.resize(embeddings.size(), std::vector<__m256i>(vector_size / intsize));

    int size = binary_embeddings.size();

    // loop embeddings (<=> vectors)
    for (int i = 0; i < size; i++)
    {
        // loop uint64_t vector components
        for (int j = 0; j < vector_size / intsize; j++)
        {
            // loop bits of uint64_t vector component
            for (int k = 0; k < intsize; k++)
            {
                uint32_t intBits = *reinterpret_cast<uint32_t *>(&embeddings[i][(intsize * j) + k]);
                bool signBit = intBits >> 31;
                signBit = !signBit;
                // binary_embeddings[i][j] |= static_cast<uint64_t>(signBit) << (intsize - k - 1);
            }
        }
    }
    return true;
}
*/

std::vector<std::pair<int, size_t>> EmbeddingSearchAvx2::binary_similarity_search(const std::vector<__m256i> &query, size_t k)
{
    if (query.size() != vector_size / 32)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<int, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        int sim = binary_cosine_similarity(query, binary_embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    // std::vector<size_t> result;
    std::vector<std::pair<int, size_t>> result;
    result.reserve(k);
    for (size_t i = 0; i < k && i < similarities.size(); ++i)
    {
        result.push_back(similarities[i]);
    }

    return result;
}

int EmbeddingSearchAvx2::binary_cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b)
{
    int dot_product = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        __m256i result = _mm256_xor_si256(a[i], b[i]);
        __m256i all_ones = _mm256_set1_epi32(-1);
        result = _mm256_xor_si256(result, all_ones);
        dot_product += popcount_avx2(result);
    }
    return dot_product;
}
