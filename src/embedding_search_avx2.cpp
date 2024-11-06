#include "embedding_search_avx2.h"
#include "embedding_io.h"
#include "embedding_utils.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <omp.h>
#include <iostream>

std::vector<std::pair<float, size_t>> EmbeddingSearchAVX2::similarity_search(const avx2_vector &query, size_t k)
{
    if (query.size() != embeddings[0].size())
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

    return std::vector<std::pair<float, size_t>>(similarities.begin(), similarities.begin() + k);
}

std::vector<std::pair<float, size_t>> EmbeddingSearchAVX2::similarity_search(const avx2_vector &query, size_t k, std::vector<std::pair<int, size_t>> &searchIndexes)
{
    if (query.size() != embeddings[0].size())
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(searchIndexes.size());

    for (size_t i = 0; i < searchIndexes.size(); i++)
    {
        float sim = cosine_similarity(query, embeddings[searchIndexes[i].second]);
        similarities.emplace_back(sim, searchIndexes[i].second);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<float, size_t>>(similarities.begin(), similarities.begin() + k);
}

bool EmbeddingSearchAVX2::setEmbeddings(const std::vector<std::vector<float>> &m)
{
    std::string error_message;
    if (!validateDimensions(m, error_message))
    {
        throw std::runtime_error(error_message);
    }

    vector_size = m[0].size() / 8;
    num_vectors = m.size();
    embeddings.clear();
    EmbeddingUtils::convertEmbeddingsToAVX2(m, embeddings, vector_size);
    return true;
}

bool EmbeddingSearchAVX2::validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message)
{
    return EmbeddingUtils::validateAVX2Dimensions(input, error_message);
}

// Helper function to sum up all elements in an __m256
inline float _mm256_reduce_add_ps(__m256 x)
{
    __m128 high128 = _mm256_extractf128_ps(x, 1);
    __m128 low128 = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(high128, low128);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

float EmbeddingSearchAVX2::cosine_similarity(const avx2_vector &a, const avx2_vector &b)
{
    __m256 dot_product = _mm256_setzero_ps();
    __m256 mag_a = _mm256_setzero_ps();
    __m256 mag_b = _mm256_setzero_ps();

    for (size_t i = 0; i < a.size(); ++i)
    {
        __m256 prod = _mm256_mul_ps(a[i], b[i]);
        dot_product = _mm256_add_ps(dot_product, prod);
        mag_a = _mm256_add_ps(mag_a, _mm256_mul_ps(a[i], a[i]));
        mag_b = _mm256_add_ps(mag_b, _mm256_mul_ps(b[i], b[i]));
    }

    float dot_product_sum = _mm256_reduce_add_ps(dot_product);
    float mag_a_sum = _mm256_reduce_add_ps(mag_a);
    float mag_b_sum = _mm256_reduce_add_ps(mag_b);

    return dot_product_sum / (std::sqrt(mag_a_sum) * std::sqrt(mag_b_sum));
}

float EmbeddingSearchAVX2::dot_product_avx2(__m256 a, __m256 b)
{
    __m256 mul = _mm256_mul_ps(a, b);
    return _mm256_reduce_add_ps(mul);
}