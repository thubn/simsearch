#include "embedding_search_uint8_avx2.h"
#include "embedding_io.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <omp.h>

bool EmbeddingSearchUint8AVX2::load(const std::string &filename)
{
    // Binary embeddings are typically created from float embeddings
    // This method could be implemented to directly load binary data if needed
    throw std::runtime_error("Direct loading of binary embeddings not implemented");
}

std::vector<std::pair<uint, size_t>> EmbeddingSearchUint8AVX2::similarity_search(const std::vector<__m256i> &query, size_t k)
{
    if (query.size() != embeddings[0].size())
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<uint, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        uint sim = cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<uint, size_t>>(similarities.begin(), similarities.begin() + k);
}

std::vector<std::pair<uint, size_t>> EmbeddingSearchUint8AVX2::similarity_search(const std::vector<__m256i> &query, size_t k, std::vector<std::pair<int, size_t>> &searchIndexes)
{
    if (query.size() != embeddings[0].size())
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<uint, size_t>> similarities;
    similarities.reserve(searchIndexes.size());

    for (int i = 0; i < searchIndexes.size(); i++)
    {
        uint sim = cosine_similarity(query, embeddings[searchIndexes[i].second]);
        similarities.emplace_back(sim, searchIndexes[i].second);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<uint, size_t>>(similarities.begin(), similarities.begin() + k);
}

std::vector<__m256i> EmbeddingSearchUint8AVX2::floatToAvx2(const std::vector<float> &v)
{
    std::vector<__m256i> result(v.size() / 32);
    for (size_t j = 0; j < v.size() / 32; j++)
    {
        size_t k = j * 32;
        // embeddings[i][j] = _mm256_loadu_ps(&m[i][k]);

        std::vector<int8_t> a(32, 0);
        for (int l = 0; l < 32; l++)
        {
            // JI: 0.756784 (sample size: 1000)
            a[l] = v[k + l] * 127;

            // JI: 0.530709 (sample size: 250) (exactly the same as binary?!? why?)
            /*
            float n = 1 / 4;
            float x = m[i][k + l];
            a[l] = std::copysign(std::pow(std::abs(x), n), x) * 127;
            */

            // JI: 0.538676 (sample size: 250)
            /*
            float val = m[i][k + l];
            uint32_t* floatBits = reinterpret_cast<uint32_t*>(&val);
            a[l] = static_cast<int8_t>(*floatBits >> 24);
            */

            // std::cout << "a[" << l << "]: " << (int)a[l] << " m: " << (m[i][k + l] + 1) * 8 << "\t| ";
        }
        // embeddings[i][j] = _mm256_loadu_epi8(a.data());
        result[j] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a.data()));
        // std::cout << std::endl;
    }
    // exit(0);
}

bool EmbeddingSearchUint8AVX2::setEmbeddings(const std::vector<std::vector<float>> &m)
{
    if (m.empty() || m[0].size() % 8 != 0)
    {
        throw std::runtime_error("Input vector size must be a multiple of 8");
    }

    size_t org_vector_size = m[0].size();
    vector_size = org_vector_size / 32;
    size_t num_embeddings = m.size();

    embeddings.resize(num_embeddings, std::vector<__m256i>(vector_size));
//#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_embeddings; i++)
    {
        embeddings[i] = floatToAvx2(m[i]);
    }

    return true;
}

uint EmbeddingSearchUint8AVX2::cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Vectors must have the same size");
    }

    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();

    for (size_t i = 0; i < a.size(); ++i)
    {
        // Multiply and add
        __m256i mul_lo = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[i], 0)),
                                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[i], 0)));
        __m256i mul_hi = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a[i], 1)),
                                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b[i], 1)));

        sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(mul_lo, _mm256_set1_epi16(1)));
        sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(mul_hi, _mm256_set1_epi16(1)));
    }

    // Combine the results
    __m256i sum = _mm256_add_epi32(sum_lo, sum_hi);

    // Horizontal sum
    __m128i sum_128 = _mm_add_epi32(_mm256_extracti128_si256(sum, 0),
                                    _mm256_extracti128_si256(sum, 1));
    sum_128 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum_128 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(2, 3, 0, 1)));

    int32_t result = _mm_cvtsi128_si32(sum_128);

    // Scale the result to match float32 range
    return result;
}

/*float EmbeddingSearchUint8AVX2::dot_product_avx2(__m256 a, __m256 b)
{
    __m256 mul = _mm256_mul_ps(a, b);
    return _mm256_reduce_add_ps(mul);
}*/