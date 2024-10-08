#include "embedding_search_uint8_avx2.h"
#include "embedding_io.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

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
    for (size_t i = 0; i < num_embeddings; i++)
    {
        for (size_t j = 0; j < vector_size; j++)
        {
            size_t k = j * 32;
            // embeddings[i][j] = _mm256_loadu_ps(&m[i][k]);

            std::vector<uint8_t> a(32, 0);
            for (int l = 0; l < 32; l++)
            {
                a[l] = (m[i][k + l] + 1) * 8;
                if (a[l] > 15)
                {
                    a[l] = 15;
                }
                std::cout << "a[" << l << "]: " << a[l] << " m: " << (m[i][k + l] + 1) * 8 << "\t| ";
            }
            // embeddings[i][j] = _mm256_loadu_epi8(a.data());
            embeddings[i][j] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a.data()));
            std::cout << std::endl;
        }
        exit(0);
    }

    return true;
}

uint EmbeddingSearchUint8AVX2::cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b)
{
    __m256i dot_product = _mm256_setzero_si256();

    for (size_t i = 0; i < a.size(); ++i)
    {
        //__m256i prod = _mm256_mullo_epi16(a[i], b[i]);
        // dot_product = _mm256_mul(dot_product, prod);
        // mag_a = _mm256_add_ps(mag_a, _mm256_mul_ps(a[i], a[i]));
        // mag_b = _mm256_add_ps(mag_b, _mm256_mul_ps(b[i], b[i]));

        __m256i mult_low = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(a[i], 0)),
                                              _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b[i], 0)));
        __m256i mult_high = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(a[i], 1)),
                                               _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b[i], 1)));

        dot_product = _mm256_add_epi32(dot_product, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mult_low, 0)));
        dot_product = _mm256_add_epi32(dot_product, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mult_low, 1)));
        dot_product = _mm256_add_epi32(dot_product, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mult_high, 0)));
        dot_product = _mm256_add_epi32(dot_product, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mult_high, 1)));
    }

    // Horizontal sum
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(dot_product, 0),
                                   _mm256_extracti128_si256(dot_product, 1));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
    uint sum = _mm_cvtsi128_si32(sum128);

    return sum;
}

/*float EmbeddingSearchUint8AVX2::dot_product_avx2(__m256 a, __m256 b)
{
    __m256 mul = _mm256_mul_ps(a, b);
    return _mm256_reduce_add_ps(mul);
}*/