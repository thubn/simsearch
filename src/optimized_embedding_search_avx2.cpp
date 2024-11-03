#include "optimized_embedding_search_avx2.h"
#include "embedding_utils.h"
#include "embedding_io.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

bool OptimizedEmbeddingSearchAVX2::setEmbeddings(const std::vector<std::vector<float>> &input_vectors)
{
    if (input_vectors.empty() || input_vectors[0].empty())
    {
        return false;
    }

    try
    {
        // Store sentences
        // sentences = input_sentences;

        // Initialize dimensions
        num_vectors = input_vectors.size();
        vector_dim = input_vectors[0].size();
        padded_dim = ((vector_dim + 7) / 8) * 8;
        vectors_per_embedding = padded_dim;

        // Allocate aligned memory for embeddings
        size_t total_size = num_vectors * padded_dim;
        embedding_data.reset(static_cast<float *>(
            std::aligned_alloc(config_.memory.alignmentSize, total_size * sizeof(float))));

        // Initialize norms
        norms.resize(num_vectors);

        // Copy and process embeddings
        for (size_t i = 0; i < num_vectors; i++)
        {
            float *dest = get_embedding_ptr(i);

            // Copy embedding data
            std::memcpy(dest, input_vectors[i].data(), vector_dim * sizeof(float));

            // Zero-pad if necessary
            if (padded_dim > vector_dim)
            {
                std::memset(dest + vector_dim, 0, (padded_dim - vector_dim) * sizeof(float));
            }

            // Compute and store norm
            norms[i] = compute_norm_avx2(dest);
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load vectors: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> OptimizedEmbeddingSearchAVX2::getEmbedding(size_t index) const
{
    if (index >= num_vectors)
    {
        throw std::out_of_range("Embedding index out of range");
    }

    std::vector<float> result(vector_dim);
    const float *embedding = get_embedding_ptr(index);
    std::copy(embedding, embedding + vector_dim, result.begin());
    return result;
}

bool OptimizedEmbeddingSearchAVX2::load(const std::string &filename)
{
    throw std::runtime_error("Direct loading not implemented");
}

std::vector<std::pair<float, size_t>> OptimizedEmbeddingSearchAVX2::similarity_search(const avx2_vector &query, size_t k)
{
    throw std::runtime_error("AVX2 vector input not supported in optimized version");
}

std::vector<std::pair<float, size_t>> OptimizedEmbeddingSearchAVX2::similarity_search(const std::vector<float> &query, size_t k)
{
    if (query.size() != vector_dim)
    {
        throw std::invalid_argument("Invalid query dimension");
    }

    // Prepare aligned query vector
    auto query_aligned = aligned_vector<float>(padded_dim);
    std::memcpy(query_aligned.data(), query.data(), vector_dim * sizeof(float));
    if (padded_dim > vector_dim)
    {
        std::memset(query_aligned.data() + vector_dim, 0, (padded_dim - vector_dim) * sizeof(float));
    }
    // crashes, when called, why?
    float query_norm = compute_norm_avx2(query_aligned.data());

    // Calculate similarities
    std::vector<std::pair<float, size_t>> results;
    results.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; i++)
    {
        float similarity = compute_similarity_avx2(
            get_embedding_ptr(i),
            query_aligned.data(),
            norms[i],
            query_norm);
        results.emplace_back(similarity, i);
    }

    // Partial sort to get top-k results
    if (results.size() > k)
    {
        std::partial_sort(
            results.begin(),
            results.begin() + k,
            results.end(),
            [](const auto &a, const auto &b)
            { return a.first > b.first; });
        results.resize(k);
    }

    return results;
}

bool OptimizedEmbeddingSearchAVX2::validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message)
{
    return EmbeddingUtils::validateAVX2Dimensions(input, error_message);
}

float OptimizedEmbeddingSearchAVX2::compute_norm_avx2(const float *vec) const
{
    __m256 sum = _mm256_setzero_ps();

    for (size_t i = 0; i < padded_dim; i += 8)
    {
        __m256 v = _mm256_load_ps(vec + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }

    // Horizontal sum and square root
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum_128 = _mm_add_ps(hi, lo);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    return std::sqrt(_mm_cvtss_f32(sum_128));
}

inline float OptimizedEmbeddingSearchAVX2::compute_similarity_avx2(const float *vec_a, const float *vec_b, float norm_a, float norm_b) const
{
    __m256 sum = _mm256_setzero_ps();

    // Compute dot product using AVX2
    for (size_t i = 0; i < padded_dim; i += 8)
    {
        __m256 a = _mm256_load_ps(vec_a + i);
        __m256 b = _mm256_load_ps(vec_b + i);
        sum = _mm256_fmadd_ps(a, b, sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum_128 = _mm_add_ps(hi, lo);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    float dot_product = _mm_cvtss_f32(sum_128);
    return dot_product / (norm_a * norm_b);
}