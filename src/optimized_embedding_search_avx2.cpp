#include "optimized_embedding_search_avx2.h"
#include "embedding_utils.h"
#include "embedding_io.h"
#include <stdexcept>
#include <cstring>

bool OptimizedEmbeddingSearchAVX2::setEmbeddings(const std::vector<std::vector<float>> &input_vectors)
{
    std::string error_message;
    if (!validateDimensions(input_vectors, error_message))
        throw std::runtime_error(error_message);

    if (!initializeDimensions(input_vectors))
        return false;

    // Calculate padding and allocation size
    padded_dim = ((vector_dim + 7) / 8) * 8;
    size_t total_size = num_vectors * padded_dim;
    vectors_per_embedding = padded_dim / 8;

    // Allocate aligned memory
    if (!allocateAlignedMemory(total_size))
        return false;

    norms.resize(num_vectors);

    // Copy and pad embeddings
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

        norms[i] = compute_norm_avx2(dest);
    }
    return true;
}

avx2_vector OptimizedEmbeddingSearchAVX2::getEmbeddingAVX2(size_t index) const
{
    if (index >= num_vectors)
    {
        throw std::out_of_range("Embedding index out of range");
    }

    avx2_vector result(vectors_per_embedding);
    const float *src = get_embedding_ptr(index);

    for (size_t i = 0; i < vectors_per_embedding; ++i)
    {
        // Load 8 floats (32 bytes) at a time using aligned load
        result[i] = _mm256_load_ps(src + i * 8);
    }

    return result;
}

bool OptimizedEmbeddingSearchAVX2::load(const std::string &filename)
{
    throw std::runtime_error("Direct loading not implemented");
}

std::vector<std::pair<float, size_t>> OptimizedEmbeddingSearchAVX2::similarity_search(const avx2_vector &query, size_t k)
{
    if (query.size() * 8 != vector_dim)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(num_vectors);

    // Compute query norm once
    float query_norm = compute_norm_avx2(reinterpret_cast<const float *>(query.data()));

    for (size_t i = 0; i < num_vectors; i++)
    {
        float *vec = get_embedding_ptr(i);
        float norm = compute_norm_avx2(vec);
        float sim = compute_similarity_avx2(vec,
                                            reinterpret_cast<const float *>(query.data()),
                                            norm, query_norm);

        similarities.emplace_back(sim, i);
    }

    // Partial sort to get top-k results
    std::partial_sort(similarities.begin(),
                      similarities.begin() + k,
                      similarities.end(),
                      [](const auto &a, const auto &b)
                      {
                          return a.first > b.first;
                      });

    return std::vector<std::pair<float, size_t>>(
        similarities.begin(),
        similarities.begin() + k);
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

float OptimizedEmbeddingSearchAVX2::compute_similarity_avx2(const float *vec_a, const float *vec_b, float norm_a, float norm_b) const
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