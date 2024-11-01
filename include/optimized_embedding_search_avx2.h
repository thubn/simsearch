// optimized_embedding_search_avx2.h
#pragma once
#include "embedding_search_base.h"
#include "embedding_io.h"
#include "aligned_types.h"
#include <immintrin.h>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <string>
#include <iostream>
#include <cmath>

class OptimizedEmbeddingSearchAVX2 : public EmbeddingSearchBase<avx2_vector, float>
{
public:
    OptimizedEmbeddingSearchAVX2() : num_vectors(0), vector_dim(0), padded_dim(0) {}

    bool load(const std::string &filename)
    {
        // Binary embeddings are typically created from float embeddings
        // This method could be implemented to directly load binary data if needed
        throw std::runtime_error("Direct loading of binary embeddings not implemented");
    }

    bool load_from_vectors(const std::vector<std::vector<float>> &input_vectors /*,
                           const std::vector<std::string>& input_sentences*/
    )
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

            // Allocate aligned memory for embeddings
            size_t total_size = num_vectors * padded_dim;
            embedding_data.reset(static_cast<float *>(
                std::aligned_alloc(AVX2_ALIGNMENT, total_size * sizeof(float))));

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

    // Add the proper override of the pure virtual function
    std::vector<std::pair<float, size_t>> similarity_search(const avx2_vector &query, size_t k) override
    {
        throw std::runtime_error("AVX2 vector input not supported in optimized version");
    }

    std::vector<std::pair<float, size_t>> similarity_search(
        const std::vector<float> &query,
        size_t k)
    {
        if (query.size() != vector_dim)
        {
            throw std::invalid_argument("Invalid query dimension");
        }

        // Prepare aligned query vector
        auto query_aligned = std::make_unique<float[]>(padded_dim);
        std::memcpy(query_aligned.get(), query.data(), vector_dim * sizeof(float));
        if (padded_dim > vector_dim)
        {
            std::memset(query_aligned.get() + vector_dim, 0, (padded_dim - vector_dim) * sizeof(float));
        }
        float query_norm = compute_norm_avx2(query_aligned.get());

        // Calculate similarities
        std::vector<std::pair<float, size_t>> results;
        results.reserve(num_vectors);

        // Process in batches for better cache utilization
        constexpr size_t BATCH_SIZE = 256;

        for (size_t batch_start = 0; batch_start < num_vectors; batch_start += BATCH_SIZE)
        {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, num_vectors);

            for (size_t i = batch_start; i < batch_end; i++)
            {
                float similarity = compute_similarity_avx2(
                    get_embedding_ptr(i),
                    query_aligned.get(),
                    norms[i],
                    query_norm);
                results.emplace_back(similarity, i);
            }
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

    std::vector<std::vector<float>> getEmbeddings() const
    {
        std::vector<std::vector<float>> result(num_vectors, std::vector<float>(vector_dim));

        // Convert from our contiguous memory layout back to vector of vectors
        for (size_t i = 0; i < num_vectors; i++)
        {
            const float *embedding = get_embedding_ptr(i);
            std::copy(embedding, embedding + vector_dim, result[i].begin());
        }

        return result;
    }

    std::vector<float> getEmbedding(size_t index) const
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

    const std::vector<std::string> &getSentences() const { return sentences; }

    bool isInitialized()
    {
        return num_vectors > 0;
    }

private:
    static constexpr size_t AVX2_ALIGNMENT = 32;

    std::unique_ptr<float[]> embedding_data;
    std::vector<std::string> sentences;
    std::vector<float> norms;

    size_t num_vectors;
    size_t vector_dim;
    size_t padded_dim;

    float *get_embedding_ptr(size_t index)
    {
        return embedding_data.get() + index * padded_dim;
    }

    const float *get_embedding_ptr(size_t index) const
    {
        return embedding_data.get() + index * padded_dim;
    }

    float compute_norm_avx2(const float *vec) const
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

    float compute_similarity_avx2(const float *vec_a, const float *vec_b,
                                  float norm_a, float norm_b) const
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
};