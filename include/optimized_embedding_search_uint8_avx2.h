// optimized_embedding_search_uint8_avx2.h
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

class OptimizedEmbeddingSearchUint8AVX2 : public EmbeddingSearchBase<avx2i_vector, uint>
{
public:
    OptimizedEmbeddingSearchUint8AVX2() : num_vectors(0), vector_dim(0), padded_dim(0) {}

    bool load(const std::string &filename) override
    {
        throw std::runtime_error("Direct loading not implemented");
    }

    bool setEmbeddings(const std::vector<std::vector<float>> &input_vectors)
    {
        if (input_vectors.empty() || input_vectors[0].empty())
        {
            return false;
        }

        try
        {
            // Initialize dimensions
            num_vectors = input_vectors.size();
            vector_dim = input_vectors[0].size();
            // Each AVX2 vector can store 32 int8 values
            padded_dim = ((vector_dim + 31) / 32) * 32;
            values_per_avx = 32;
            avx_vectors_per_embedding = padded_dim / values_per_avx;

            // Allocate aligned memory for embeddings
            size_t total_size = num_vectors * avx_vectors_per_embedding;
            embedding_data.reset(static_cast<__m256i *>(
                std::aligned_alloc(AVX2_ALIGNMENT, total_size * sizeof(__m256i))));

            // Convert and store each vector
            for (size_t i = 0; i < num_vectors; i++)
            {
                __m256i *dest = get_embedding_ptr(i);
                convert_float_to_uint8_avx2(input_vectors[i], dest);
            }

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to convert vectors: " << e.what() << std::endl;
            return false;
        }
    }

    // Override for the virtual function
    std::vector<std::pair<uint, size_t>> similarity_search(const avx2i_vector &query, size_t k) override
    {
        throw std::runtime_error("AVX2 vector input not supported in optimized version");
    }

    std::vector<std::pair<uint, size_t>> similarity_search(
        const std::vector<float> &query,
        size_t k)
    {
        if (query.size() != vector_dim)
        {
            throw std::invalid_argument("Invalid query dimension");
        }

        // Convert query to uint8 AVX2 format
        auto query_aligned = std::make_unique<__m256i[]>(avx_vectors_per_embedding);
        convert_float_to_uint8_avx2(query, query_aligned.get());

        // Calculate similarities
        std::vector<std::pair<uint, size_t>> results;
        results.reserve(num_vectors);

        // Process in batches for better cache utilization
        constexpr size_t BATCH_SIZE = 256;

        for (size_t batch_start = 0; batch_start < num_vectors; batch_start += BATCH_SIZE)
        {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, num_vectors);

            for (size_t i = batch_start; i < batch_end; i++)
            {
                uint similarity = compute_similarity_avx2(
                    get_embedding_ptr(i),
                    query_aligned.get());
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

    std::vector<std::vector<int8_t>> getEmbeddings() const
    {
        std::vector<std::vector<int8_t>> result(num_vectors,
                                                std::vector<int8_t>(vector_dim));

        for (size_t i = 0; i < num_vectors; i++)
        {
            const __m256i *embedding = get_embedding_ptr(i);
            convert_avx2_to_uint8(embedding, result[i]);
        }

        return result;
    }

    std::vector<int8_t> getEmbedding(size_t index) const
    {
        if (index >= num_vectors)
        {
            throw std::out_of_range("Embedding index out of range");
        }

        std::vector<int8_t> result(vector_dim);
        const __m256i *embedding = get_embedding_ptr(index);
        convert_avx2_to_uint8(embedding, result);
        return result;
    }

    avx2i_vector getEmbeddingAVX2(size_t index) const
    {
        if (index >= num_vectors)
        {
            throw std::out_of_range("Embedding index out of range");
        }

        avx2i_vector result(avx_vectors_per_embedding);
        const __m256i *embedding = get_embedding_ptr(index);
        std::memcpy(result.data(), embedding, avx_vectors_per_embedding * sizeof(__m256i));
        return result;
    }

    bool isInitialized() const
    {
        return num_vectors > 0;
    }

    size_t getNumVectors() const { return num_vectors; }
    size_t getVectorDim() const { return vector_dim; }
    size_t getPaddedDim() const { return padded_dim; }
    size_t getAVXVectorsPerEmbedding() const { return avx_vectors_per_embedding; }

private:
    static constexpr size_t AVX2_ALIGNMENT = 32;

    std::unique_ptr<__m256i[]> embedding_data;
    //std::vector<std::string> sentences;

    size_t num_vectors;
    size_t vector_dim;                // Original dimension
    size_t padded_dim;                // Padded dimension (multiple of 32)
    size_t values_per_avx;            // Number of int8 values per AVX2 vector (32)
    size_t avx_vectors_per_embedding; // Number of AVX2 vectors per embedding

    __m256i *get_embedding_ptr(size_t index)
    {
        return embedding_data.get() + index * avx_vectors_per_embedding;
    }

    const __m256i *get_embedding_ptr(size_t index) const
    {
        return embedding_data.get() + index * avx_vectors_per_embedding;
    }

    void convert_float_to_uint8_avx2(const std::vector<float> &input, __m256i *output)
    {
        for (size_t i = 0; i < avx_vectors_per_embedding; i++)
        {
            std::vector<int8_t> temp(32, 0);
            for (size_t j = 0; j < 32 && (i * 32 + j) < input.size(); j++)
            {
                float val = input[i * 32 + j];
                temp[j] = static_cast<int8_t>(val * 127.0f);
            }
            output[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(temp.data()));
        }
    }

    void convert_avx2_to_uint8(const __m256i *avx2_data, std::vector<int8_t> &uint8_output) const
    {
        for (size_t i = 0; i < avx_vectors_per_embedding; i++)
        {
            int8_t temp[32];
            _mm256_storeu_si256((__m256i *)temp, avx2_data[i]);

            for (size_t j = 0; j < 32 && (i * 32 + j) < vector_dim; j++)
            {
                uint8_output[i * 32 + j] = temp[j];
            }
        }
    }

    uint compute_similarity_avx2(const __m256i *vec_a, const __m256i *vec_b) const
    {
        __m256i sum_lo = _mm256_setzero_si256();
        __m256i sum_hi = _mm256_setzero_si256();

        for (size_t i = 0; i < avx_vectors_per_embedding; i++)
        {
            // Load vectors
            __m256i a = vec_a[i];
            __m256i b = vec_b[i];

            // Convert int8 to int16 and multiply
            __m256i mul_lo = _mm256_mullo_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));

            __m256i mul_hi = _mm256_mullo_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)));

            // Accumulate products
            sum_lo = _mm256_add_epi32(sum_lo,
                                      _mm256_madd_epi16(mul_lo, _mm256_set1_epi16(1)));
            sum_hi = _mm256_add_epi32(sum_hi,
                                      _mm256_madd_epi16(mul_hi, _mm256_set1_epi16(1)));
        }

        // Combine results
        __m256i sum = _mm256_add_epi32(sum_lo, sum_hi);

        // Horizontal sum
        __m128i sum_128 = _mm_add_epi32(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256(sum, 1));
        sum_128 = _mm_add_epi32(sum_128,
                                _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum_128 = _mm_add_epi32(sum_128,
                                _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(2, 3, 0, 1)));

        return _mm_cvtsi128_si32(sum_128);
    }
};