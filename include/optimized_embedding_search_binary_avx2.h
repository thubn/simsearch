// optimized_embedding_search_binary_avx2.h
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

class OptimizedEmbeddingSearchBinaryAVX2 : public EmbeddingSearchBase<avx2i_vector, int>
{
public:
    OptimizedEmbeddingSearchBinaryAVX2() : num_vectors(0), vector_dim(0), padded_dim(0) {}

    bool load(const std::string &filename) override
    {
        throw std::runtime_error("Direct loading of binary embeddings not implemented");
    }

    bool create_binary_embedding_from_float(const std::vector<std::vector<float>> &input_vectors)
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
            // Each AVX2 vector (__m256i) can store 256 bits
            padded_dim = ((vector_dim + 255) / 256) * 256;
            bits_per_avx = 256;
            avx_vectors_per_embedding = padded_dim / bits_per_avx;

            // Allocate aligned memory for embeddings
            size_t total_size = num_vectors * avx_vectors_per_embedding;
            embedding_data.reset(static_cast<__m256i *>(
                std::aligned_alloc(AVX2_ALIGNMENT, total_size * sizeof(__m256i))));

            // Process each input vector
            for (size_t i = 0; i < num_vectors; i++)
            {
                __m256i *dest = get_embedding_ptr(i);
                convert_float_to_binary_avx2(input_vectors[i], dest);
            }

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to convert vectors: " << e.what() << std::endl;
            return false;
        }
    }

    // Add the proper override of the pure virtual function
    std::vector<std::pair<int, size_t>> similarity_search(const avx2i_vector &query, size_t k) override
    {
        throw std::runtime_error("AVX2 vector input not supported in optimized version");
    }

    std::vector<std::pair<int, size_t>> similarity_search(
        const std::vector<float> &query,
        size_t k)
    {
        if (query.size() != vector_dim)
        {
            throw std::invalid_argument("Invalid query dimension");
        }

        // Convert query to binary AVX2 format
        auto query_aligned = std::make_unique<__m256i[]>(avx_vectors_per_embedding);
        convert_float_to_binary_avx2(query, query_aligned.get());

        // Calculate similarities
        std::vector<std::pair<int, size_t>> results;
        results.reserve(num_vectors);

        // Process in batches for better cache utilization
        constexpr size_t BATCH_SIZE = 256;

        for (size_t batch_start = 0; batch_start < num_vectors; batch_start += BATCH_SIZE)
        {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, num_vectors);

            for (size_t i = batch_start; i < batch_end; i++)
            {
                int similarity = compute_similarity_avx2(
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

    bool isInitialized() const
    {
        return num_vectors > 0;
    }

    std::vector<std::vector<uint64_t>> getEmbeddings() const
    {
        std::vector<std::vector<uint64_t>> result(num_vectors,
                                                  std::vector<uint64_t>((vector_dim + 63) / 64)); // Round up to nearest 64 bits

        // Convert from AVX2 format back to regular binary format
        for (size_t i = 0; i < num_vectors; i++)
        {
            const __m256i *embedding = get_embedding_ptr(i);
            convert_avx2_to_binary(embedding, result[i]);
        }

        return result;
    }

    std::vector<uint64_t> getEmbedding(size_t index) const
    {
        if (index >= num_vectors)
        {
            throw std::out_of_range("Embedding index out of range");
        }

        std::vector<uint64_t> result((vector_dim + 63) / 64); // Round up to nearest 64 bits
        const __m256i *embedding = get_embedding_ptr(index);
        convert_avx2_to_binary(embedding, result);
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

    size_t getNumVectors() const { return num_vectors; }
    size_t getVectorDim() const { return vector_dim; }
    size_t getPaddedDim() const { return padded_dim; }
    size_t getAVXVectorsPerEmbedding() const { return avx_vectors_per_embedding; }

private:
    static constexpr size_t AVX2_ALIGNMENT = 32;

    std::unique_ptr<__m256i[]> embedding_data;
    std::vector<std::string> sentences;

    size_t num_vectors;
    size_t vector_dim;                // Original dimension
    size_t padded_dim;                // Padded dimension (multiple of 256 bits)
    size_t bits_per_avx;              // Number of bits per AVX2 vector (256)
    size_t avx_vectors_per_embedding; // Number of AVX2 vectors per embedding

    __m256i *get_embedding_ptr(size_t index)
    {
        return embedding_data.get() + index * avx_vectors_per_embedding;
    }

    const __m256i *get_embedding_ptr(size_t index) const
    {
        return embedding_data.get() + index * avx_vectors_per_embedding;
    }

    void convert_float_to_binary_avx2(const std::vector<float> &input, __m256i *output)
    {
        // Process 256 bits at a time
        for (size_t i = 0; i < avx_vectors_per_embedding; i++)
        {
            uint64_t bits[4] = {0, 0, 0, 0}; // 256 bits total (4 * 64)

            // Process each 64-bit chunk
            for (size_t chunk = 0; chunk < 4; chunk++)
            {
                // Process each bit within the chunk
                for (size_t bit = 0; bit < 64; bit++)
                {
                    size_t input_idx = i * 256 + chunk * 64 + bit;
                    if (input_idx < input.size() && input[input_idx] >= 0)
                    {
                        bits[chunk] |= (1ULL << (63 - bit));
                    }
                }
            }

            // Combine the four 64-bit values into one 256-bit AVX2 vector
            output[i] = _mm256_set_epi64x(
                bits[3],
                bits[2],
                bits[1],
                bits[0]);
        }
    }

    int compute_similarity_avx2(const __m256i *vec_a, const __m256i *vec_b) const
    {
        int total_popcount = 0;

        for (size_t i = 0; i < avx_vectors_per_embedding; i++)
        {
            // XOR the vectors and invert to count matching bits
            __m256i xor_result = _mm256_xor_si256(vec_a[i], vec_b[i]);
            __m256i all_ones = _mm256_set1_epi32(-1);
            __m256i match_bits = _mm256_xor_si256(xor_result, all_ones);

            // Count set bits (population count)
            uint64_t *match_ptr = (uint64_t *)&match_bits;
            total_popcount += __builtin_popcountll(match_ptr[0]);
            total_popcount += __builtin_popcountll(match_ptr[1]);
            total_popcount += __builtin_popcountll(match_ptr[2]);
            total_popcount += __builtin_popcountll(match_ptr[3]);
        }

        return total_popcount;
    }

    void convert_avx2_to_binary(const __m256i *avx2_data, std::vector<uint64_t> &binary_output) const
    {
        size_t output_size = binary_output.size();
        size_t bits_processed = 0;

        for (size_t i = 0; i < avx_vectors_per_embedding && bits_processed < vector_dim; i++)
        {
            // Extract 64-bit chunks from the AVX2 vector
            uint64_t *chunks = (uint64_t *)&avx2_data[i];

            // Copy each 64-bit chunk to the output
            for (size_t j = 0; j < 4 && bits_processed < vector_dim; j++)
            {
                size_t output_idx = (i * 4 + j);
                if (output_idx < output_size)
                {
                    binary_output[output_idx] = chunks[j];
                }
                bits_processed += 64;
            }
        }

        // Handle the last partial chunk if vector_dim is not a multiple of 64
        size_t remaining_bits = vector_dim % 64;
        if (remaining_bits > 0 && !binary_output.empty())
        {
            // Mask off any extra bits in the last uint64_t
            uint64_t mask = (1ULL << remaining_bits) - 1;
            binary_output.back() &= mask;
        }
    }

    void debug_print_binary(const std::vector<uint64_t> &binary_data) const
    {
        for (size_t i = 0; i < binary_data.size(); i++)
        {
            std::cout << "Chunk " << i << ": ";
            for (int j = 63; j >= 0; j--)
            {
                std::cout << ((binary_data[i] >> j) & 1);
                if (j % 8 == 0)
                    std::cout << " ";
            }
            std::cout << std::endl;
        }
    }
};