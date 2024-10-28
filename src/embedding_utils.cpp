#include "embedding_utils.h"
#include <stdexcept>
#include <omp.h>
#include <cstdint>

namespace EmbeddingUtils
{
    void convertSingleEmbeddingToAVX2(
        const std::vector<float> &input,
        std::vector<__m256> &output,
        size_t vector_size)
    {
        for (size_t j = 0; j < vector_size; j++)
        {
            size_t k = j * 8; // Each AVX2 vector holds 8 floats
            output[j] = _mm256_loadu_ps(&input[k]);
        }
    }

    void convertEmbeddingsToAVX2(
        const std::vector<std::vector<float>> &input,
        std::vector<std::vector<__m256>> &output,
        size_t vector_size)
    {
        size_t num_embeddings = input.size();
        output.resize(num_embeddings, std::vector<__m256>(vector_size));

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_embeddings; i++)
        {
            convertSingleEmbeddingToAVX2(input[i], output[i], vector_size);
        }
    }

    bool validateAVX2Dimensions(const std::vector<std::vector<float>> &input, std::string &error_message)
    {
        if (input.empty())
        {
            error_message = "Input vector is empty";
            return false;
        }

        if (input[0].size() % 8 != 0)
        {
            error_message = "Input vector size must be a multiple of 8";
            return false;
        }

        return true;
    }

    size_t calculateBinaryAVX2VectorSize(size_t float_vector_size)
    {
        return ((float_vector_size + 63) / 64 + 3) / 4;
    }

    void convertSingleFloatToBinaryAVX2(
        const std::vector<float> &input,
        std::vector<__m256i> &output,
        size_t float_vector_size,
        size_t vector_size)
    {
        // Process each AVX2 vector (256 bits = 4 x 64 bits)
        for (size_t j = 0; j < vector_size; j++)
        {
            uint64_t temp_bits[4] = {0, 0, 0, 0};

            // Process each 64-bit chunk within this AVX2 vector
            for (size_t chunk = 0; chunk < 4; chunk++)
            {
                // Calculate base index for this 64-bit chunk
                size_t base_idx = j * 256 + chunk * 64;

                // Process each bit within the 64-bit chunk
                for (size_t bit = 0; bit < 64 && (base_idx + bit) < float_vector_size; bit++)
                {
                    if (base_idx + bit < input.size() && input[base_idx + bit] >= 0)
                    {
                        temp_bits[chunk] |= (1ULL << (63 - bit));
                    }
                }
            }

            // Convert to AVX2 vector
            output[j] = _mm256_setr_epi64x(
                temp_bits[0],
                temp_bits[1],
                temp_bits[2],
                temp_bits[3]);
        }
    }

    bool validateBinaryAVX2Dimensions(
        const std::vector<std::vector<float>> &input,
        std::string &error_message)
    {
        if (input.empty())
        {
            error_message = "Input vector is empty";
            return false;
        }

        if (input[0].empty())
        {
            error_message = "Input vectors cannot be empty";
            return false;
        }

        return true;
    }

    void convertSingleFloatToUint8AVX2(
        const std::vector<float> &input,
        std::vector<__m256i> &output,
        size_t vector_size)
    {
        for (size_t j = 0; j < vector_size; j++)
        {
            size_t k = j * 32;

            // Create temporary array for int8 values
            std::vector<int8_t> temp_int8(32, 0);

            // Convert float values to int8
            for (int l = 0; l < 32; l++)
            {
                temp_int8[l] = static_cast<int8_t>(input[k + l] * 127);
            }

            // Load int8 values into AVX2 vector
            output[j] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(temp_int8.data()));
        }
    }

    bool validateUint8AVX2Dimensions(
        const std::vector<std::vector<float>> &input,
        std::string &error_message)
    {
        if (input.empty())
        {
            error_message = "Input vector is empty";
            return false;
        }

        if (input[0].empty())
        {
            error_message = "Input vectors cannot be empty";
            return false;
        }

        if (input[0].size() % 32 != 0)
        {
            error_message = "Input vector size must be a multiple of 32";
            return false;
        }

        return true;
    }

    size_t calculateUint8AVX2VectorSize(size_t float_vector_size)
    {
        return float_vector_size / 32; // Each AVX2 vector holds 32 int8 values
    }
}