// embedding_utils.h
#pragma once
#include <vector>
#include <immintrin.h>
#include <string>

namespace EmbeddingUtils {
    // Convert a single float vector to AVX2 format
    void convertSingleEmbeddingToAVX2(
        const std::vector<float>& input,
        std::vector<__m256>& output,
        size_t vector_size);
        
    // Convert multiple float vectors to AVX2 format in parallel
    void convertEmbeddingsToAVX2(
        const std::vector<std::vector<float>>& input,
        std::vector<std::vector<__m256>>& output,
        size_t vector_size);
        
    // Validate input dimensions for AVX2 conversion
    bool validateAVX2Dimensions(const std::vector<std::vector<float>>& input, std::string& error_message);

    size_t calculateBinaryAVX2VectorSize(size_t float_vector_size);

    void convertSingleFloatToBinaryAVX2(
        const std::vector<float>& input,
        std::vector<__m256i>& output,
        size_t float_vector_size,
        size_t vector_size);

    bool validateBinaryAVX2Dimensions(
        const std::vector<std::vector<float>>& input,
        std::string& error_message);

    void convertSingleFloatToUint8AVX2(
        const std::vector<float>& input,
        std::vector<__m256i>& output,
        size_t vector_size);

    bool validateUint8AVX2Dimensions(
        const std::vector<std::vector<float>>& input,
        std::string& error_message);
        
    size_t calculateUint8AVX2VectorSize(size_t float_vector_size);
}