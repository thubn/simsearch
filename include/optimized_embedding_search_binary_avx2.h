#pragma once
#include "embedding_search_base.h"
#include "aligned_types.h"
#include <immintrin.h>

class OptimizedEmbeddingSearchBinaryAVX2 : public OptimizedEmbeddingSearchBase<avx2i_vector, int32_t, __m256i>
{
public:
    OptimizedEmbeddingSearchBinaryAVX2() = default;

    bool setEmbeddings(const std::vector<std::vector<float>> &input_vectors) override;
    std::vector<std::pair<int32_t, size_t>> similarity_search(const avx2i_vector &query, size_t k) override;
    bool load(const std::string &filename) override;
    avx2i_vector getEmbeddingAVX2(size_t index) const;

protected:
    bool validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message) override;

private:
    int32_t compute_similarity_avx2(const __m256i *vec_a, const __m256i *vec_b) const;
    void convert_float_to_binary_avx2(const std::vector<float> &input, __m256i *output) const;
};