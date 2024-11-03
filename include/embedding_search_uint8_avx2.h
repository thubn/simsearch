#pragma once
#include "embedding_search_base.h"
#include "aligned_types.h"
#include <immintrin.h>

class EmbeddingSearchUint8AVX2 : public EmbeddingSearchBase<avx2i_vector, uint>
{
public:
    EmbeddingSearchUint8AVX2() = default;

    // Override virtual functions from base class
    bool load(const std::string &filename) override;
    std::vector<std::pair<uint, size_t>> similarity_search(const avx2i_vector &query, size_t k) override;
    bool validateDimensions(const std::vector<std::vector<float>> &input,std::string &error_message) override;

    // Additional functions
    std::vector<std::pair<uint, size_t>> similarity_search(const avx2i_vector &query,size_t k, std::vector<std::pair<int, size_t>> &searchIndexes);
    bool setEmbeddings(const std::vector<std::vector<float>> &m);

private:
    uint cosine_similarity(const avx2i_vector &a, const avx2i_vector &b);
    float dot_product_avx2(__m256i a, __m256i b);
};