#pragma once
#include "embedding_search_base.h"
#include <immintrin.h>

class EmbeddingSearchAVX2 : public EmbeddingSearchBase<__m256, float>
{
public:
    bool load(const std::string &filename) override;
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<__m256> &query, size_t k) override;
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<__m256> &query, size_t k, std::vector<std::pair<int, size_t>> &searchIndexes);
    bool setEmbeddings(const std::vector<std::vector<float>> &m);
    std::vector<__m256> floatToAvx2(const std::vector<float> &v);

private:
    float cosine_similarity(const std::vector<__m256> &a, const std::vector<__m256> &b);
    float dot_product_avx2(__m256 a, __m256 b);
};