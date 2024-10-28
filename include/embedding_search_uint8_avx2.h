#pragma once
#include "embedding_search_base.h"
#include <immintrin.h>

class EmbeddingSearchUint8AVX2 : public EmbeddingSearchBase<__m256i, uint>
{
public:
    bool load(const std::string &filename) override;
    std::vector<std::pair<uint, size_t>> similarity_search(const std::vector<__m256i> &query, size_t k) override;
    std::vector<std::pair<uint, size_t>> similarity_search(const std::vector<__m256i> &query, size_t k, std::vector<std::pair<int, size_t>> &searchIndexes);
    bool setEmbeddings(const std::vector<std::vector<float>> &m);
    std::vector<__m256i> floatToAvx2(const std::vector<float>& v);

private:
    uint cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b);
    float dot_product_avx2(__m256i a, __m256i b);
};