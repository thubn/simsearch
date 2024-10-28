#pragma once
#include "embedding_search_base.h"
#include <immintrin.h>

class EmbeddingSearchBinaryAVX2 : public EmbeddingSearchBase<__m256i, int>
{
public:
    bool load(const std::string &filename) override;
    std::vector<std::pair<int, size_t>> similarity_search(const std::vector<__m256i> &query, size_t k) override;
    bool create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_embeddings);
    std::vector<__m256i> floatToBinaryAvx2(const std::vector<float> &v);

private:
    int binary_cosine_similarity(const std::vector<__m256i> &a, const std::vector<__m256i> &b);
};