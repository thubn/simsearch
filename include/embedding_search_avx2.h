#pragma once
#include "embedding_search_base.h"
#include "aligned_types.h"
#include <immintrin.h>

class EmbeddingSearchAVX2 : public EmbeddingSearchBase<avx2_vector, float>
{
private:
    std::vector<avx2_vector> embeddings;
    size_t vector_size;

public:
    bool load(const std::string &filename) override;
    bool setEmbeddings(const std::vector<std::vector<float>> &m);
    std::vector<std::pair<float, size_t>> similarity_search(const avx2_vector &query, size_t k);
    std::vector<std::pair<float, size_t>> similarity_search(const avx2_vector &query, size_t k,
                                                            std::vector<std::pair<int, size_t>> &searchIndexes);

private:
    float cosine_similarity(const avx2_vector &a, const avx2_vector &b);
    float dot_product_avx2(__m256 a, __m256 b);
};
