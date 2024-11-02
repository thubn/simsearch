#pragma once
#include "embedding_search_base.h"
#include "aligned_types.h"
#include <immintrin.h>

class EmbeddingSearchBinaryAVX2 : public EmbeddingSearchBase<avx2i_vector, int>
{
public:
    EmbeddingSearchBinaryAVX2() = default;
    bool load(const std::string &filename) override;
    std::vector<std::pair<int, size_t>> similarity_search(const avx2i_vector &query, size_t k) override;
    bool validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message) override;

    bool create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_embeddings);

private:
    int binary_cosine_similarity(const avx2i_vector &a, const avx2i_vector &b);
};