#pragma once
#include "embedding_search_base.h"
#include <cstdint>

class EmbeddingSearchBinary : public EmbeddingSearchBase<uint64_t, int>
{
public:
    bool load(const std::string &filename) override;
    std::vector<std::pair<int, size_t>> similarity_search(const std::vector<uint64_t> &query, size_t k) override;
    bool create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_embeddings);

private:
    int binary_cosine_similarity(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b);
};