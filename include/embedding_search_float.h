#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchFloat : public EmbeddingSearchBase<float, float>
{
public:
    bool load(const std::string &filename) override;
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float> &query, size_t k) override;

private:
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
};