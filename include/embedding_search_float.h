#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchFloat : public EmbeddingSearchBase<std::vector<float>, float>
{
public:
    bool load(const std::string &filename) override;
    void unsetEmbeddings();
    void unsetSentences();
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float> &query, size_t k);
    bool pca_dimension_reduction(int target_dim);

private:
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
};