#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchFloat : public EmbeddingSearchBase<std::vector<float>, float>
{
public:
    EmbeddingSearchFloat() = default;

    bool load(const std::string &filename) override;
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float> &query, size_t k) override;

    void unsetEmbeddings();
    void unsetSentences();
    bool pca_dimension_reduction(int target_dim);

    bool validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message) override
    {
        if (input.empty())
        {
            error_message = "Input vector is empty";
            return false;
        }
        return true;
    }

private:
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
};