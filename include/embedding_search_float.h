#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchFloat : public EmbeddingSearchBase<std::vector<float>, float>
{
public:
    EmbeddingSearchFloat() = default;

    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float> &query, size_t k) override;

    bool pca_dimension_reduction(int factor);

    bool validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message) override
    {
        if (input.empty())
        {
            error_message = "Input vector is empty";
            return false;
        }
        return true;
    }

    bool setEmbeddings(const std::vector<std::vector<float>> &float_embeddings) override;
private:
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
};