#pragma once
#include "embedding_search_base.h"
#include <stdfloat>

class EmbeddingSearchFloat16 : public EmbeddingSearchBase<std::vector<std::bfloat16_t>, float>
{
public:
    EmbeddingSearchFloat16() = default;

    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<std::bfloat16_t> &query, size_t k) override;

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
    float cosine_similarity(const std::vector<std::bfloat16_t> &a, const std::vector<std::bfloat16_t> &b);
};