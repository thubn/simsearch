#pragma once
#include "embedding_search_base.h"

class EmbeddingSearchMappedFloat : public EmbeddingSearchBase<std::vector<uint8_t>, float>
{
public:
    EmbeddingSearchMappedFloat() = default;

    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<uint8_t> &query, size_t k) override;

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
    float mapped_floats[256];
    float cosine_similarity(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b);
};