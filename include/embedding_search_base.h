#pragma once
#include <vector>
#include <string>
#include <cstdint>

template <typename VectorType, typename SimilarityType>
class EmbeddingSearchBase
{
protected:
    std::vector<VectorType> embeddings;
    std::vector<std::string> sentences;
    size_t vector_size;

public:
    virtual ~EmbeddingSearchBase() = default;

    bool isInitialized();
    virtual bool load(const std::string &filename) = 0;
    virtual std::vector<std::pair<SimilarityType, size_t>> similarity_search(const VectorType &query, size_t k) = 0;

    const std::vector<VectorType> &getEmbeddings() const { return embeddings; }
    const std::vector<std::string> &getSentences() const { return sentences; }
    bool setSentences(const std::vector<std::string> &s)
    {
        sentences = s;
        return true;
    };
    size_t getVectorSize() const { return vector_size; }
};

template <typename VectorType, typename SimilarityType>
inline bool EmbeddingSearchBase<VectorType, SimilarityType>::isInitialized()
{
    if (embeddings.size() > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}
