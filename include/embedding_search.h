#ifndef EMBEDDING_SEARCH_H
#define EMBEDDING_SEARCH_H

#include <vector>
#include <string>

class EmbeddingSearch {
private:
    std::vector<std::vector<float>> embeddings;
    std::vector<std::string> sentences;
    size_t vector_size;

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

public:
    EmbeddingSearch();

    const std::vector<std::vector<float>>& getEmbeddings();
    const std::vector<std::string>& getSentences();
    const size_t& getVectorSize();

    bool load_safetensors(const std::string& filename);
    bool load_json(const std::string& filename);
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float>& query, size_t k);
};

#endif // EMBEDDING_SEARCH_H