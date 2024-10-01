#ifndef EMBEDDING_SEARCH_H
#define EMBEDDING_SEARCH_H

#include <vector>
#include <string>

class EmbeddingSearch {
private:
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<bool>> binary_embeddings;
    std::vector<std::string> sentences;
    size_t vector_size;

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    int binary_cosine_similarity(const std::vector<bool>& a, const std::vector<bool>& b);

public:
    EmbeddingSearch();

    const std::vector<std::vector<float>>& getEmbeddings();
    const std::vector<std::vector<bool>>& getBinaryEmbeddings();
    const std::vector<std::string>& getSentences();
    const size_t& getVectorSize();

    bool load_safetensors(const std::string& filename);
    bool load_json(const std::string& filename);
    bool load_json2(const std::string& filename);
    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<float>& query, size_t k);
    bool create_binary_embedding_from_float();
    std::vector<std::pair<int, size_t>> binary_similarity_search(const std::vector<bool>& query, size_t k);
};

#endif // EMBEDDING_SEARCH_H