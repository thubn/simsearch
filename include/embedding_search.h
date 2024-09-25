#ifndef EMBEDDING_SEARCH_H
#define EMBEDDING_SEARCH_H

#include <vector>
#include <string>

class EmbeddingSearch {
private:
    std::vector<std::vector<float>> embeddings;
    size_t vector_size;

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

public:
    EmbeddingSearch();

    bool load_safetensors(const std::string& filename);
    std::vector<size_t> similarity_search(const std::vector<float>& query, size_t k);
};

#endif // EMBEDDING_SEARCH_H