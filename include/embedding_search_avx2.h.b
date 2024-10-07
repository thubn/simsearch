#ifndef EMBEDDING_SEARCH_AVX2_H
#define EMBEDDING_SEARCH_AVX2_H

#include <vector>
#include <string>
#include <cstdint>
#include <immintrin.h>

class EmbeddingSearchAvx2 {
private:
    std::vector<std::vector<__m256>> embeddings;
    std::vector<std::vector<__m256i>> binary_embeddings;
    std::vector<std::string> sentences;
    size_t vector_size;

    float cosine_similarity(const std::vector<__m256>& a, const std::vector<__m256>& b);
    int binary_cosine_similarity(const std::vector<__m256i>& a, const std::vector<__m256i>& b);

public:
    EmbeddingSearchAvx2();

    const std::vector<std::vector<__m256>>& getEmbeddings();
    const std::vector<std::vector<__m256i>>& getBinaryEmbeddings();
    const std::vector<std::string>& getSentences();
    const size_t& getVectorSize();
    bool setEbeddings(const std::vector<std::vector<float>>& m);
    bool setBinaryEbeddings(const std::vector<std::vector<uint64_t>> &m);

    std::vector<std::pair<float, size_t>> similarity_search(const std::vector<__m256>& query, size_t k);
    //bool create_binary_embedding_from_float();
    std::vector<std::pair<int, size_t>> binary_similarity_search(const std::vector<__m256i>& query, size_t k);
};

#endif // EMBEDDING_SEARCH_AVX2_H