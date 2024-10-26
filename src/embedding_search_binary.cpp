// embedding_search_binary.cpp
#include "embedding_search_binary.h"
#include "embedding_search_float.h"
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <omp.h>

bool EmbeddingSearchBinary::load(const std::string &filename)
{
    // Binary embeddings are typically created from float embeddings
    // This method could be implemented to directly load binary data if needed
    throw std::runtime_error("Direct loading of binary embeddings not implemented");
}

std::vector<std::pair<int, size_t>> EmbeddingSearchBinary::similarity_search(const std::vector<uint64_t> &query, size_t k)
{
    if (query.size() != vector_size)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<int, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        int sim = binary_cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<int, size_t>>(similarities.begin(), similarities.begin() + k);
}

bool EmbeddingSearchBinary::create_binary_embedding_from_float(const std::vector<std::vector<float>> &float_data)
{
    size_t num_vectors = float_data.size();
    size_t float_vector_size = float_data[0].size();

    vector_size = (float_vector_size + 63) / 64; // Round up to nearest multiple of 64
    embeddings.resize(num_vectors, std::vector<uint64_t>(vector_size, 0));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_vectors; ++i)
    {
        for (size_t j = 0; j < float_vector_size; ++j)
        {
            if (float_data[i][j] >= 0)
            {
                embeddings[i][j / 64] |= (1ULL << (63 - (j % 64)));
            }
        }
    }

    // sentences = float_embeddings.getSentences();
    return true;
}

int EmbeddingSearchBinary::binary_cosine_similarity(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b)
{
    int dot_product = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += __builtin_popcountll(~(a[i] ^ b[i]));
    }
    return dot_product;
}