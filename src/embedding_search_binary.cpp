#include "embedding_search_binary.h"
#include "embedding_io.h"
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <omp.h>

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

bool EmbeddingSearchBinary::validateDimensions(const std::vector<std::vector<float>> &input, std::string &error_message)
{
    if (input.empty())
    {
        error_message = "Input vector is empty";
        return false;
    }
    if (input[0].empty())
    {
        error_message = "Input vectors cannot be empty";
        return false;
    }
    return true;
}

bool EmbeddingSearchBinary::setEmbeddings(
    const std::vector<std::vector<float>> &float_data)
{

    std::string error_message;
    if (!validateDimensions(float_data, error_message))
    {
        throw std::runtime_error(error_message);
    }

    num_vectors = float_data.size();
    size_t float_vector_size = float_data[0].size();

    vector_size = (float_vector_size + 63) / 64; // Round up to nearest multiple of 64
    embeddings.resize(num_vectors, std::vector<uint64_t>(vector_size, 0));

    for (size_t i = 0; i < num_vectors; ++i)
    {
        for (size_t j = 0; j < float_vector_size; ++j)
        {
            if (float_data[i][j] >= 0)
            {
                size_t chunk_idx = j / 64;
                size_t bit_pos = j % 64;
                embeddings[i][chunk_idx] |= (1ULL << (63 - bit_pos));
            }
        }
    }

    return true;
}

int EmbeddingSearchBinary::binary_cosine_similarity(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b)
{
    int dot_product = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        // Count matching bits using XOR and NOT
        dot_product += __builtin_popcountll(~(a[i] ^ b[i]));
    }
    return dot_product;
}