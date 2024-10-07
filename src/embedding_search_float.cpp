// embedding_search_float.cpp
#include "embedding_search_float.h"
#include "embedding_io.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

bool EmbeddingSearchFloat::load(const std::string &filename)
{
    // Determine file type and call appropriate loader
    if (filename.ends_with(".safetensors"))
    {
        return EmbeddingIO::load_safetensors(filename, embeddings, sentences);
    }
    else if (filename.ends_with(".ndjson"))
    {
        return EmbeddingIO::load_json(filename, embeddings, sentences);
    }
    else if (filename.ends_with(".jsonl"))
    {
        return EmbeddingIO::load_json2(filename, embeddings, sentences);
    }
    else
    {
        throw std::runtime_error("Unsupported file format");
    }
}

std::vector<std::pair<float, size_t>> EmbeddingSearchFloat::similarity_search(const std::vector<float> &query, size_t k)
{
    if (query.size() != embeddings[0].size())
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        float sim = cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<float, size_t>>(similarities.begin(), similarities.begin() + k);
}

float EmbeddingSearchFloat::cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    float dot_product = 0.0f;
    float mag_a = 0.0f;
    float mag_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    return dot_product / (std::sqrt(mag_a) * std::sqrt(mag_b));
}