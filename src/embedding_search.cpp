#include "embedding_search.h"
#include "util.h"
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

EmbeddingSearch::EmbeddingSearch() : vector_size(0) {}

const std::vector<std::vector<float>>& EmbeddingSearch::getEmbeddings(){
    return embeddings;
}

const size_t& EmbeddingSearch::getVectorSize(){
    return vector_size;
}

bool EmbeddingSearch::load_safetensors(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::cout << "in1" << std::endl;

    auto stringJson = readUTF8StringFromFile(filename, 8);

    std::cout << "in2" << std::endl;

    json safetensorsHeader = json::parse(stringJson);

    std::cout << "Json Header:\n"
              << safetensorsHeader.dump() << std::endl;
    // exit(0);

    uint64_t num_vectors = safetensorsHeader.at("shard_0").at("shape").at(0).get<int>();
    uint64_t vector_dim = safetensorsHeader.at("shard_0").at("shape").at(1).get<int>();

    vector_size = vector_dim;
    embeddings.resize(num_vectors, std::vector<float>(vector_dim));

    // Read embedding data
    for (auto &vec : embeddings)
    {
        file.read(reinterpret_cast<char *>(vec.data()), vector_dim * sizeof(float));
    }

    std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_dim << std::endl;
    return true;
}

std::vector<size_t> EmbeddingSearch::similarity_search(const std::vector<float> &query, size_t k)
{
    if (query.size() != vector_size)
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

    std::vector<size_t> result;
    result.reserve(k);
    for (size_t i = 0; i < k && i < similarities.size(); ++i)
    {
        result.push_back(similarities[i].second);
    }

    return result;
}

float EmbeddingSearch::cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
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