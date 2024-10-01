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

const std::vector<std::vector<float>> &EmbeddingSearch::getEmbeddings()
{
    return embeddings;
}

const std::vector<std::vector<bool>> &EmbeddingSearch::getBinaryEmbeddings()
{
    return binary_embeddings;
}

const std::vector<std::string> &EmbeddingSearch::getSentences()
{
    return sentences;
}

const size_t &EmbeddingSearch::getVectorSize()
{
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

    // get Header length
    uint64_t headerLength;
    auto stringJson = readUTF8StringFromFile(filename, 8, headerLength);
    std::cout << "header length: " << headerLength << std::endl;

    json safetensorsHeader = json::parse(stringJson);

    std::cout << "Json Header:\n"
              << safetensorsHeader.dump() << std::endl;

    // get details of safetensors file
    uint64_t num_vectors = safetensorsHeader.at("shard_0").at("shape").at(0).get<int>();
    uint64_t vector_dim = safetensorsHeader.at("shard_0").at("shape").at(1).get<int>();
    uint64_t data_offset = safetensorsHeader.at("shard_0").at("data_offsets").at(0).get<int>();
    uint64_t num_shard_offset = safetensorsHeader.at("num_shard").at("data_offsets").at(0).get<int>();

    // get number of shards in safetensors file (unused)
    file.seekg(8 + headerLength + num_shard_offset);
    int64_t num_shards;
    file.read(reinterpret_cast<char *>(&num_shards), sizeof(num_shards));

    vector_size = vector_dim;
    embeddings.resize(num_vectors, std::vector<float>(vector_size));

    // change read position to the position of embedding/vector data
    file.seekg(8 + headerLength + data_offset);

    // Read embedding data
    for (auto &vec : embeddings)
    {
        file.read(reinterpret_cast<char *>(vec.data()), vector_dim * sizeof(float));
    }
    file.close();

    std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_dim << std::endl;
    return true;
}

bool EmbeddingSearch::load_json(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    std::string line;
    int lineNumber = 0;
    int num_vectors = countLines(filename);

    std::cout << "Num Lines: :" << num_vectors << std::endl;

    std::getline(file, line);
    json j = json::parse(line);

    std::cout << "Dimensions: " << j.at("all-MiniLM-L6-v2").size() << std::endl;
    vector_size = j.at("all-MiniLM-L6-v2").size();

    embeddings.resize(num_vectors, std::vector<float>(vector_size));
    sentences.resize(num_vectors);

    file.clear();
    file.seekg(0);
    while (std::getline(file, line))
    {
        try
        {
            json j = json::parse(line);
            // std::cout << "Line " << lineNumber << ": " << j.dump() << std::endl;
            embeddings[lineNumber] = j.at("all-MiniLM-L6-v2").get<std::vector<float>>();
            sentences[lineNumber] = j.at("body").get<std::string>();
        }
        catch (json::parse_error &e)
        {
            std::cerr << "Parse error on line " << lineNumber << ": " << e.what() << std::endl;
        }
        lineNumber++;
    }
    file.close();
    std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_size << std::endl;
    return true;
}

bool EmbeddingSearch::load_json2(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    std::string line;
    int lineNumber = 0;
    int num_vectors = countLines(filename);

    std::cout << "Num Lines: " << num_vectors << std::endl;

    std::getline(file, line);
    json j = json::parse(line);

    
    vector_size = j.at(1).at("data").at(0).at("embedding").size();
    std::cout << "Dimensions: " << vector_size << std::endl;

    embeddings.resize(num_vectors, std::vector<float>(vector_size));
    sentences.resize(num_vectors);

    file.clear();
    file.seekg(0);
    while (std::getline(file, line))
    {
        try
        {
            json j = json::parse(line);
            // std::cout << "Line " << lineNumber << ": " << j.dump() << std::endl;
            embeddings[lineNumber] = j.at(1).at("data").at(0).at("embedding").get<std::vector<float>>();
            sentences[lineNumber] = j.at(0).at("input").get<std::string>();
        }
        catch (json::parse_error &e)
        {
            std::cerr << "Parse error on line " << lineNumber << ": " << e.what() << std::endl;
        }
        lineNumber++;
        // exit(0);
    }
    file.close();
    std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_size << std::endl;
    return true;
}

std::vector<std::pair<float, size_t>> EmbeddingSearch::similarity_search(const std::vector<float> &query, size_t k)
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

    // std::vector<size_t> result;
    std::vector<std::pair<float, size_t>> result;
    result.reserve(k);
    for (size_t i = 0; i < k && i < similarities.size(); ++i)
    {
        result.push_back(similarities[i]);
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

bool EmbeddingSearch::create_binary_embedding_from_float(){
    binary_embeddings.resize(embeddings.size(), std::vector<bool>(vector_size));
    
    int size = embeddings.size();

    for(int i = 0; i < size; i++){
        for(int j = 0; j < vector_size; j++){
            uint32_t intBits = *reinterpret_cast<uint32_t*>(&embeddings[i][j]);
            bool signBit = intBits >> 31;
            binary_embeddings[i][j] = !signBit;
        }
    }
    return true;
}

std::vector<std::pair<int, size_t>> EmbeddingSearch::binary_similarity_search(const std::vector<bool> &query, size_t k)
{
    if (query.size() != vector_size)
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<int, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        int sim = binary_cosine_similarity(query, binary_embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    // std::vector<size_t> result;
    std::vector<std::pair<int, size_t>> result;
    result.reserve(k);
    for (size_t i = 0; i < k && i < similarities.size(); ++i)
    {
        result.push_back(similarities[i]);
    }

    return result;
}

int EmbeddingSearch::binary_cosine_similarity(const std::vector<bool> &a, const std::vector<bool> &b){
    int dot_product = 0;
    for(size_t i = 0; i < vector_size; i++){
        dot_product += !(a[i] ^ b[i]);
    }
    return dot_product;
}