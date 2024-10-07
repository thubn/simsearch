// embedding_io.cpp
#include "embedding_io.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace EmbeddingIO
{

    std::string readUTF8StringFromFile(const std::string &filename, size_t offset, uint64_t &length)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        file.seekg(offset);
        file.read(reinterpret_cast<char *>(&length), sizeof(length));

        std::string result(length, '\0');
        file.read(&result[0], length);

        if (!file)
        {
            throw std::runtime_error("Error reading from file: " + filename);
        }

        return result;
    }

    bool load_safetensors(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            uint64_t headerLength;
            auto headerJson = readUTF8StringFromFile(filename, 8, headerLength);

            json safetensorsHeader = json::parse(headerJson);

            uint64_t num_vectors = safetensorsHeader.at("shard_0").at("shape").at(0).get<int>();
            uint64_t vector_dim = safetensorsHeader.at("shard_0").at("shape").at(1).get<int>();
            uint64_t data_offset = safetensorsHeader.at("shard_0").at("data_offsets").at(0).get<int>();

            std::ifstream file(filename, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            embeddings.resize(num_vectors, std::vector<float>(vector_dim));

            file.seekg(8 + headerLength + data_offset);

            for (auto &vec : embeddings)
            {
                file.read(reinterpret_cast<char *>(vec.data()), vector_dim * sizeof(float));
                if (!file)
                {
                    throw std::runtime_error("Error reading embedding data from file: " + filename);
                }
            }

            std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_dim << " from " << filename << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_safetensors: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_json(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            std::ifstream file(filename);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            std::string line;
            int lineNumber = 0;

            while (std::getline(file, line))
            {
                try
                {
                    json j = json::parse(line);
                    embeddings.push_back(j.at("all-MiniLM-L6-v2").get<std::vector<float>>());
                    sentences.push_back(j.at("body").get<std::string>());
                    lineNumber++;
                }
                catch (json::parse_error &e)
                {
                    std::cerr << "Parse error on line " << lineNumber << ": " << e.what() << std::endl;
                }
                catch (json::out_of_range &e)
                {
                    std::cerr << "JSON key error on line " << lineNumber << ": " << e.what() << std::endl;
                }
            }

            if (embeddings.empty())
            {
                throw std::runtime_error("No valid embeddings found in file: " + filename);
            }

            std::cout << "Loaded " << lineNumber << " embeddings of dimension " << embeddings[0].size() << " from " << filename << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_json: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_json2(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            std::ifstream file(filename);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            std::string line;
            int lineNumber = 0;

            while (std::getline(file, line))
            {
                try
                {
                    json j = json::parse(line);
                    embeddings.push_back(j.at(1).at("data").at(0).at("embedding").get<std::vector<float>>());
                    sentences.push_back(j.at(0).at("input").get<std::string>());
                    lineNumber++;
                }
                catch (json::parse_error &e)
                {
                    std::cerr << "Parse error on line " << lineNumber << ": " << e.what() << std::endl;
                }
                catch (json::out_of_range &e)
                {
                    std::cerr << "JSON key error on line " << lineNumber << ": " << e.what() << std::endl;
                }
            }

            if (embeddings.empty())
            {
                throw std::runtime_error("No valid embeddings found in file: " + filename);
            }

            std::cout << "Loaded " << lineNumber << " embeddings of dimension " << embeddings[0].size() << " from " << filename << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_json2: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace EmbeddingIO