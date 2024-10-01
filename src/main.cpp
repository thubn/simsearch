#include "embedding_search.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <chrono>

template <typename T1, typename T2>
double calculateJaccardIndex(const std::vector<std::pair<T1, size_t>> &set1,
                             const std::vector<std::pair<T2, size_t>> &set2)
{
    std::vector<size_t> vec1, vec2;

    // Extract the second values from the pairs
    for (const auto &pair : set1)
        vec1.push_back(pair.second);
    for (const auto &pair : set2)
        vec2.push_back(pair.second);

    // Sort the vectors (required for set operations)
    std::sort(vec1.begin(), vec1.end());
    std::sort(vec2.begin(), vec2.end());

    // Remove duplicates
    vec1.erase(std::unique(vec1.begin(), vec1.end()), vec1.end());
    vec2.erase(std::unique(vec2.begin(), vec2.end()), vec2.end());

    // Calculate intersection
    std::vector<size_t> intersection;
    std::set_intersection(vec1.begin(), vec1.end(),
                          vec2.begin(), vec2.end(),
                          std::back_inserter(intersection));

    // Calculate union
    std::vector<size_t> union_set;
    std::set_union(vec1.begin(), vec1.end(),
                   vec2.begin(), vec2.end(),
                   std::back_inserter(union_set));

    // Avoid division by zero
    if (union_set.empty())
    {
        return 0.0;
    }

    return static_cast<double>(intersection.size()) / union_set.size();
}

// Utility function for benchmarking
template <typename Func>
long long benchmarkTime(Func func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main()
{
    EmbeddingSearch searcher;

    // Load embeddings
    /*if (!searcher.load_safetensors("../data/embeddings.safetensors"))
    {
        return 1;
    }*/
    /*if (!searcher.load_json("../data/wiki_minilm.ndjson"))
    {
        return 1;
    }*/
    if (!searcher.load_json2("../data/requests_for_openai_embeddings_result.jsonl"))
    {
        return 1;
    }

    searcher.create_binary_embedding_from_float();

    std::cout << "vector_size: " << searcher.getVectorSize() << "\nEmbeddings: " << searcher.getEmbeddings().size() << std::endl;

    // Generate random index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, searcher.getEmbeddings().size());

    for (int i = 0; i < 10; i++)
    {
        auto random_index = distrib(gen);
        std::cout << "random index: " << random_index << std::endl;

        // Example query vector
        std::vector<float> query = searcher.getEmbeddings()[random_index];
        std::vector<bool> binary_query = searcher.getBinaryEmbeddings()[random_index];

        // Perform similarity search
        size_t k = 100; // Number of similar vectors to retrieve

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<float, size_t>> results = searcher.similarity_search(query, k);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        start = end;
        std::vector<std::pair<int, size_t>> binary_results = searcher.binary_similarity_search(binary_query, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Time similarity_search:        " << time_similarity_search << "us\n"
                  << "Time binary_similarity_search: " << time_binary_similarity_search << "us" << std::endl;
        std::cout << "Jaccard Index: " << calculateJaccardIndex(results, binary_results) << std::endl;
        std::cout << "Top " << k << " similar vectors:" << std::endl;
        for (int i = 0; i < k; i++)
        {
            std::cout << "Index: " << results[i].second << "\tScore: " << results[i].first << " Index: " << binary_results[i].second << "\tScore: " << binary_results[i].first
                      << std::endl;
        }
        std::cout << "===========================================================" << std::endl;
    }

    // auto sentences = searcher.getSentences();

    /*
    // Print results
    std::cout << "Top " << k << " similar vectors:" << std::endl;
    for (std::pair<float, size_t> idx : results)
    {
        std::cout << "Index: " << idx.second << "\tScore: " << idx.first << "\n\nSentence:\n"
                  << sentences[idx.second] << std::endl
                  << std::endl
                  << "==========================================================="
                  << std::endl
                  << std::endl;
    }
    */
    /*
    // Print results
    std::cout << "Top " << k << " similar vectors:" << std::endl;
    for (int i = 0; i < k; i++)
    {
        std::cout << "Index: " << results[i].second << "\tScore: " << results[i].first << " Index: " << binary_results[i].second << "\tScore: " << binary_results[i].first << "\n\nSentence:\n"
                  << sentences[results[i].second] << std::endl
                  << std::endl
                  << "==========================================================="
                  << std::endl
                  << std::endl;
    }
    */

    return 0;
}
