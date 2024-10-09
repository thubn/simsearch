// #include "embedding_search.h"
#include "embedding_search_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_binary.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_uint8_avx2.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

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

const uint F32 = 0;
const uint BINARY = 1;
const uint F32_AVX2 = 2;
const uint BINARY_AVX2 = 3;
const uint BAVX2_F32AVX2 = 4;
const uint UINT8_AVX2 = 5;

int main()
{
    EmbeddingSearchFloat searcher;
    EmbeddingSearchAVX2 searcherAvx2;
    EmbeddingSearchBinary searcherBinary;
    EmbeddingSearchBinaryAVX2 searcherBinaryAvx2;
    EmbeddingSearchUint8AVX2 searcherUint8Avx2;

    // Load embeddings
    searcher.load("../data/requests_for_openai_embeddings_result.jsonl");
    searcherAvx2.setEmbeddings(searcher.getEmbeddings());
    searcherBinary.create_binary_embedding_from_float(searcher.getEmbeddings());
    searcherBinaryAvx2.create_binary_embedding_from_float(searcher.getEmbeddings());
    searcherUint8Avx2.setEmbeddings(searcher.getEmbeddings());

    // Generate random index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, searcher.getEmbeddings().size());

    // Number of similar vectors to retrieve
    size_t k = 25;
    size_t runs = 250;
    size_t rescoring_factor = 50;

    // arrays for saving results times and jaccard
    std::vector<int64_t> times;
    times.resize(6, 0);
    std::vector<double> jaccardIndexes;
    jaccardIndexes.resize(6, 0);

    for (int i = 0; i < runs; i++)
    {
        auto random_index = distrib(gen);
        //std::cout << "random index: " << random_index << std::endl;

        // Example query vector
        std::vector<float> query = searcher.getEmbeddings()[random_index];
        std::vector<u_int64_t> binary_query = searcherBinary.getEmbeddings()[random_index];
        std::vector<__m256> queryAvx2 = searcherAvx2.getEmbeddings()[random_index];
        std::vector<__m256i> binary_queryAvx2 = searcherBinaryAvx2.getEmbeddings()[random_index];
        std::vector<__m256i> uint8_queryAvx2 = searcherUint8Avx2.getEmbeddings()[random_index];

        // float32
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<float, size_t>> results = searcher.similarity_search(query, k);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // binary
        start = end;
        std::vector<std::pair<int, size_t>> binary_results = searcherBinary.similarity_search(binary_query, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 AVX2
        start = end;
        std::vector<std::pair<float, size_t>> avx2_results = searcherAvx2.similarity_search(queryAvx2, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_avx2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Binary AVX2
        start = end;
        std::vector<std::pair<int, size_t>> binary_avx2_results = searcherBinaryAvx2.similarity_search(binary_queryAvx2, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_avx2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Binary AVX2 -> float32 AVX2
        start = end;
        std::vector<std::pair<int, size_t>> binary_avx2_rescore_results = searcherBinaryAvx2.similarity_search(binary_queryAvx2, k * rescoring_factor);
        std::vector<std::pair<float, size_t>> avx2_rescore_results = searcherAvx2.similarity_search(queryAvx2, k, binary_avx2_rescore_results);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_avx2_float32_avx2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 AVX2
        start = end;
        std::vector<std::pair<uint, size_t>> uint8_avx2_results = searcherUint8Avx2.similarity_search(uint8_queryAvx2, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_uint8_avx2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        /*
        std::cout << "Time similarity_search:             " << time_similarity_search << "us\n"
                  << "Time avx2_similarity_search:        " << time_avx2_similarity_search << "us\n"
                  << "Time binary_similarity_search:      " << time_binary_similarity_search << "us\n"
                  << "Time binary_avx2_similarity_search: " << time_binary_avx2_similarity_search << "us\n"
                  << "Time b_avx2-f32_avx2_sim_search:    " << time_binary_avx2_float32_avx2_similarity_search << "us\n"
                  << "Time uint8_avx2_similarity_search:  " << time_uint8_avx2_similarity_search << "us\n"
                  << std::endl;
        std::cout << "Jaccard Index:\n"
                  << "f32 - avx2:            " << calculateJaccardIndex(results, avx2_results) << "\n"
                  << "f32 - binary:          " << calculateJaccardIndex(results, binary_results) << "\n"
                  << "f32 - binary_avx2:     " << calculateJaccardIndex(results, binary_avx2_results) << "\n"
                  << "f32 - b_avx2-f32_avx2: " << calculateJaccardIndex(results, avx2_rescore_results) << "\n"
                  << "f32 - uint8_avx2:      " << calculateJaccardIndex(results, uint8_avx2_results)
                  << std::endl;
        std::cout << "Top " << k << " similar vectors:" << std::endl
                  << "f32 | avx2 | binary | binary_avx2 | b_avx2-f32_avx2 | uint8_avx2"
                  << std::endl;
        for (int i = 0; i < k; i++)
        {
            std::cout << "Index: " << results[i].second << "\tScore: " << results[i].first
                      << "\t| Index: " << avx2_results[i].second << "\tScore: " << avx2_results[i].first
                      << "\t| Index: " << binary_results[i].second << "\tScore: " << binary_results[i].first
                      << "\t| Index: " << binary_avx2_results[i].second << "\tScore: " << binary_avx2_results[i].first
                      << "\t| Index: " << avx2_rescore_results[i].second << "\tScore: " << avx2_rescore_results[i].first
                      << "\t| Index: " << uint8_avx2_results[i].second << "\tScore: " << uint8_avx2_results[i].first
                      << std::endl;
        }
        std::cout << "===========================================================" << std::endl;
        */

        times[F32] += time_similarity_search;
        times[F32_AVX2] += time_avx2_similarity_search;
        times[BINARY] += time_binary_similarity_search;
        times[BINARY_AVX2] += time_binary_avx2_similarity_search;
        times[BAVX2_F32AVX2] += time_binary_avx2_float32_avx2_similarity_search;
        times[UINT8_AVX2] += time_uint8_avx2_similarity_search;

        //jaccardIndexes[F32] += time_similarity_search;
        jaccardIndexes[F32_AVX2] += calculateJaccardIndex(results, avx2_results);
        jaccardIndexes[BINARY] += calculateJaccardIndex(results, binary_results);
        jaccardIndexes[BINARY_AVX2] += calculateJaccardIndex(results, binary_avx2_results);
        jaccardIndexes[BAVX2_F32AVX2] += calculateJaccardIndex(results, avx2_rescore_results);
        jaccardIndexes[UINT8_AVX2] += calculateJaccardIndex(results, uint8_avx2_results);
    }
    std::cout << "num runs: " << runs << " | k: " << k << " | rescoring_factor: " << rescoring_factor << " | num Embeddings: " << searcher.getEmbeddings().size() << " | vector dimensions: " << searcher.getEmbeddings()[0].size() << std::endl
              << std::endl
              << "Average Times:" << std::endl
              << "F32:           " << times[F32] / runs << "us" << std::endl
              << "F32_AVX2:      " << times[F32_AVX2] / runs << "us" << std::endl
              << "BINARY:        " << times[BINARY] / runs << "us" << std::endl
              << "BINARY_AVX2:   " << times[BINARY_AVX2] / runs << "us" << std::endl
              << "BAVX2_F32AVX2: " << times[BAVX2_F32AVX2] / runs << "us" << std::endl
              << "UINT8_AVX2:    " << times[UINT8_AVX2] / runs << "us" << std::endl
              << std::endl
              << "Average Jaccard Index (compared to F32):" << std::endl
              << "F32_AVX2:      " << jaccardIndexes[F32_AVX2] / runs << std::endl
              << "BINARY:        " << jaccardIndexes[BINARY] / runs << std::endl
              << "BINARY_AVX2:   " << jaccardIndexes[BINARY_AVX2] / runs << std::endl
              << "BAVX2_F32AVX2: " << jaccardIndexes[BAVX2_F32AVX2] / runs << std::endl
              << "UINT8_AVX2:    " << jaccardIndexes[UINT8_AVX2] / runs << std::endl
              << std::endl;

    return 0;
}
