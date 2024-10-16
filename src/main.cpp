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
const uint F32_PCA2 = 6;
const uint F32_PCA4 = 7;
const uint F32_PCA2x2 = 8;
const uint F32_PCA6 = 9;
const uint F32_PCA8 = 10;
const uint F32_PCA16 = 11;
const uint F32_PCA32 = 12;
const uint F32_AVX2_PCA8 = 13;
const uint BINARY_AVX2_PCA6 = 14;

int main()
{
    EmbeddingSearchFloat searcher;
    EmbeddingSearchFloat searcherPCA2;
    EmbeddingSearchFloat searcherPCA2x2;
    EmbeddingSearchFloat searcherPCA4;
    EmbeddingSearchFloat searcherPCA6;
    EmbeddingSearchFloat searcherPCA8;
    EmbeddingSearchFloat searcherPCA16;
    EmbeddingSearchFloat searcherPCA32;
    EmbeddingSearchAVX2 searcherAvx2;
    EmbeddingSearchAVX2 searcherAvx2PCA8;
    EmbeddingSearchBinary searcherBinary;
    EmbeddingSearchBinaryAVX2 searcherBinaryAvx2;
    EmbeddingSearchBinaryAVX2 searcherBinaryAvx2PCA6;
    EmbeddingSearchUint8AVX2 searcherUint8Avx2;

    // Load embeddings
    searcher.load("../data/requests_for_openai_embeddings_result.jsonl");
    searcherPCA2 = searcher;
    searcherPCA2.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 2);
    searcherPCA4 = searcher;
    searcherPCA4.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 4);
    searcherPCA2x2 = searcherPCA2;
    searcherPCA2x2.pca_dimension_reduction(searcherPCA2.getEmbeddings()[0].size() / 2);
    searcherPCA6 = searcher;
    searcherPCA6.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 6);
    searcherPCA8 = searcher;
    searcherPCA8.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 8);
    searcherPCA16 = searcher;
    searcherPCA16.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 16);
    searcherPCA32 = searcher;
    searcherPCA32.pca_dimension_reduction(searcher.getEmbeddings()[0].size() / 32);
    searcherAvx2.setEmbeddings(searcher.getEmbeddings());
    searcherAvx2PCA8.setEmbeddings(searcherPCA8.getEmbeddings());
    searcherBinary.create_binary_embedding_from_float(searcher.getEmbeddings());
    searcherBinaryAvx2.create_binary_embedding_from_float(searcher.getEmbeddings());
    searcherBinaryAvx2PCA6.create_binary_embedding_from_float(searcherPCA6.getEmbeddings());
    searcherUint8Avx2.setEmbeddings(searcher.getEmbeddings());

    // Generate random index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, searcher.getEmbeddings().size());

    // Number of similar vectors to retrieve
    size_t k = 25;
    size_t runs = 500;
    size_t rescoring_factor = 50;

    // arrays for saving results times and jaccard
    std::vector<int64_t> times;
    times.resize(15, 0);
    std::vector<double> jaccardIndexes;
    jaccardIndexes.resize(15, 0);

    for (int i = 0; i < runs; i++)
    {
        auto random_index = distrib(gen);
        // std::cout << "random index: " << random_index << std::endl;

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

        // float32 PCA2
        start = end;
        std::vector<std::pair<float, size_t>> pca2_results = searcherPCA2.similarity_search(searcherPCA2.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA4
        start = end;
        std::vector<std::pair<float, size_t>> pca4_results = searcherPCA4.similarity_search(searcherPCA4.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca4_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA2x2
        start = end;
        std::vector<std::pair<float, size_t>> pca2x2_results = searcherPCA2x2.similarity_search(searcherPCA2x2.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca2x2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA6
        start = end;
        std::vector<std::pair<float, size_t>> pca6_results = searcherPCA6.similarity_search(searcherPCA6.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca6_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA8
        start = end;
        std::vector<std::pair<float, size_t>> pca8_results = searcherPCA8.similarity_search(searcherPCA8.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca8_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA16
        start = end;
        std::vector<std::pair<float, size_t>> pca16_results = searcherPCA16.similarity_search(searcherPCA16.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca16_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // float32 PCA32
        start = end;
        std::vector<std::pair<float, size_t>> pca32_results = searcherPCA32.similarity_search(searcherPCA32.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_pca32_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

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

        // float32 AVX2 PCA8
        start = end;
        std::vector<std::pair<float, size_t>> avx2_pca8_results = searcherAvx2PCA8.similarity_search(searcherAvx2PCA8.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_avx2_pca8_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Binary AVX2
        start = end;
        std::vector<std::pair<int, size_t>> binary_avx2_results = searcherBinaryAvx2.similarity_search(binary_queryAvx2, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_avx2_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Binary AVX2 PCA6
        start = end;
        std::vector<std::pair<int, size_t>> binary_avx2_pca6_results = searcherBinaryAvx2PCA6.similarity_search(searcherBinaryAvx2PCA6.getEmbeddings()[random_index], k);
        end = std::chrono::high_resolution_clock::now();
        auto time_binary_avx2_pca6_similarity_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

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
        times[F32_PCA2] += time_pca2_similarity_search;
        times[F32_PCA4] += time_pca4_similarity_search;
        times[F32_PCA2x2] += time_pca2x2_similarity_search;
        times[F32_PCA6] += time_pca6_similarity_search;
        times[F32_PCA8] += time_pca8_similarity_search;
        times[F32_PCA16] += time_pca16_similarity_search;
        times[F32_PCA32] += time_pca32_similarity_search;
        times[F32_AVX2] += time_avx2_similarity_search;
        times[F32_AVX2_PCA8] += time_avx2_pca8_similarity_search;
        times[BINARY] += time_binary_similarity_search;
        times[BINARY_AVX2] += time_binary_avx2_similarity_search;
        times[BINARY_AVX2_PCA6] += time_binary_avx2_pca6_similarity_search;
        times[BAVX2_F32AVX2] += time_binary_avx2_float32_avx2_similarity_search;
        times[UINT8_AVX2] += time_uint8_avx2_similarity_search;

        // jaccardIndexes[F32] += time_similarity_search;
        jaccardIndexes[F32_PCA2] += calculateJaccardIndex(results, pca2_results);
        jaccardIndexes[F32_PCA4] += calculateJaccardIndex(results, pca4_results);
        jaccardIndexes[F32_PCA2x2] += calculateJaccardIndex(results, pca2x2_results);
        jaccardIndexes[F32_PCA6] += calculateJaccardIndex(results, pca6_results);
        jaccardIndexes[F32_PCA8] += calculateJaccardIndex(results, pca8_results);
        jaccardIndexes[F32_PCA16] += calculateJaccardIndex(results, pca16_results);
        jaccardIndexes[F32_PCA32] += calculateJaccardIndex(results, pca32_results);
        jaccardIndexes[F32_AVX2] += calculateJaccardIndex(results, avx2_results);
        jaccardIndexes[F32_AVX2_PCA8] += calculateJaccardIndex(results, avx2_pca8_results);
        jaccardIndexes[BINARY] += calculateJaccardIndex(results, binary_results);
        jaccardIndexes[BINARY_AVX2] += calculateJaccardIndex(results, binary_avx2_results);
        jaccardIndexes[BINARY_AVX2_PCA6] += calculateJaccardIndex(results, binary_avx2_pca6_results);
        jaccardIndexes[BAVX2_F32AVX2] += calculateJaccardIndex(results, avx2_rescore_results);
        jaccardIndexes[UINT8_AVX2] += calculateJaccardIndex(results, uint8_avx2_results);
    }
    std::cout << "num runs: " << runs << " | k: " << k << " | rescoring_factor: " << rescoring_factor << " | num Embeddings: " << searcher.getEmbeddings().size() << " | vector dimensions: " << searcher.getEmbeddings()[0].size() << std::endl
              << std::endl
              << "Average Times:" << std::endl
              << "F32:           " << times[F32] / runs << "us" << std::endl
              << "F32_PCA2:      " << times[F32_PCA2] / runs << "us" << std::endl
              << "F32_PCA4:      " << times[F32_PCA4] / runs << "us" << std::endl
              << "F32_PCA2x2:    " << times[F32_PCA2x2] / runs << "us" << std::endl
              << "F32_PCA6:      " << times[F32_PCA6] / runs << "us" << std::endl
              << "F32_PCA8:      " << times[F32_PCA8] / runs << "us" << std::endl
              << "F32_PCA16:     " << times[F32_PCA16] / runs << "us" << std::endl
              << "F32_PCA32:     " << times[F32_PCA32] / runs << "us" << std::endl
              << "F32_AVX2:      " << times[F32_AVX2] / runs << "us" << std::endl
              << "F32_AVX2_PCA8: " << times[F32_AVX2_PCA8] / runs << "us" << std::endl
              << "BINARY:        " << times[BINARY] / runs << "us" << std::endl
              << "BINARY_AVX2:   " << times[BINARY_AVX2] / runs << "us" << std::endl
              << "BIN_AVX2_PCA8: " << times[BINARY_AVX2_PCA6] / runs << "us" << std::endl
              << "BAVX2_F32AVX2: " << times[BAVX2_F32AVX2] / runs << "us" << std::endl
              << "UINT8_AVX2:    " << times[UINT8_AVX2] / runs << "us" << std::endl
              << std::endl
              << "Average Jaccard Index (compared to F32):" << std::endl
              << "F32_PCA2:      " << jaccardIndexes[F32_PCA2] / runs << std::endl
              << "F32_PCA4:      " << jaccardIndexes[F32_PCA4] / runs << std::endl
              << "F32_PCA2x2:    " << jaccardIndexes[F32_PCA2x2] / runs << std::endl
              << "F32_PCA6:      " << jaccardIndexes[F32_PCA6] / runs << std::endl
              << "F32_PCA8:      " << jaccardIndexes[F32_PCA8] / runs << std::endl
              << "F32_PCA16      " << jaccardIndexes[F32_PCA16] / runs << std::endl
              << "F32_PCA32:     " << jaccardIndexes[F32_PCA32] / runs << std::endl
              << "F32_AVX2:      " << jaccardIndexes[F32_AVX2] / runs << std::endl
              << "F32_AVX2_PCA8: " << jaccardIndexes[F32_AVX2_PCA8] / runs << std::endl
              << "BINARY:        " << jaccardIndexes[BINARY] / runs << std::endl
              << "BINARY_AVX2:   " << jaccardIndexes[BINARY_AVX2] / runs << std::endl
              << "BIN_AVX2_PCA6: " << jaccardIndexes[BINARY_AVX2_PCA6] / runs << std::endl
              << "BAVX2_F32AVX2: " << jaccardIndexes[BAVX2_F32AVX2] / runs << std::endl
              << "UINT8_AVX2:    " << jaccardIndexes[UINT8_AVX2] / runs << std::endl
              << std::endl
              << "F32           num_emb, vec_size: " << searcher.getEmbeddings().size() << "\t, " << searcher.getEmbeddings()[0].size() << std::endl
              << "F32_PCA2      num_emb, vec_size: " << searcherPCA2.getEmbeddings().size() << "\t, " << searcherPCA2.getEmbeddings()[0].size() << std::endl
              << "F32_PCA4      num_emb, vec_size: " << searcherPCA4.getEmbeddings().size() << "\t, " << searcherPCA4.getEmbeddings()[0].size() << std::endl
              << "F32_PCA2x2    num_emb, vec_size: " << searcherPCA2x2.getEmbeddings().size() << "\t, " << searcherPCA2x2.getEmbeddings()[0].size() << std::endl
              << "F32_PCA6      num_emb, vec_size: " << searcherPCA6.getEmbeddings().size() << "\t, " << searcherPCA6.getEmbeddings()[0].size() << std::endl
              << "F32_PCA8      num_emb, vec_size: " << searcherPCA8.getEmbeddings().size() << "\t, " << searcherPCA8.getEmbeddings()[0].size() << std::endl
              << "F32_PCA16     num_emb, vec_size: " << searcherPCA16.getEmbeddings().size() << "\t, " << searcherPCA16.getEmbeddings()[0].size() << std::endl
              << "F32_PCA32     num_emb, vec_size: " << searcherPCA32.getEmbeddings().size() << "\t, " << searcherPCA32.getEmbeddings()[0].size() << std::endl
              << "F32_AVX2_PCA8 num_emb, vec_size: " << searcherAvx2PCA8.getEmbeddings().size() << "\t, " << searcherAvx2PCA8.getEmbeddings()[0].size() << std::endl
              << "BIN_AVX2_PCA6 num_emb, vec_size: " << searcherBinaryAvx2PCA6.getEmbeddings().size() << "\t, " << searcherBinaryAvx2PCA6.getEmbeddings()[0].size() << std::endl
              << std::endl;
    return 0;
}
