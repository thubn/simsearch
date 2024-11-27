#include "aligned_types.h"
#include "config_manager.h"
#include "embedding_search_avx2.h"
#include "embedding_search_binary.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_float16.h"
// #include "embedding_search_float_int8.h"
#include "common_structs.h"
//#include "embedding_search_mapped_float.h"
#include "embedding_search_uint8_avx2.h"
#include "embedding_utils.h"
#include "optimized_embedding_search_avx2.h"
#include "optimized_embedding_search_binary_avx2.h"
#include "optimized_embedding_search_uint8_avx2.h"
#include "stats_exporter.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;

std::vector<size_t> generateRandomIndexes(size_t numRuns, size_t maxIndex) {
  std::vector<size_t> indexes(numRuns);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, maxIndex);

  for (size_t i = 0; i < numRuns; ++i) {
    indexes[i] = distrib(gen);
  }

  return indexes;
}

simsearch::BenchmarkResults
runBenchmark(simsearch::Searchers &searchers,
             const simsearch::BenchmarkConfig &config) {
  simsearch::BenchmarkResults results;

  // Create vectors to store different types of results
  std::vector<SearchMethodStats<float>> floatResults;
  std::vector<SearchMethodStats<uint32_t>> uint32Results;
  std::vector<SearchMethodStats<int>> intResults;

  // Generate random indexes first
  std::vector<size_t> randomIndexes = generateRandomIndexes(
      config.runs, searchers.base.getEmbeddings().size() - 1);

  // Run base F32 searches
  std::cout << "Running F32 searches...";
  std::vector<std::vector<std::pair<float, size_t>>> baseResults(config.runs);
  SearchMethodStats<float> baseStats("F32_base");
  for (size_t i = 0; i < config.runs; i++) {

    auto start = std::chrono::high_resolution_clock::now();
    auto result = searchers.base.similarity_search(
        searchers.base.getEmbeddings()[randomIndexes[i]], config.k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    baseResults[i] = result;
    results.times[simsearch::F32] += time;
    baseStats.times.push_back(time);
    baseStats.results.push_back(result);
  }
  std::cout << " ✓" << std::endl;

  // Lambda to run benchmark for a specific searcher type
  auto runFloatSearcherBenchmark = [&](const std::string &name, auto &searcher,
                                       simsearch::SearcherType type,
                                       const auto &getQuery) {
    if (searcher.isInitialized()) {
      std::cout << "Running " << name << " searches...";
      SearchMethodStats<float> stats(name);
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results = searcher.similarity_search(q, config.k);
        auto end = std::chrono::high_resolution_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        auto jaccard = EmbeddingUtils::calculateJaccardIndex(baseResults[i],
                                                             search_results);
        auto ndcg =
            EmbeddingUtils::calculateNDCG(baseResults[i], search_results);
        results.times[type] += time;
        results.jaccardIndexes[type] += jaccard;
        results.NDCG[type] += ndcg;
        stats.times.push_back(time);
        stats.jaccard_indexes.push_back(jaccard);
        stats.ndcg_scores.push_back(ndcg);
        stats.results.push_back(search_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  auto runUint32SearcherBenchmark = [&](const std::string &name, auto &searcher,
                                        simsearch::SearcherType type,
                                        const auto &getQuery) {
    if (searcher.isInitialized()) {
      std::cout << "Running " << name << " searches...";
      SearchMethodStats<uint32_t> stats(name);
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results = searcher.similarity_search(q, config.k);
        auto end = std::chrono::high_resolution_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        auto jaccard = EmbeddingUtils::calculateJaccardIndex(baseResults[i],
                                                             search_results);
        auto ndcg =
            EmbeddingUtils::calculateNDCG(baseResults[i], search_results);
        results.times[type] += time;
        results.jaccardIndexes[type] += jaccard;
        results.NDCG[type] += ndcg;
        stats.times.push_back(time);
        stats.jaccard_indexes.push_back(jaccard);
        stats.ndcg_scores.push_back(ndcg);
        stats.results.push_back(search_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  auto runIntSearcherBenchmark = [&](const std::string &name, auto &searcher,
                                     simsearch::SearcherType type,
                                     const auto &getQuery) {
    if (searcher.isInitialized()) {
      std::cout << "Running " << name << " searches...";
      SearchMethodStats<int> stats(name);
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results = searcher.similarity_search(q, config.k);
        auto end = std::chrono::high_resolution_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        auto jaccard = EmbeddingUtils::calculateJaccardIndex(baseResults[i],
                                                             search_results);
        auto ndcg =
            EmbeddingUtils::calculateNDCG(baseResults[i], search_results);
        results.times[type] += time;
        results.jaccardIndexes[type] += jaccard;
        results.NDCG[type] += ndcg;
        stats.times.push_back(time);
        stats.jaccard_indexes.push_back(jaccard);
        stats.ndcg_scores.push_back(ndcg);
        stats.results.push_back(search_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  auto runSearcherBenchmarkTwoStep = [&](const std::string &name,
                                         auto &searcher, auto &searcherRescore,
                                         simsearch::SearcherType type,
                                         const auto &getQuery,
                                         const auto &getQueryRescore) {
    if (searcher.isInitialized() && searcherRescore.isInitialized()) {
      std::cout << "Running " << name << " 2step searches...";
      SearchMethodStats<int> stats(name);
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        auto q_rescore = getQueryRescore(randomIndexes[i]);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results =
            searcher.similarity_search(q, config.k * config.rescoring_factor);
        auto rescore_results = searcherRescore.similarity_search(
            q_rescore, config.k, search_results);
        auto end = std::chrono::high_resolution_clock::now();

        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        auto jaccard = EmbeddingUtils::calculateJaccardIndex(baseResults[i],
                                                             rescore_results);
        auto ndcg =
            EmbeddingUtils::calculateNDCG(baseResults[i], rescore_results);

        results.times[type] += time;
        results.jaccardIndexes[type] += jaccard;
        results.NDCG[type] += ndcg;
        stats.times.push_back(time);
        stats.jaccard_indexes.push_back(jaccard);
        stats.ndcg_scores.push_back(ndcg);
        stats.results.push_back(search_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  // Run benchmarks for each searcher type sequentially
  runFloatSearcherBenchmark(
      "PCA2", searchers.pca2, simsearch::F32_PCA2,
      [&](size_t idx) { return searchers.pca2.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "PCA4", searchers.pca4, simsearch::F32_PCA4,
      [&](size_t idx) { return searchers.pca4.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "PCA8", searchers.pca8, simsearch::F32_PCA8,
      [&](size_t idx) { return searchers.pca8.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "PCA16", searchers.pca16, simsearch::F32_PCA16,
      [&](size_t idx) { return searchers.pca16.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "PCA32", searchers.pca32, simsearch::F32_PCA32,
      [&](size_t idx) { return searchers.pca32.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "AVX2", searchers.avx2, simsearch::F32_AVX2,
      [&](size_t idx) { return searchers.avx2.getEmbeddings()[idx]; });

  runIntSearcherBenchmark(
      "Binary", searchers.binary, simsearch::BINARY,
      [&](size_t idx) { return searchers.binary.getEmbeddings()[idx]; });

  runIntSearcherBenchmark(
      "Binary AVX2", searchers.binary_avx2, simsearch::BINARY_AVX2,
      [&](size_t idx) { return searchers.binary_avx2.getEmbeddings()[idx]; });

  runUint32SearcherBenchmark(
      "UINT8 AVX2", searchers.uint8_avx2, simsearch::UINT8_AVX2,
      [&](size_t idx) { return searchers.uint8_avx2.getEmbeddings()[idx]; });

  runSearcherBenchmarkTwoStep(
      "OBAVX2_F32OAVX2", searchers.obinary_avx2, searchers.oavx2,
      simsearch::OBAVX2_F32OAVX2,
      [&](size_t idx) { return searchers.obinary_avx2.getEmbeddingAVX2(idx); },
      [&](size_t idx) { return searchers.oavx2.getEmbedding(idx); });

  runFloatSearcherBenchmark(
      "OAVX2", searchers.oavx2, simsearch::F32_OAVX2,
      [&](size_t idx) { return searchers.oavx2.getEmbedding(idx); });

  runIntSearcherBenchmark(
      "O Binary AVX2", searchers.obinary_avx2, simsearch::OBINAR_AVX2,
      [&](size_t idx) { return searchers.obinary_avx2.getEmbeddingAVX2(idx); });

  runUint32SearcherBenchmark(
      "O UINT8 AVX2", searchers.ouint8_avx2, simsearch::OUINT8_AVX2,
      [&](size_t idx) { return searchers.ouint8_avx2.getEmbeddingAVX2(idx); });

  /*runFloatSearcherBenchmark(
      "MAPPED_FLOAT", searchers.mappedFloat, simsearch::MAPPED_FLOAT,
      [&](size_t idx) { return searchers.mappedFloat.getEmbeddings()[idx]; });*/

  /*runFloatSearcherBenchmark(
      "MAPPED_FLOAT2", searchers.mappedFloat2, simsearch::MAPPED_FLOAT2,
      [&](size_t idx) { return searchers.base.getEmbeddings()[idx]; });*/

  long timestamp = std::time(nullptr);
  std::string strTimestmap = std::to_string(timestamp);
  std::string filename = "random_emb_search_results_" + strTimestmap + ".json";
  // Export results
  exportQuerySearchResults(floatResults, uint32Results, intResults, filename,
                           config, searchers);

  return results;
}

void printResults(const simsearch::BenchmarkResults &results,
                  const simsearch::BenchmarkConfig &config,
                  const simsearch::Searchers &searchers) {
  const std::vector<std::string> names = {
      "F32",          "F32_PCA2",         "F32_PCA4",     "F32_PCA2x2",
      "F32_PCA6",     "F32_PCA8",         "F32_PCA16",    "F32_PCA32",
      "F32_AVX2",     "F32_AVX2_PCA8",    "F32_OAVX2",    "BINARY",
      "BINARY_AVX2",  "BINARY_AVX2_PCA6", "OBINARY_AVX2", "OBAVX2_F32OAVX2",
      "UINT8_AVX2",   "OUINT8_AVX2",      "FLOAT_INT8",   "FLOAT16",
      "MAPPED_FLOAT", "MAPPED_FLOAT2"};

  std::cout << "Configuration:\n"
            << "Runs: " << config.runs << " | k: " << config.k
            << " | Rescoring factor: " << config.rescoring_factor
            << " | Embeddings: " << searchers.base.getEmbeddings().size()
            << " | Dimensions: " << searchers.base.getEmbeddings()[0].size()
            << "\n\nAverage Times:\n";

  for (int i = 0; i < names.size(); i++) {
    if (results.times[i] != -1) {
      std::cout << std::left << std::setw(15) << names[i] << ": "
                << results.times[i] / config.runs << "us\n";
    }
  }

  std::cout << "\nAverage Jaccard Index (compared to F32):\n";
  for (int i = 1; i < names.size(); i++) {
    if (results.times[i] != -1) {
      std::cout << std::left << std::setw(15) << names[i] << ": "
                << results.jaccardIndexes[i] / config.runs << "\n";
    }
  }

  std::cout << "\nAverage NDCG Index (compared to F32):\n";
  for (int i = 1; i < names.size(); i++) {
    if (results.times[i] != -1) {
      std::cout << std::left << std::setw(15) << names[i] << ": "
                << results.NDCG[i] / config.runs << "\n";
    }
  }
}

std::vector<Query> loadQueries(const std::string &filename) {
  std::vector<Query> queries;
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Failed to open query file: " + filename);
  }

  std::string line;
  int lineNumber = 0;
  while (std::getline(file, line)) {
    try {
      json j = json::parse(line);
      Query q;
      q.query = j.at("query").get<std::string>();
      q.formatted_query = j.at("formatted_query").get<std::string>();
      q.embedding = j.at("embedding").get<std::vector<float>>();
      queries.push_back(q);
      lineNumber++;
    } catch (json::parse_error &e) {
      std::cerr << "Parse error on line " << lineNumber << ": " << e.what()
                << std::endl;
    } catch (json::out_of_range &e) {
      std::cerr << "JSON key error on line " << lineNumber << ": " << e.what()
                << std::endl;
    }
  }

  if (queries.empty()) {
    throw std::runtime_error("No valid queries found in file: " + filename);
  }

  std::cout << "Loaded " << queries.size() << " queries from " << filename
            << std::endl;
  return queries;
}
template <typename NumberType>
void printSearchResults(
    const std::string &query, const std::string &formatted_query,
    const std::vector<std::pair<NumberType, size_t>> &results,
    const std::vector<std::string> &sentences) {
  std::cout << "\nQuery: " << query << "\n";
  std::cout << "Formatted query: " << formatted_query << "\n";
  std::cout << "Results (similarity, index, text):\n";
  for (const auto &result : results) {
    std::cout << std::fixed << std::setprecision(4) << result.first << "\t"
              << result.second << "\t"
              << sentences[result.second].substr(0, 1024)
              << "\nwwwwwwwwwwwww\n";
  }
  std::cout << "-------------------\n";
}

void runQuerySearch(simsearch::Searchers &searchers,
                    const std::string &query_file, size_t k,
                    size_t rescoring_factor = 10) {
  // Load and validate queries
  std::vector<Query> queries = loadQueries(query_file);
  if (queries.empty()) {
    throw std::runtime_error("No queries loaded");
  }

  SearchMethodStats<float> f32Results{"F32_base"};
  SearchMethodStats<float> avx2Results{"AVX2_optimized"};
  SearchMethodStats<int> binaryResults{"Binary_AVX2"};
  SearchMethodStats<uint32_t> uint8Results{"UINT8_AVX2"};
  SearchMethodStats<float> twostepResults{"Two_Step"};

  // Process each query
  for (size_t i = 0; i < queries.size(); i++) {
    std::cout << "\n=== Processing Query " << (i + 1) << " of "
              << queries.size() << " ===\n";
    std::cout << "Query: " << queries[i].query << "\n";
    std::cout << "Formatted query: " << queries[i].formatted_query << "\n\n";

    // Normalize query vector
    std::vector<float> normalized_query = queries[i].embedding;
    float norm = EmbeddingUtils::calcNorm(normalized_query);
    for (float &val : normalized_query) {
      val = val / norm;
    }

    // Convert query for different formats
    avx2_vector queryAvx2(normalized_query.size() / 8);
    EmbeddingUtils::convertSingleEmbeddingToAVX2(normalized_query, queryAvx2,
                                                 normalized_query.size() / 8);

    avx2i_vector queryBinaryAvx2(normalized_query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(
        normalized_query, queryBinaryAvx2, normalized_query.size() / 8 / 32);

    avx2i_vector queryUint8Avx2(normalized_query.size() / 8 / 4);
    EmbeddingUtils::convertSingleFloatToUint8AVX2(
        normalized_query, queryUint8Avx2, normalized_query.size() / 8 / 4);

    // F32 (base) search
    {
      auto start = std::chrono::high_resolution_clock::now();
      auto results = searchers.base.similarity_search(normalized_query, k);
      auto end = std::chrono::high_resolution_clock::now();

      f32Results.times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
      f32Results.results.push_back(results);
    }

    // opt AVX2 search
    {
      auto start = std::chrono::high_resolution_clock::now();
      // auto results = searchers.avx2.similarity_search(queryAvx2, k);
      auto results = searchers.oavx2.similarity_search(normalized_query, k);
      auto end = std::chrono::high_resolution_clock::now();

      avx2Results.times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
      avx2Results.results.push_back(results);
      avx2Results.jaccard_indexes.push_back(
          EmbeddingUtils::calculateJaccardIndex(f32Results.results.back(),
                                                results));
      avx2Results.ndcg_scores.push_back(
          EmbeddingUtils::calculateNDCG(f32Results.results.back(), results));
    }

    // opt Binary AVX2 search
    {
      auto start = std::chrono::high_resolution_clock::now();
      auto results =
          searchers.obinary_avx2.similarity_search(queryBinaryAvx2, k);
      auto end = std::chrono::high_resolution_clock::now();

      binaryResults.times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
      binaryResults.results.push_back(results);
      binaryResults.jaccard_indexes.push_back(
          EmbeddingUtils::calculateJaccardIndex(f32Results.results.back(),
                                                results));
      binaryResults.ndcg_scores.push_back(
          EmbeddingUtils::calculateNDCG(f32Results.results.back(), results));
    }

    // opt UINT8 AVX2 search
    {
      auto start = std::chrono::high_resolution_clock::now();
      auto results = searchers.ouint8_avx2.similarity_search(queryUint8Avx2, k);
      auto end = std::chrono::high_resolution_clock::now();

      uint8Results.times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
      uint8Results.results.push_back(results);
      uint8Results.jaccard_indexes.push_back(
          EmbeddingUtils::calculateJaccardIndex(f32Results.results.back(),
                                                results));
      uint8Results.ndcg_scores.push_back(
          EmbeddingUtils::calculateNDCG(f32Results.results.back(), results));
    }

    // Two-step search
    {
      auto start = std::chrono::high_resolution_clock::now();
      auto results_binary = searchers.obinary_avx2.similarity_search(
          queryBinaryAvx2, k * rescoring_factor);
      auto results = searchers.oavx2.similarity_search(normalized_query, k,
                                                       results_binary);
      auto end = std::chrono::high_resolution_clock::now();

      twostepResults.times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
      twostepResults.results.push_back(results);
      twostepResults.jaccard_indexes.push_back(
          EmbeddingUtils::calculateJaccardIndex(f32Results.results.back(),
                                                results));
      twostepResults.ndcg_scores.push_back(
          EmbeddingUtils::calculateNDCG(f32Results.results.back(), results));
    }

    // Print results for current query
    auto printResults =
        [&](const std::string &name, const auto &results,
            const std::vector<int64_t> &times,
            const std::vector<double> &jaccardIndexes = std::vector<double>(),
            const std::vector<double> &ndcgScores = std::vector<double>()) {
          size_t idx = times.size() - 1;
          std::cout << "\n" << name << " Results:\n";
          std::cout << "Time: " << times[idx] << "us\n";
          if (!jaccardIndexes.empty()) {
            std::cout << "Jaccard Index: " << jaccardIndexes[idx] << "\n";
            std::cout << "NDCG Score: " << ndcgScores[idx] << "\n";
          }
          std::cout << "Top " << k << " matches:\n";
          for (const auto &match : results[idx]) {
            std::cout << "Score: " << match.first << ", Text: "
                      << searchers.base.getSentences()[match.second] << "...\n";
          }
          std::cout << "-------------------\n";
        };

    printResults("F32 (base)", f32Results.results, f32Results.times);
    printResults("OAVX2", avx2Results.results, avx2Results.times,
                 avx2Results.jaccard_indexes, avx2Results.ndcg_scores);
    printResults("OBinary AVX2", binaryResults.results, binaryResults.times,
                 binaryResults.jaccard_indexes, binaryResults.ndcg_scores);
    printResults("OUINT8 AVX2", uint8Results.results, uint8Results.times,
                 uint8Results.jaccard_indexes, uint8Results.ndcg_scores);
    printResults("Two-step bin+float", twostepResults.results,
                 twostepResults.times, twostepResults.jaccard_indexes,
                 twostepResults.ndcg_scores);
  }

  std::cout << "Configuration:\n"
            << "Runs: " << queries.size() << " | k: " << k
            << " | Rescoring factor: " << rescoring_factor
            << " | Embeddings: " << searchers.base.getEmbeddings().size()
            << " | Dimensions: " << searchers.base.getEmbeddings()[0].size()
            << "\n\nAverage Times:\n";

  // Print aggregate statistics
  std::cout << "\n=== Aggregate Statistics ===\n";

  auto printAggregateStats =
      [](const std::string &name, const std::vector<int64_t> &times,
         const std::vector<double> &jaccardIndexes = std::vector<double>(),
         const std::vector<double> &ndcgScores = std::vector<double>()) {
        double avgTime =
            std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::cout << "\n" << name << " Statistics:\n";
        std::cout << "Average Time: " << avgTime << "us\n";

        if (!jaccardIndexes.empty()) {
          double avgJaccard = std::accumulate(jaccardIndexes.begin(),
                                              jaccardIndexes.end(), 0.0) /
                              jaccardIndexes.size();
          double avgNDCG =
              std::accumulate(ndcgScores.begin(), ndcgScores.end(), 0.0) /
              ndcgScores.size();

          std::cout << "Average Jaccard Index: " << avgJaccard << "\n";
          std::cout << "Average NDCG Score: " << avgNDCG << "\n";
        }
      };

  printAggregateStats("F32 (base)", f32Results.times);
  printAggregateStats("AVX2", avx2Results.times, avx2Results.jaccard_indexes,
                      avx2Results.ndcg_scores);
  printAggregateStats("Binary AVX2", binaryResults.times,
                      binaryResults.jaccard_indexes, binaryResults.ndcg_scores);
  printAggregateStats("UINT8 AVX2", uint8Results.times,
                      uint8Results.jaccard_indexes, uint8Results.ndcg_scores);
  printAggregateStats("Two-step bin+float", twostepResults.times,
                      twostepResults.jaccard_indexes,
                      twostepResults.ndcg_scores);

  // Get current timestamp
  long timestamp = std::time(nullptr);
  std::string strTimestmap = std::to_string(timestamp);
  std::string filename = "query_search_results_" + strTimestmap + ".json";
  exportQuerySearchResults(queries, f32Results, avx2Results, binaryResults,
                           uint8Results, twostepResults, filename,
                           {k, queries.size(), rescoring_factor}, searchers);
}

void printUsage(const char *programName) {
  std::cout << "Usage: " << programName << " [options]\n"
            << "\nOptions:\n"
            << "  -f, --file <path>              Input file path (required)\n"
            << "  -q, --query-file <path>        Query file path (optional)\n"
            << "  -k, --topk <number>            Number of similar vectors to "
               "retrieve (default: 25)\n"
            << "  -r, --runs <number>            Number of benchmark runs "
               "(default: 500)\n"
            << "  -s, --rescoring-factor <number> Rescoring factor for "
               "two-step search (default: 50)\n"
            << "  -h, --help                     Show this help message\n"
            << "\nExamples:\n"
            << "  " << programName
            << " -f embeddings.jsonl -k 10 -r 100 -s 25\n"
            << "  " << programName
            << " -f embeddings.jsonl -q queries.jsonl -k 10\n";
}

struct CommandLineArgs {
  std::string filename;
  std::string query_file;
  size_t k = 25;
  size_t runs = 500;
  size_t rescoring_factor = 50;
  bool valid = false;
  bool is_query_mode = false;

  static CommandLineArgs parse(int argc, char *argv[]) {
    CommandLineArgs args;

    if (argc < 2) {
      std::cerr << "Error: No arguments provided.\n\n";
      printUsage(argv[0]);
      return args;
    }

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "-h" || arg == "--help") {
        printUsage(argv[0]);
        return args;
      }

      if (i + 1 >= argc) {
        std::cerr << "Error: Missing value for " << arg << "\n\n";
        printUsage(argv[0]);
        return args;
      }

      if (arg == "-f" || arg == "--file") {
        args.filename = argv[++i];
      } else if (arg == "-q" || arg == "--query-file") {
        args.query_file = argv[++i];
        args.is_query_mode = true;
      } else if (arg == "-k" || arg == "--topk") {
        try {
          args.k = std::stoul(argv[++i]);
          if (args.k == 0)
            throw std::out_of_range("k must be positive");
        } catch (const std::exception &e) {
          std::cerr << "Error: Invalid value for k: " << e.what() << "\n\n";
          printUsage(argv[0]);
          return args;
        }
      } else if (arg == "-r" || arg == "--runs") {
        try {
          args.runs = std::stoul(argv[++i]);
          if (args.runs == 0)
            throw std::out_of_range("runs must be positive");
        } catch (const std::exception &e) {
          std::cerr << "Error: Invalid value for runs: " << e.what() << "\n\n";
          printUsage(argv[0]);
          return args;
        }
      } else if (arg == "-s" || arg == "--rescoring-factor") {
        try {
          args.rescoring_factor = std::stoul(argv[++i]);
          if (args.rescoring_factor == 0)
            throw std::out_of_range("rescoring factor must be positive");
        } catch (const std::exception &e) {
          std::cerr << "Error: Invalid value for rescoring factor: " << e.what()
                    << "\n\n";
          printUsage(argv[0]);
          return args;
        }
      } else {
        std::cerr << "Error: Unknown argument: " << arg << "\n\n";
        printUsage(argv[0]);
        return args;
      }
    }

    // Validate required arguments
    if (args.filename.empty()) {
      std::cerr << "Error: Input file path is required\n\n";
      printUsage(argv[0]);
      return args;
    }

    if (!std::filesystem::exists(args.filename)) {
      std::cerr << "Error: File not found: " << args.filename << "\n\n";
      printUsage(argv[0]);
      return args;
    }

    if (args.is_query_mode && !std::filesystem::exists(args.query_file)) {
      std::cerr << "Error: Query file not found: " << args.query_file << "\n\n";
      printUsage(argv[0]);
      return args;
    }

    args.valid = true;
    return args;
  }
};

int main(int argc, char *argv[]) {
  auto args = CommandLineArgs::parse(argc, argv);
  if (!args.valid) {
    return 1;
  }

  try {
    ConfigManager::getInstance().initialize("../config.json");
    const auto &config = ConfigRef::get();

    simsearch::Searchers searchers;
    std::cout << "Initializing searchers with file: " << args.filename << "\n";
    initializeSearchers(searchers, args.filename);

    if (args.is_query_mode) {
      std::cout << "number of embds: " << searchers.base.getEmbeddings().size()
                << std::endl;
      // Run in query search mode
      std::cout << "Running in query search mode...\n";
      runQuerySearch(searchers, args.query_file, args.k, args.rescoring_factor);
    } else {
      // Run in benchmark mode
      std::cout << "Running in benchmark mode...\n";
      simsearch::BenchmarkConfig benchConfig{args.k, args.runs,
                                             args.rescoring_factor};
      auto results = runBenchmark(searchers, benchConfig);
      printResults(results, benchConfig, searchers);
    }
  } catch (const config::ConfigException &e) {
    std::cerr << "Configuration error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
