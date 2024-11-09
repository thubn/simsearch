#include "aligned_types.h"
#include "config_manager.h"
#include "embedding_search_avx2.h"
#include "embedding_search_binary.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_float16.h"
#include "embedding_search_float_int8.h"
#include "embedding_search_mapped_float.h"
#include "embedding_search_uint8_avx2.h"
#include "embedding_utils.h"
#include "optimized_embedding_search_avx2.h"
#include "optimized_embedding_search_binary_avx2.h"
#include "optimized_embedding_search_uint8_avx2.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <valgrind/callgrind.h>
#include <vector>

using json = nlohmann::json;

// Benchmark configuration
struct BenchmarkConfig {
  size_t k;
  size_t runs;
  size_t rescoring_factor;
};

// Searcher types enum
enum SearcherType {
  F32,
  F32_PCA2,
  F32_PCA4,
  F32_PCA2x2,
  F32_PCA6,
  F32_PCA8,
  F32_PCA16,
  F32_PCA32,
  F32_AVX2,
  F32_AVX2_PCA8,
  F32_OAVX2,
  BINARY,
  BINARY_AVX2,
  BINARY_AVX2_PCA6,
  OBINAR_AVX2,
  OBAVX2_F32OAVX2,
  UINT8_AVX2,
  OUINT8_AVX2,
  FLOAT_INT8,
  FLOAT16,
  NUM_SEARCHER_TYPES // Used to determine array sizes
};

// Struct to hold benchmark results
struct BenchmarkResults {
  std::vector<int64_t> times;
  std::vector<double> jaccardIndexes;
  std::vector<double> NDCG;

  BenchmarkResults()
      : times(NUM_SEARCHER_TYPES, -1), jaccardIndexes(NUM_SEARCHER_TYPES, 0),
        NDCG(NUM_SEARCHER_TYPES, 0) {}
};

// Struct to hold all searchers
struct Searchers {
  EmbeddingSearchFloat base;
  OptimizedEmbeddingSearchAVX2 pca2;
  EmbeddingSearchFloat pca2x2;
  OptimizedEmbeddingSearchAVX2 pca4;
  EmbeddingSearchFloat pca6;
  OptimizedEmbeddingSearchAVX2 pca8;
  OptimizedEmbeddingSearchAVX2 pca16;
  OptimizedEmbeddingSearchAVX2 pca32;
  EmbeddingSearchAVX2 avx2;
  // EmbeddingSearchAVX2 avx2_pca8;
  EmbeddingSearchBinary binary;
  EmbeddingSearchBinaryAVX2 binary_avx2;
  EmbeddingSearchBinaryAVX2 binary_avx2_pca6;
  EmbeddingSearchUint8AVX2 uint8_avx2;
  OptimizedEmbeddingSearchAVX2 oavx2;
  OptimizedEmbeddingSearchBinaryAVX2 obinary_avx2;
  OptimizedEmbeddingSearchUint8AVX2 ouint8_avx2;
  EmbeddingSearchFloatInt8 float_int8;
  EmbeddingSearchFloat16 float16;
  EmbeddingSearchMappedFloat mappedFloat;
  Searchers() : base(), avx2() {} // Explicit initialization

  void initBase(const std::string &filename) {
    if (!base.load(filename, false)) {
      throw std::runtime_error("Failed to load base embeddings");
    }
  }

  void initPca2() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(2);
    pca2.setEmbeddings(temp.getEmbeddings());
  }
  void initPca2x2() {
    pca2x2.setEmbeddings(base.getEmbeddings());
    pca2x2.pca_dimension_reduction(2);
    pca2x2.pca_dimension_reduction(2);
  }
  void initPca4() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(4);
    pca4.setEmbeddings(temp.getEmbeddings());
  }
  void initPca6() {
    pca6.setEmbeddings(base.getEmbeddings());
    pca6.pca_dimension_reduction(6);
  }
  void initPca8() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(8);
    pca8.setEmbeddings(temp.getEmbeddings());
  }
  void initPca16() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(16);
    pca16.setEmbeddings(temp.getEmbeddings());
  }
  void initPca32() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(32);
    pca32.setEmbeddings(temp.getEmbeddings());
  }
  void initAvx2() { avx2.setEmbeddings(base.getEmbeddings()); }
  /*void initAvx2_pca8()
  {
      avx2_pca8.setEmbeddings(pca8.getEmbeddings());
  }*/
  void initBinary() { binary.setEmbeddings(base.getEmbeddings()); }
  void initBinary_avx2() { binary_avx2.setEmbeddings(base.getEmbeddings()); }
  void initBinary_avx2_pca6() {
    binary_avx2_pca6.setEmbeddings(pca6.getEmbeddings());
  }
  void initUint8_avx2() { uint8_avx2.setEmbeddings(base.getEmbeddings()); }
  void initOavx2() { oavx2.setEmbeddings(base.getEmbeddings()); }
  void initObinary_avx2() { obinary_avx2.setEmbeddings(base.getEmbeddings()); }
  void initOuint_avx2() { ouint8_avx2.setEmbeddings(base.getEmbeddings()); }
  void initFloatInt8() { float_int8.setEmbeddings(base.getEmbeddings()); }
  void initFloat16() { float16.setEmbeddings(base.getEmbeddings()); }
  void initMappedFloat() { mappedFloat.setEmbeddings(base.getEmbeddings()); }
};

void initializeSearchers(Searchers &searchers, const std::string &filename) {
  // Load base embeddings
  searchers.initBase(filename);

  // std::thread tPca2(&Searchers::initPca2, &searchers);
  // std::thread tPca2x2(&Searchers::initPca2x2, &searchers);
  // std::thread tPca4(&Searchers::initPca4, &searchers);
  // std::thread tPca6(&Searchers::initPca6, &searchers);
  // std::thread tPca8(&Searchers::initPca8, &searchers);
  // std::thread tPca16(&Searchers::initPca16, &searchers);
  // std::thread tPca32(&Searchers::initPca32, &searchers);

  /*
  std::thread tAvx2(&Searchers::initAvx2, &searchers);
  std::thread tBinary(&Searchers::initBinary, &searchers);
  std::thread tBinary_avx2(&Searchers::initBinary_avx2, &searchers);
  std::thread tUint8_avx2(&Searchers::initUint8_avx2, &searchers);
  std::thread tOavx2(&Searchers::initOavx2, &searchers);
  std::thread tObinary_avx2(&Searchers::initObinary_avx2, &searchers);
  std::thread tOuint_avx2(&Searchers::initOuint_avx2, &searchers);
  std::thread tFloat_int8(&Searchers::initFloatInt8, &searchers);
  */
  // std::thread tFloat16(&Searchers::initFloat16, &searchers);
  std::thread tMappedFloat(&Searchers::initMappedFloat, &searchers);
  // tPca8.join();
  // std::thread tAvx2_pca8(&Searchers::initAvx2_pca8, &searchers);
  // tPca6.join();
  // std::thread tBinary_avx2_pca6(&Searchers::initBinary_avx2_pca6,
  // &searchers); tPca2.join(); tPca2x2.join(); tPca4.join(); tPca16.join();
  // tPca32.join();

  /*
  tAvx2.join();
  tBinary.join();
  tBinary_avx2.join();
  tUint8_avx2.join();
  tOavx2.join();
  tObinary_avx2.join();
  tOuint_avx2.join();
  tFloat_int8.join();
  */
  // tFloat16.join();
  tMappedFloat.join();
  // tAvx2_pca8.join();
  // tBinary_avx2_pca6.join();
}

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

BenchmarkResults runBenchmark(Searchers &searchers,
                              const BenchmarkConfig &config) {
  BenchmarkResults results;

  // Generate random indexes first
  std::vector<size_t> randomIndexes = generateRandomIndexes(
      config.runs, searchers.base.getEmbeddings().size() - 1);

  // Run base F32 searches
  std::cout << "Running F32 (currently using avx2 for this!!) searches...";
  std::vector<std::vector<std::pair<float, size_t>>> baseResults(config.runs);
  for (size_t i = 0; i < config.runs; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // baseResults[i] =
    // searchers.avx2.similarity_search(searchers.avx2.getEmbeddings()[randomIndexes[i]],
    // config.k);
    baseResults[i] = searchers.base.similarity_search(
        searchers.base.getEmbeddings()[randomIndexes[i]], config.k);
    auto end = std::chrono::high_resolution_clock::now();
    results.times[F32] +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }
  std::cout << " ✓" << std::endl;

  // Lambda to run benchmark for a specific searcher type
  auto runSearcherBenchmark = [&](const std::string &name, auto &searcher,
                                  SearcherType type, const auto &getQuery) {
    if (searcher.isInitialized()) {
      std::cout << "Running " << name << " searches...";
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        CALLGRIND_START_INSTRUMENTATION;
        CALLGRIND_TOGGLE_COLLECT;
        CALLGRIND_ZERO_STATS;
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results = searcher.similarity_search(q, config.k);
        auto end = std::chrono::high_resolution_clock::now();
        CALLGRIND_DUMP_STATS_AT(name.data());
        CALLGRIND_TOGGLE_COLLECT;
        CALLGRIND_STOP_INSTRUMENTATION;

        results.times[type] +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        results.jaccardIndexes[type] += EmbeddingUtils::calculateJaccardIndex(
            baseResults[i], search_results);
        results.NDCG[type] +=
            EmbeddingUtils::calculateNDCG(baseResults[i], search_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  auto runSearcherBenchmarkTwoStep = [&](const std::string &name,
                                         auto &searcher, auto &searcherRescore,
                                         SearcherType type,
                                         const auto &getQuery,
                                         const auto &getQueryRescore) {
    if (searcher.isInitialized() && searcherRescore.isInitialized()) {
      std::cout << "Running " << name << " 2step searches...";
      for (size_t i = 0; i < config.runs; i++) {
        auto q = getQuery(randomIndexes[i]);
        auto q_rescore = getQueryRescore(randomIndexes[i]);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_results =
            searcher.similarity_search(q, config.k * config.rescoring_factor);
        auto rescore_results = searcherRescore.similarity_search(
            q_rescore, config.k, search_results);
        auto end = std::chrono::high_resolution_clock::now();

        results.times[type] +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        results.jaccardIndexes[type] += EmbeddingUtils::calculateJaccardIndex(
            baseResults[i], rescore_results);
        results.NDCG[type] +=
            EmbeddingUtils::calculateNDCG(baseResults[i], rescore_results);
      }
      std::cout << " ✓" << std::endl;
    } else {
      std::cout << name << " is not initialized..." << std::endl;
    }
  };

  // Run benchmarks for each searcher type sequentially
  runSearcherBenchmark("PCA2", searchers.pca2, F32_PCA2, [&](size_t idx) {
    return searchers.pca2.getEmbedding(idx);
  });

  runSearcherBenchmark("PCA4", searchers.pca4, F32_PCA4, [&](size_t idx) {
    return searchers.pca4.getEmbedding(idx);
  });

  runSearcherBenchmark("PCA2x2", searchers.pca2x2, F32_PCA2x2, [&](size_t idx) {
    return searchers.pca2x2.getEmbeddings()[idx];
  });

  runSearcherBenchmark("PCA6", searchers.pca6, F32_PCA6, [&](size_t idx) {
    return searchers.pca6.getEmbeddings()[idx];
  });

  runSearcherBenchmark("PCA8", searchers.pca8, F32_PCA8, [&](size_t idx) {
    return searchers.pca8.getEmbedding(idx);
  });

  runSearcherBenchmark("PCA16", searchers.pca16, F32_PCA16, [&](size_t idx) {
    return searchers.pca16.getEmbedding(idx);
  });

  runSearcherBenchmark("PCA32", searchers.pca32, F32_PCA32, [&](size_t idx) {
    return searchers.pca32.getEmbedding(idx);
  });

  runSearcherBenchmark("AVX2", searchers.avx2, F32_AVX2, [&](size_t idx) {
    return searchers.avx2.getEmbeddings()[idx];
  });

  /*runSearcherBenchmark("AVX2_PCA8", searchers.avx2_pca8, F32_AVX2_PCA8,
                       [&](size_t idx)
                       { return searchers.avx2_pca8.getEmbeddings()[idx]; });*/

  runSearcherBenchmark("Binary", searchers.binary, BINARY, [&](size_t idx) {
    return searchers.binary.getEmbeddings()[idx];
  });

  runSearcherBenchmark(
      "Binary AVX2", searchers.binary_avx2, BINARY_AVX2,
      [&](size_t idx) { return searchers.binary_avx2.getEmbeddings()[idx]; });

  runSearcherBenchmark("Binary AVX2 PCA6", searchers.binary_avx2_pca6,
                       BINARY_AVX2_PCA6, [&](size_t idx) {
                         return searchers.binary_avx2_pca6.getEmbeddings()[idx];
                       });

  runSearcherBenchmark(
      "UINT8 AVX2", searchers.uint8_avx2, UINT8_AVX2,
      [&](size_t idx) { return searchers.uint8_avx2.getEmbeddings()[idx]; });

  runSearcherBenchmarkTwoStep(
      "OBAVX2_F32OAVX2", searchers.obinary_avx2, searchers.oavx2,
      OBAVX2_F32OAVX2,
      [&](size_t idx) { return searchers.obinary_avx2.getEmbeddingAVX2(idx); },
      [&](size_t idx) { return searchers.oavx2.getEmbedding(idx); });

  runSearcherBenchmark("OAVX2", searchers.oavx2, F32_OAVX2, [&](size_t idx) {
    return searchers.oavx2.getEmbedding(idx);
  });

  runSearcherBenchmark(
      "O Binary AVX2", searchers.obinary_avx2, OBINAR_AVX2,
      [&](size_t idx) { return searchers.obinary_avx2.getEmbeddingAVX2(idx); });

  runSearcherBenchmark(
      "O UINT8 AVX2", searchers.ouint8_avx2, OUINT8_AVX2,
      [&](size_t idx) { return searchers.ouint8_avx2.getEmbeddingAVX2(idx); });

  runSearcherBenchmark(
      "FLOAT_INT8", searchers.float_int8, FLOAT_INT8,
      [&](size_t idx) { return searchers.float_int8.getEmbeddings()[idx]; });

  runSearcherBenchmark("FLOAT16", searchers.float16, FLOAT16, [&](size_t idx) {
    return searchers.float16.getEmbeddings()[idx];
  });

  return results;
}

void printResults(const BenchmarkResults &results,
                  const BenchmarkConfig &config, const Searchers &searchers) {
  const std::vector<std::string> names = {
      "F32",         "F32_PCA2",         "F32_PCA4",     "F32_PCA2x2",
      "F32_PCA6",    "F32_PCA8",         "F32_PCA16",    "F32_PCA32",
      "F32_AVX2",    "F32_AVX2_PCA8",    "F32_OAVX2",    "BINARY",
      "BINARY_AVX2", "BINARY_AVX2_PCA6", "OBINARY_AVX2", "OBAVX2_F32OAVX2",
      "UINT8_AVX2",  "OUINT8_AVX2",      "FLOAT_INT8",   "FLOAT16"};

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

struct Query {
  std::string query;
  std::string formatted_query;
  std::vector<float> embedding;
};

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

void runQuerySearch(Searchers &searchers, const std::string &query_file,
                    size_t k) {
  std::vector<Query> queries = loadQueries(query_file);

  std::cout << "\nRunning similarity search for " << queries.size()
            << " queries...\n";

  // Run searches with different methods
  for (size_t i = 0; i < queries.size(); i++) {
    std::cout << "\n=== Query " << (i + 1) << " of " << queries.size()
              << " ===\n";

    // F32 (base) search
    std::cout << "\nF32 Search Results:\n";
    auto f32_results =
        searchers.base.similarity_search(queries[i].embedding, k);
    printSearchResults(queries[i].query, queries[i].formatted_query,
                       f32_results, searchers.base.getSentences());

    /*
    // AVX2 search
    std::cout << "\nAVX2 Search Results:\n";
    auto avx2_results =
    searchers.avx2.similarity_search(searchers.avx2.floatToAvx2(queries[i].embedding),
    k); printSearchResults(queries[i].query, queries[i].formatted_query,
    avx2_results, searchers.base.getSentences());

    // Binary AVX2 search
    std::cout << "\nBinary AVX2 Search Results:\n";
    auto binaryAvx2_results =
    searchers.binary_avx2.similarity_search(searchers.binary_avx2.floatToBinaryAvx2(queries[i].embedding),
    k); printSearchResults(queries[i].query, queries[i].formatted_query,
    binaryAvx2_results, searchers.base.getSentences());
    */
  }
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

    Searchers searchers;
    std::cout << "Initializing searchers with file: " << args.filename << "\n";
    initializeSearchers(searchers, args.filename);

    if (args.is_query_mode) {
      std::cout << "number of embds: " << searchers.base.getEmbeddings().size()
                << std::endl;
      // Run in query search mode
      std::cout << "Running in query search mode...\n";
      runQuerySearch(searchers, args.query_file, args.k);
    } else {
      // Run in benchmark mode
      std::cout << "Running in benchmark mode...\n";
      BenchmarkConfig benchConfig{args.k, args.runs, args.rescoring_factor};
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
