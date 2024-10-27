#include "embedding_search_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_binary.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_uint8_avx2.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

// Benchmark configuration
struct BenchmarkConfig
{
    size_t k;
    size_t runs;
    size_t rescoring_factor;
};

// Searcher types enum
enum SearcherType
{
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
    BINARY,
    BINARY_AVX2,
    BINARY_AVX2_PCA6,
    BAVX2_F32AVX2,
    UINT8_AVX2,
    NUM_SEARCHER_TYPES // Used to determine array sizes
};

// Struct to hold benchmark results
struct BenchmarkResults
{
    std::vector<int64_t> times;
    std::vector<double> jaccardIndexes;
    std::vector<double> NDCG;

    BenchmarkResults() : times(NUM_SEARCHER_TYPES, -1), jaccardIndexes(NUM_SEARCHER_TYPES, 0), NDCG(NUM_SEARCHER_TYPES, 0) {}
};

// Struct to hold all searchers
struct Searchers
{
    EmbeddingSearchFloat base;
    EmbeddingSearchFloat pca2;
    EmbeddingSearchFloat pca2x2;
    EmbeddingSearchFloat pca4;
    EmbeddingSearchFloat pca6;
    EmbeddingSearchFloat pca8;
    EmbeddingSearchFloat pca16;
    EmbeddingSearchFloat pca32;
    EmbeddingSearchAVX2 avx2;
    EmbeddingSearchAVX2 avx2_pca8;
    EmbeddingSearchBinary binary;
    EmbeddingSearchBinaryAVX2 binary_avx2;
    EmbeddingSearchBinaryAVX2 binary_avx2_pca6;
    EmbeddingSearchUint8AVX2 uint8_avx2;
};

template <typename T1, typename T2>
double calculateJaccardIndex(const std::vector<std::pair<T1, size_t>> &set1,
                             const std::vector<std::pair<T2, size_t>> &set2)
{
    std::vector<size_t> vec1, vec2;

    for (const auto &pair : set1)
        vec1.push_back(pair.second);
    for (const auto &pair : set2)
        vec2.push_back(pair.second);

    std::sort(vec1.begin(), vec1.end());
    std::sort(vec2.begin(), vec2.end());

    vec1.erase(std::unique(vec1.begin(), vec1.end()), vec1.end());
    vec2.erase(std::unique(vec2.begin(), vec2.end()), vec2.end());

    std::vector<size_t> intersection;
    std::set_intersection(vec1.begin(), vec1.end(),
                          vec2.begin(), vec2.end(),
                          std::back_inserter(intersection));

    std::vector<size_t> union_set;
    std::set_union(vec1.begin(), vec1.end(),
                   vec2.begin(), vec2.end(),
                   std::back_inserter(union_set));

    return union_set.empty() ? 0.0 : static_cast<double>(intersection.size()) / union_set.size();
}

template <typename T1, typename T2>
double calculateNDCG(const std::vector<std::pair<T1, size_t>> &groundTruth,
                     const std::vector<std::pair<T2, size_t>> &prediction)
{
    if (groundTruth.empty() || prediction.empty())
    {
        return 0.0;
    }

    const size_t k = std::min(groundTruth.size(), prediction.size());

    // Create position lookup for ground truth
    std::unordered_map<size_t, size_t> truthPositions;
    for (size_t i = 0; i < k; ++i)
    {
        truthPositions[groundTruth[i].second] = i;
    }

    // Calculate DCG
    double dcg = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
        auto it = truthPositions.find(prediction[i].second);
        if (it != truthPositions.end())
        {
            // Calculate relevance score based on position difference
            // Perfect match (same position) gets relevance of 1.0
            // Relevance decreases based on position difference
            double positionDiff = std::abs(static_cast<double>(it->second) - i);
            double relevance = std::exp(-positionDiff / k); // Exponential decay

            // DCG formula: rel_i / log2(i + 2)
            dcg += relevance / std::log2(i + 2);
        }
    }

    // Calculate IDCG (ideal DCG - when order is perfect)
    double idcg = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
        idcg += 1.0 / std::log2(i + 2); // Perfect relevance of 1.0
    }

    return idcg > 0 ? dcg / idcg : 0.0;
}

void initializeSearchers(Searchers &searchers, const std::string &filename)
{
    // Load base embeddings
    searchers.base.load(filename);

    // Initialize PCA variants
    // searchers.pca2 = searchers.base;
    // searchers.pca2.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 2);

    // searchers.pca4 = searchers.base;
    // searchers.pca4.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 4);
    // searchers.pca4.unsetSentences();

    // searchers.pca2x2 = searchers.pca2;
    // searchers.pca2x2.pca_dimension_reduction(searchers.pca2.getEmbeddings()[0].size() / 2);

    // searchers.pca6 = searchers.base;
    // searchers.pca6.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 6);

    // searchers.pca8 = searchers.base;
    // searchers.pca8.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 8);
    // searchers.pca8.unsetSentences();

    // searchers.pca16 = searchers.base;
    // searchers.pca16.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 16);
    // searchers.pca16.unsetSentences();

    // searchers.pca32 = searchers.base;
    // searchers.pca32.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 32);

    // Initialize specialized variants
    searchers.avx2.setEmbeddings(searchers.base.getEmbeddings());
    searchers.uint8_avx2.setEmbeddings(searchers.base.getEmbeddings());
    // searchers.binary.create_binary_embedding_from_float(searchers.base.getEmbeddings());
    searchers.binary_avx2.create_binary_embedding_from_float(searchers.base.getEmbeddings());
    // searchers.avx2_pca8.setEmbeddings(searchers.pca8.getEmbeddings());
    // searchers.binary_avx2_pca6.create_binary_embedding_from_float(searchers.pca4.getEmbeddings());
    // searchers.pca4.unsetEmbeddings();
}

std::vector<size_t> generateRandomIndexes(size_t numRuns, size_t maxIndex)
{
    std::vector<size_t> indexes(numRuns);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, maxIndex);

    for (size_t i = 0; i < numRuns; ++i)
    {
        indexes[i] = distrib(gen);
    }

    return indexes;
}

BenchmarkResults runBenchmark(Searchers &searchers, const BenchmarkConfig &config)
{
    BenchmarkResults results;

    // Generate random indexes first
    std::vector<size_t> randomIndexes = generateRandomIndexes(config.runs, searchers.base.getEmbeddings().size() - 1);

    // Run base F32 searches
    std::cout << "Running F32 searches...";
    std::vector<std::vector<std::pair<float, size_t>>> baseResults(config.runs);
    for (size_t i = 0; i < config.runs; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        baseResults[i] = searchers.base.similarity_search(searchers.base.getEmbeddings()[randomIndexes[i]], config.k);
        auto end = std::chrono::high_resolution_clock::now();
        results.times[F32] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << " ✓" << std::endl;

    // Lambda to run benchmark for a specific searcher type
    auto runSearcherBenchmark = [&](const std::string &name, auto &searcher, SearcherType type, const auto &getQuery)
    {
        if (searcher.isInitialized())
        {
            std::cout << "Running " << name << " searches...";
            for (size_t i = 0; i < config.runs; i++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                auto search_results = searcher.similarity_search(getQuery(randomIndexes[i]), config.k);
                auto end = std::chrono::high_resolution_clock::now();

                results.times[type] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                results.jaccardIndexes[type] += calculateJaccardIndex(baseResults[i], search_results);
                results.NDCG[type] += calculateNDCG(baseResults[i], search_results);
            }
            std::cout << " ✓" << std::endl;
        }
    };

    auto runSearcherBenchmarkTwoStep = [&](const std::string &name, auto &searcher, auto &searcherRescore, SearcherType type, const auto &getQuery, const auto &getQueryRescore)
    {
        if (searcher.isInitialized() && searcherRescore.isInitialized())
        {
            std::cout << "Running " << name << " 2step searches...";
            for (size_t i = 0; i < config.runs; i++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                auto search_results = searcher.similarity_search(getQuery(randomIndexes[i]), config.k * config.rescoring_factor);
                auto rescore_results = searcherRescore.similarity_search(getQueryRescore(randomIndexes[i]), config.k, search_results);
                auto end = std::chrono::high_resolution_clock::now();

                results.times[type] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                results.jaccardIndexes[type] += calculateJaccardIndex(baseResults[i], rescore_results);
                results.NDCG[type] += calculateNDCG(baseResults[i], rescore_results);
            }
            std::cout << " ✓" << std::endl;
        }
    };

    // Run benchmarks for each searcher type sequentially
    runSearcherBenchmark("PCA2", searchers.pca2, F32_PCA2, [&](size_t idx)
                         { return searchers.pca2.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA4", searchers.pca4, F32_PCA4,
                         [&](size_t idx)
                         { return searchers.pca4.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA2x2", searchers.pca2x2, F32_PCA2x2,
                         [&](size_t idx)
                         { return searchers.pca2x2.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA6", searchers.pca6, F32_PCA6,
                         [&](size_t idx)
                         { return searchers.pca6.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA8", searchers.pca8, F32_PCA8,
                         [&](size_t idx)
                         { return searchers.pca8.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA16", searchers.pca16, F32_PCA16,
                         [&](size_t idx)
                         { return searchers.pca16.getEmbeddings()[idx]; });

    runSearcherBenchmark("PCA32", searchers.pca32, F32_PCA32,
                         [&](size_t idx)
                         { return searchers.pca32.getEmbeddings()[idx]; });

    runSearcherBenchmark("AVX2", searchers.avx2, F32_AVX2,
                         [&](size_t idx)
                         { return searchers.avx2.getEmbeddings()[idx]; });

    runSearcherBenchmark("AVX2_PCA8", searchers.avx2_pca8, F32_AVX2_PCA8,
                         [&](size_t idx)
                         { return searchers.avx2_pca8.getEmbeddings()[idx]; });

    runSearcherBenchmark("Binary", searchers.binary, BINARY,
                         [&](size_t idx)
                         { return searchers.binary.getEmbeddings()[idx]; });

    runSearcherBenchmark("Binary AVX2", searchers.binary_avx2, BINARY_AVX2,
                         [&](size_t idx)
                         { return searchers.binary_avx2.getEmbeddings()[idx]; });

    runSearcherBenchmark("Binary AVX2 PCA6", searchers.binary_avx2_pca6, BINARY_AVX2_PCA6,
                         [&](size_t idx)
                         { return searchers.binary_avx2_pca6.getEmbeddings()[idx]; });

    runSearcherBenchmark("UINT8 AVX2", searchers.uint8_avx2, UINT8_AVX2,
                         [&](size_t idx)
                         { return searchers.uint8_avx2.getEmbeddings()[idx]; });

    runSearcherBenchmarkTwoStep("BAVX2_F32AVX2", searchers.binary_avx2, searchers.avx2, BAVX2_F32AVX2, [&](size_t idx)
                                { return searchers.binary_avx2.getEmbeddings()[idx]; }, [&](size_t idx)
                                { return searchers.avx2.getEmbeddings()[idx]; });

    return results;
}

void printResults(const BenchmarkResults &results, const BenchmarkConfig &config, const Searchers &searchers)
{
    const std::vector<std::string> names = {
        "F32", "F32_PCA2", "F32_PCA4", "F32_PCA2x2", "F32_PCA6", "F32_PCA8",
        "F32_PCA16", "F32_PCA32", "F32_AVX2", "F32_AVX2_PCA8", "BINARY",
        "BINARY_AVX2", "BINARY_AVX2_PCA6", "BAVX2_F32AVX2", "UINT8_AVX2"};

    std::cout << "Configuration:\n"
              << "Runs: " << config.runs
              << " | k: " << config.k
              << " | Rescoring factor: " << config.rescoring_factor
              << " | Embeddings: " << searchers.base.getEmbeddings().size()
              << " | Dimensions: " << searchers.base.getEmbeddings()[0].size()
              << "\n\nAverage Times:\n";

    for (int i = 0; i < names.size(); i++)
    {
        if (results.times[i] != -1)
        {
            std::cout << std::left << std::setw(15) << names[i]
                      << ": " << results.times[i] / config.runs << "us\n";
        }
    }

    std::cout << "\nAverage Jaccard Index (compared to F32):\n";
    for (int i = 1; i < names.size(); i++)
    {
        if (results.times[i] != -1)
        {
            std::cout << std::left << std::setw(15) << names[i]
                      << ": " << results.jaccardIndexes[i] / config.runs << "\n";
        }
    }

    std::cout << "\nAverage NDCG Index (compared to F32):\n";
    for (int i = 1; i < names.size(); i++)
    {
        if (results.times[i] != -1)
        {
            std::cout << std::left << std::setw(15) << names[i]
                      << ": " << results.NDCG[i] / config.runs << "\n";
        }
    }
}

struct Query
{
    std::string query;
    std::string formatted_query;
    std::vector<float> embedding;
};

std::vector<Query> loadQueries(const std::string &filename)
{
    std::vector<Query> queries;
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open query file: " + filename);
    }

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line))
    {
        try
        {
            json j = json::parse(line);
            Query q;
            q.query = j.at("query").get<std::string>();
            q.formatted_query = j.at("formatted_query").get<std::string>();
            q.embedding = j.at("embedding").get<std::vector<float>>();
            queries.push_back(q);
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

    if (queries.empty())
    {
        throw std::runtime_error("No valid queries found in file: " + filename);
    }

    std::cout << "Loaded " << queries.size() << " queries from " << filename << std::endl;
    return queries;
}
template <typename NumberType>
void printSearchResults(const std::string &query, const std::string &formatted_query,
                        const std::vector<std::pair<NumberType, size_t>> &results,
                        const std::vector<std::string> &sentences)
{
    std::cout << "\nQuery: " << query << "\n";
    std::cout << "Formatted query: " << formatted_query << "\n";
    std::cout << "Results (similarity, index, text):\n";
    for (const auto &result : results)
    {
        std::cout << std::fixed << std::setprecision(4)
                  << result.first << "\t"
                  << result.second << "\t"
                  << sentences[result.second].substr(0, 1024) << "\nwwwwwwwwwwwww\n";
    }
    std::cout << "-------------------\n";
}

void runQuerySearch(Searchers &searchers, const std::string &query_file, size_t k)
{
    std::vector<Query> queries = loadQueries(query_file);

    std::cout << "\nRunning similarity search for " << queries.size() << " queries...\n";

    // Run searches with different methods
    for (size_t i = 0; i < queries.size(); i++)
    {
        std::cout << "\n=== Query " << (i + 1) << " of " << queries.size() << " ===\n";

        // F32 (base) search
        std::cout << "\nF32 Search Results:\n";
        auto f32_results = searchers.base.similarity_search(queries[i].embedding, k);
        printSearchResults(queries[i].query, queries[i].formatted_query, f32_results, searchers.base.getSentences());

        /*
        // AVX2 search
        std::cout << "\nAVX2 Search Results:\n";
        std::vector<__m256> avx2_query;
        avx2_query.reserve(queries[i].embedding.size() / 8);
        for (size_t j = 0; j < queries[i].embedding.size(); j += 8)
        {
            avx2_query.push_back(_mm256_loadu_ps(&queries[i].embedding[j]));
        }
        auto avx2_results = searchers.avx2.similarity_search(avx2_query, k);
        printSearchResults(queries[i].query, queries[i].formatted_query, avx2_results, searchers.avx2.getSentences());

        // Binary search
        std::cout << "\nBinary Search Results:\n";
        std::vector<uint64_t> binary_query;
        binary_query.resize((queries[i].embedding.size() + 63) / 64, 0);
        for (size_t j = 0; j < queries[i].embedding.size(); j++)
        {
            if (queries[i].embedding[j] >= 0)
            {
                binary_query[j / 64] |= (1ULL << (63 - (j % 64)));
            }
        }
        auto binary_results = searchers.binary.similarity_search(binary_query, k);
        printSearchResults(queries[i].query, queries[i].formatted_query, binary_results, searchers.binary.getSentences());
        */
    }
}

void printUsage(const char *programName)
{
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  -f, --file <path>              Input file path (required)\n"
              << "  -q, --query-file <path>        Query file path (optional)\n"
              << "  -k, --topk <number>            Number of similar vectors to retrieve (default: 25)\n"
              << "  -r, --runs <number>            Number of benchmark runs (default: 500)\n"
              << "  -s, --rescoring-factor <number> Rescoring factor for two-step search (default: 50)\n"
              << "  -h, --help                     Show this help message\n"
              << "\nExamples:\n"
              << "  " << programName << " -f embeddings.jsonl -k 10 -r 100 -s 25\n"
              << "  " << programName << " -f embeddings.jsonl -q queries.jsonl -k 10\n";
}

struct CommandLineArgs
{
    std::string filename;
    std::string query_file;
    size_t k = 25;
    size_t runs = 500;
    size_t rescoring_factor = 50;
    bool valid = false;
    bool is_query_mode = false;

    static CommandLineArgs parse(int argc, char *argv[])
    {
        CommandLineArgs args;

        if (argc < 2)
        {
            std::cerr << "Error: No arguments provided.\n\n";
            printUsage(argv[0]);
            return args;
        }

        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help")
            {
                printUsage(argv[0]);
                return args;
            }

            if (i + 1 >= argc)
            {
                std::cerr << "Error: Missing value for " << arg << "\n\n";
                printUsage(argv[0]);
                return args;
            }

            if (arg == "-f" || arg == "--file")
            {
                args.filename = argv[++i];
            }
            else if (arg == "-q" || arg == "--query-file")
            {
                args.query_file = argv[++i];
                args.is_query_mode = true;
            }
            else if (arg == "-k" || arg == "--topk")
            {
                try
                {
                    args.k = std::stoul(argv[++i]);
                    if (args.k == 0)
                        throw std::out_of_range("k must be positive");
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: Invalid value for k: " << e.what() << "\n\n";
                    printUsage(argv[0]);
                    return args;
                }
            }
            else if (arg == "-r" || arg == "--runs")
            {
                try
                {
                    args.runs = std::stoul(argv[++i]);
                    if (args.runs == 0)
                        throw std::out_of_range("runs must be positive");
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: Invalid value for runs: " << e.what() << "\n\n";
                    printUsage(argv[0]);
                    return args;
                }
            }
            else if (arg == "-s" || arg == "--rescoring-factor")
            {
                try
                {
                    args.rescoring_factor = std::stoul(argv[++i]);
                    if (args.rescoring_factor == 0)
                        throw std::out_of_range("rescoring factor must be positive");
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: Invalid value for rescoring factor: " << e.what() << "\n\n";
                    printUsage(argv[0]);
                    return args;
                }
            }
            else
            {
                std::cerr << "Error: Unknown argument: " << arg << "\n\n";
                printUsage(argv[0]);
                return args;
            }
        }

        // Validate required arguments
        if (args.filename.empty())
        {
            std::cerr << "Error: Input file path is required\n\n";
            printUsage(argv[0]);
            return args;
        }

        if (!std::filesystem::exists(args.filename))
        {
            std::cerr << "Error: File not found: " << args.filename << "\n\n";
            printUsage(argv[0]);
            return args;
        }

        if (args.is_query_mode && !std::filesystem::exists(args.query_file))
        {
            std::cerr << "Error: Query file not found: " << args.query_file << "\n\n";
            printUsage(argv[0]);
            return args;
        }

        args.valid = true;
        return args;
    }
};

int main(int argc, char *argv[])
{
    auto args = CommandLineArgs::parse(argc, argv);
    if (!args.valid)
    {
        return 1;
    }

    try
    {
        Searchers searchers;
        std::cout << "Initializing searchers with file: " << args.filename << "\n";
        initializeSearchers(searchers, args.filename);

        if (args.is_query_mode)
        {
            std::cout << "number of embds: " << searchers.base.getEmbeddings().size() << std::endl;
            // Run in query search mode
            std::cout << "Running in query search mode...\n";
            runQuerySearch(searchers, args.query_file, args.k);
        }
        else
        {
            // Run in benchmark mode
            std::cout << "Running in benchmark mode...\n";
            BenchmarkConfig config{args.k, args.runs, args.rescoring_factor};
            auto results = runBenchmark(searchers, config);
            printResults(results, config, searchers);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during execution: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
