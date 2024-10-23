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
#include <string>
#include <filesystem>

// Benchmark configuration
struct BenchmarkConfig {
    size_t k;
    size_t runs;
    size_t rescoring_factor;
};

// Searcher types enum
enum SearcherType {
    F32, F32_PCA2, F32_PCA4, F32_PCA2x2, F32_PCA6, F32_PCA8, 
    F32_PCA16, F32_PCA32, F32_AVX2, F32_AVX2_PCA8, BINARY, 
    BINARY_AVX2, BINARY_AVX2_PCA6, BAVX2_F32AVX2, UINT8_AVX2,
    NUM_SEARCHER_TYPES  // Used to determine array sizes
};

// Struct to hold benchmark results
struct BenchmarkResults {
    std::vector<int64_t> times;
    std::vector<double> jaccardIndexes;
    
    BenchmarkResults() : times(NUM_SEARCHER_TYPES, 0), jaccardIndexes(NUM_SEARCHER_TYPES, 0) {}
};

// Struct to hold all searchers
struct Searchers {
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
double calculateJaccardIndex(const std::vector<std::pair<T1, size_t>>& set1,
                           const std::vector<std::pair<T2, size_t>>& set2) {
    std::vector<size_t> vec1, vec2;
    
    for (const auto& pair : set1) vec1.push_back(pair.second);
    for (const auto& pair : set2) vec2.push_back(pair.second);
    
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

void initializeSearchers(Searchers& searchers, const std::string& filename) {
    // Load base embeddings
    searchers.base.load(filename);
    
    // Initialize PCA variants
    searchers.pca2 = searchers.base;
    searchers.pca2.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 2);
    
    searchers.pca4 = searchers.base;
    searchers.pca4.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 4);
    
    searchers.pca2x2 = searchers.pca2;
    searchers.pca2x2.pca_dimension_reduction(searchers.pca2.getEmbeddings()[0].size() / 2);
    
    searchers.pca6 = searchers.base;
    searchers.pca6.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 6);
    
    searchers.pca8 = searchers.base;
    searchers.pca8.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 8);
    
    searchers.pca16 = searchers.base;
    searchers.pca16.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 16);
    
    searchers.pca32 = searchers.base;
    searchers.pca32.pca_dimension_reduction(searchers.base.getEmbeddings()[0].size() / 32);
    
    // Initialize specialized variants
    searchers.avx2.setEmbeddings(searchers.base.getEmbeddings());
    searchers.avx2_pca8.setEmbeddings(searchers.pca8.getEmbeddings());
    searchers.binary.create_binary_embedding_from_float(searchers.base.getEmbeddings());
    searchers.binary_avx2.create_binary_embedding_from_float(searchers.base.getEmbeddings());
    searchers.binary_avx2_pca6.create_binary_embedding_from_float(searchers.pca6.getEmbeddings());
    searchers.uint8_avx2.setEmbeddings(searchers.base.getEmbeddings());
}

BenchmarkResults runBenchmark(Searchers& searchers, const BenchmarkConfig& config) {
    BenchmarkResults results;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, searchers.base.getEmbeddings().size());

    for (size_t i = 0; i < config.runs; i++) {
        auto random_index = distrib(gen);

        // Store base results for Jaccard Index comparison
        auto start = std::chrono::high_resolution_clock::now();
        auto base_results = searchers.base.similarity_search(searchers.base.getEmbeddings()[random_index], config.k);
        auto end = std::chrono::high_resolution_clock::now();
        results.times[F32] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Lambda to measure time and update results
        auto benchmark = [&](auto& searcher, auto& query, SearcherType type) {
            start = std::chrono::high_resolution_clock::now();
            auto search_results = searcher.similarity_search(query, config.k);
            end = std::chrono::high_resolution_clock::now();
            
            results.times[type] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            results.jaccardIndexes[type] += calculateJaccardIndex(base_results, search_results);
            return search_results;
        };

        // Run benchmarks for all variants
        benchmark(searchers.pca2, searchers.pca2.getEmbeddings()[random_index], F32_PCA2);
        benchmark(searchers.pca4, searchers.pca4.getEmbeddings()[random_index], F32_PCA4);
        benchmark(searchers.pca2x2, searchers.pca2x2.getEmbeddings()[random_index], F32_PCA2x2);
        benchmark(searchers.pca6, searchers.pca6.getEmbeddings()[random_index], F32_PCA6);
        benchmark(searchers.pca8, searchers.pca8.getEmbeddings()[random_index], F32_PCA8);
        benchmark(searchers.pca16, searchers.pca16.getEmbeddings()[random_index], F32_PCA16);
        benchmark(searchers.pca32, searchers.pca32.getEmbeddings()[random_index], F32_PCA32);
        benchmark(searchers.avx2, searchers.avx2.getEmbeddings()[random_index], F32_AVX2);
        benchmark(searchers.avx2_pca8, searchers.avx2_pca8.getEmbeddings()[random_index], F32_AVX2_PCA8);
        benchmark(searchers.binary, searchers.binary.getEmbeddings()[random_index], BINARY);
        benchmark(searchers.binary_avx2, searchers.binary_avx2.getEmbeddings()[random_index], BINARY_AVX2);
        benchmark(searchers.binary_avx2_pca6, searchers.binary_avx2_pca6.getEmbeddings()[random_index], BINARY_AVX2_PCA6);
        benchmark(searchers.uint8_avx2, searchers.uint8_avx2.getEmbeddings()[random_index], UINT8_AVX2);

        // Special case for BAVX2_F32AVX2 (two-step search)
        start = std::chrono::high_resolution_clock::now();
        auto binary_avx2_rescore_results = searchers.binary_avx2.similarity_search(searchers.binary_avx2.getEmbeddings()[random_index], config.k * config.rescoring_factor);
        auto avx2_rescore_results = searchers.avx2.similarity_search(searchers.avx2.getEmbeddings()[random_index], config.k, binary_avx2_rescore_results);
        end = std::chrono::high_resolution_clock::now();
        results.times[BAVX2_F32AVX2] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        results.jaccardIndexes[BAVX2_F32AVX2] += calculateJaccardIndex(base_results, avx2_rescore_results);
    }

    return results;
}

void printResults(const BenchmarkResults& results, const BenchmarkConfig& config, const Searchers& searchers) {
    const std::vector<std::string> names = {
        "F32", "F32_PCA2", "F32_PCA4", "F32_PCA2x2", "F32_PCA6", "F32_PCA8", 
        "F32_PCA16", "F32_PCA32", "F32_AVX2", "F32_AVX2_PCA8", "BINARY", 
        "BINARY_AVX2", "BINARY_AVX2_PCA6", "BAVX2_F32AVX2", "UINT8_AVX2"
    };

    std::cout << "Configuration:\n"
              << "Runs: " << config.runs
              << " | k: " << config.k
              << " | Rescoring factor: " << config.rescoring_factor
              << " | Embeddings: " << searchers.base.getEmbeddings().size()
              << " | Dimensions: " << searchers.base.getEmbeddings()[0].size()
              << "\n\nAverage Times:\n";

    for (int i = 0; i < names.size(); i++) {
        std::cout << std::left << std::setw(15) << names[i] 
                  << ": " << results.times[i] / config.runs << "us\n";
    }

    std::cout << "\nAverage Jaccard Index (compared to F32):\n";
    for (int i = 1; i < names.size(); i++) {
        std::cout << std::left << std::setw(15) << names[i]
                  << ": " << results.jaccardIndexes[i] / config.runs << "\n";
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  -f, --file <path>              Input file path (required)\n"
              << "  -k, --topk <number>            Number of similar vectors to retrieve (default: 25)\n"
              << "  -r, --runs <number>            Number of benchmark runs (default: 500)\n"
              << "  -s, --rescoring-factor <number> Rescoring factor for two-step search (default: 50)\n"
              << "  -h, --help                     Show this help message\n"
              << "\nExample:\n"
              << "  " << programName << " -f embeddings.jsonl -k 10 -r 100 -s 25\n";
}

struct CommandLineArgs {
    std::string filename;
    size_t k = 25;
    size_t runs = 500;
    size_t rescoring_factor = 50;
    bool valid = false;

    static CommandLineArgs parse(int argc, char* argv[]) {
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
            } else if (arg == "-k" || arg == "--topk") {
                try {
                    args.k = std::stoul(argv[++i]);
                    if (args.k == 0) throw std::out_of_range("k must be positive");
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value for k: " << e.what() << "\n\n";
                    printUsage(argv[0]);
                    return args;
                }
            } else if (arg == "-r" || arg == "--runs") {
                try {
                    args.runs = std::stoul(argv[++i]);
                    if (args.runs == 0) throw std::out_of_range("runs must be positive");
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value for runs: " << e.what() << "\n\n";
                    printUsage(argv[0]);
                    return args;
                }
            } else if (arg == "-s" || arg == "--rescoring-factor") {
                try {
                    args.rescoring_factor = std::stoul(argv[++i]);
                    if (args.rescoring_factor == 0) throw std::out_of_range("rescoring factor must be positive");
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value for rescoring factor: " << e.what() << "\n\n";
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

        // Check if file exists
        if (!std::filesystem::exists(args.filename)) {
            std::cerr << "Error: File not found: " << args.filename << "\n\n";
            printUsage(argv[0]);
            return args;
        }

        args.valid = true;
        return args;
    }
};

int main(int argc, char* argv[]) {
    auto args = CommandLineArgs::parse(argc, argv);
    if (!args.valid) {
        return 1;
    }

    try {
        BenchmarkConfig config{args.k, args.runs, args.rescoring_factor};
        Searchers searchers;
        
        std::cout << "Initializing searchers with file: " << args.filename << "\n";
        initializeSearchers(searchers, args.filename);
        
        std::cout << "Running benchmark...\n";
        auto results = runBenchmark(searchers, config);
        
        std::cout << "\nBenchmark complete. Results:\n";
        printResults(results, config, searchers);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during execution: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

