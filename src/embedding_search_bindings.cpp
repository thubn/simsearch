// embedding_search_bindings.cpp
#include "config_manager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "embedding_search_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_uint8_avx2.h"
#include "optimized_embedding_search_avx2.h"
#include "optimized_embedding_search_uint8_avx2.h"
#include "optimized_embedding_search_binary_avx2.h"
#include "embedding_utils.h"
#include <chrono>
#include <immintrin.h>
#include <iostream>

namespace py = pybind11;

class PyEmbeddingSearch
{
private:
    // EmbeddingSearchFloat float_searcher;
    std::unique_ptr<OptimizedEmbeddingSearchAVX2> avx2_searcher;
    std::unique_ptr<OptimizedEmbeddingSearchBinaryAVX2> binary_avx2_searcher;
    std::unique_ptr<OptimizedEmbeddingSearchUint8AVX2> uint8_avx2_searcher;
    bool is_initialized = false;
    std::string config_path;

public:
    PyEmbeddingSearch()
    {
        // Just store the config path in constructor, don't initialize anything yet
        config_path = std::filesystem::path(py::module::import("embedding_search").attr("__file__").cast<std::string>())
                          .parent_path()
                          .string() +
                      "/config.json";
        // ensure_config_exists(config_path);
    }

    bool load(const std::string &filename)
    {
        try
        {
            // Initialize config manager
            ConfigManager::getInstance().initialize(config_path);

            avx2_searcher = std::make_unique<OptimizedEmbeddingSearchAVX2>();
            binary_avx2_searcher = std::make_unique<OptimizedEmbeddingSearchBinaryAVX2>();
            uint8_avx2_searcher = std::make_unique<OptimizedEmbeddingSearchUint8AVX2>();

            EmbeddingSearchFloat float_searcher;
            bool success = float_searcher.load(filename);
            if (success)
            {
                avx2_searcher->setEmbeddings(float_searcher.getEmbeddings());
                avx2_searcher->setSentences(float_searcher.getSentences());
                binary_avx2_searcher->setEmbeddings(float_searcher.getEmbeddings());
                uint8_avx2_searcher->setEmbeddings(float_searcher.getEmbeddings());
                is_initialized = true;
                float_searcher.unsetEmbeddings();
                float_searcher.unsetSentences();
            }
            return success;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(std::string("Error during initialization: ") + e.what());
        }
    }

    py::tuple benchmark_search(py::array_t<float> query_vectors, size_t k, size_t num_runs)
    {
        if (!is_initialized)
        {
            throw std::runtime_error("Searcher not initialized. Call load() first.");
        }

        // Validate input
        if (query_vectors.ndim() != 2)
        {
            throw std::runtime_error("Query vectors must be a 2D numpy array");
        }

        auto buf = query_vectors.request();
        float *ptr = static_cast<float *>(buf.ptr);
        size_t num_queries = buf.shape[0];
        size_t dim = buf.shape[1];

        // Initialize result vectors
        std::vector<double> float_times, avx2_times, binary_times, uint8_times;
        std::vector<double> avx2_accuracies, binary_accuracies, uint8_accuracies;

        // Run benchmarks for each query
        for (size_t q = 0; q < num_queries; q++)
        {
            std::vector<float> query(ptr + q * dim, ptr + (q + 1) * dim);
            avx2_vector queryAvx2(query.size() / 8);
            EmbeddingUtils::convertSingleEmbeddingToAVX2(query, queryAvx2, query.size() / 8);
            avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
            EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2, query.size() / 8 / 32);

            // Float search (baseline)
            {
                /*
                auto start = std::chrono::high_resolution_clock::now();
                auto float_results = float_searcher.similarity_search(query, k);
                auto end = std::chrono::high_resolution_clock::now();
                float_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                */

                // AVX2 search
                auto start = std::chrono::high_resolution_clock::now();
                auto avx2_results = avx2_searcher->similarity_search(queryAvx2, k);
                auto end = std::chrono::high_resolution_clock::now();
                avx2_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                // avx2_accuracies.push_back(EmbeddingUtils::calculateJaccardIndex(float_results, avx2_results));

                // Binary AVX2 search
                start = std::chrono::high_resolution_clock::now();
                auto binary_results = binary_avx2_searcher->similarity_search(queryBinaryAvx2, k);
                end = std::chrono::high_resolution_clock::now();
                binary_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                binary_accuracies.push_back(EmbeddingUtils::calculateJaccardIndex(avx2_results, binary_results));

                /*
                // UINT8 AVX2 search
                start = std::chrono::high_resolution_clock::now();
                auto uint8_results = uint8_avx2_searcher.similarity_search(uint8_avx2_searcher.floatToAvx2(query), k);
                end = std::chrono::high_resolution_clock::now();
                uint8_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                uint8_accuracies.push_back(EmbeddingUtils::calculateJaccardIndex(float_results, uint8_results));
                */
            }
        }

        // Return results as a tuple
        return py::make_tuple(
            float_times, avx2_times, binary_times, uint8_times,
            avx2_accuracies, binary_accuracies, uint8_accuracies);
    }

    py::tuple search(py::array_t<float> query_vector, size_t k, size_t rescoring_factor)
    {
        if (!is_initialized)
        {
            throw std::runtime_error("Searcher not initialized. Call load() first.");
        }

        // Convert numpy array to vector
        auto buf = query_vector.request();
        if (buf.ndim != 1)
        {
            throw std::runtime_error("Query vector must be 1-dimensional");
        }
        std::vector<float> query(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.shape[0]);
        float norm = EmbeddingUtils::calcNorm(query);
        for (float &val : query)
        {
            val = val / norm;
        }

        try
        {
            // Convert query for different searchers
            avx2_vector queryAvx2(query.size() / 8);
            EmbeddingUtils::convertSingleEmbeddingToAVX2(query, queryAvx2, query.size() / 8);

            avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
            EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2, query.size() / 8 / 32);

            avx2i_vector queryUint8Avx2(query.size() / 8 / 4);
            EmbeddingUtils::convertSingleFloatToUint8AVX2(query, queryUint8Avx2, query.size() / 8 / 4);

            // Timing variables
            int64_t avx2_time, binary_time, uint8_time, twostep_time;

            // Perform AVX2 search with timing
            auto avx2_start = std::chrono::high_resolution_clock::now();
            auto avx2_results = avx2_searcher->similarity_search(query, k);
            auto avx2_end = std::chrono::high_resolution_clock::now();
            avx2_time = std::chrono::duration_cast<std::chrono::microseconds>(avx2_end - avx2_start).count();

            // Perform Binary AVX2 search with timing
            auto binary_start = std::chrono::high_resolution_clock::now();
            auto binary_results = binary_avx2_searcher->similarity_search(queryBinaryAvx2, k);
            auto binary_end = std::chrono::high_resolution_clock::now();
            binary_time = std::chrono::duration_cast<std::chrono::microseconds>(binary_end - binary_start).count();

            // Perform UINT8 AVX2 search with timing
            auto uint8_start = std::chrono::high_resolution_clock::now();
            auto uint8_results = uint8_avx2_searcher->similarity_search(queryUint8Avx2, k);
            auto uint8_end = std::chrono::high_resolution_clock::now();
            uint8_time = std::chrono::duration_cast<std::chrono::microseconds>(uint8_end - uint8_start).count();

            // Perform two step Binary AVX2 search with timing
            auto twostep_start = std::chrono::high_resolution_clock::now();
            auto twostep_binary_results = binary_avx2_searcher->similarity_search(queryBinaryAvx2, k * rescoring_factor);
            auto twostep_avx2_results = avx2_searcher->similarity_search(query, k, twostep_binary_results);
            auto twostep_end = std::chrono::high_resolution_clock::now();
            twostep_time = std::chrono::duration_cast<std::chrono::microseconds>(twostep_end - twostep_start).count();

            // Calculate comparison metrics
            double binary_jaccard = EmbeddingUtils::calculateJaccardIndex(avx2_results, binary_results);
            double uint8_jaccard = EmbeddingUtils::calculateJaccardIndex(avx2_results, uint8_results);
            double twostep_jaccard = EmbeddingUtils::calculateJaccardIndex(avx2_results, twostep_avx2_results);

            double binary_ndcg = EmbeddingUtils::calculateNDCG(avx2_results, binary_results);
            double uint8_ndcg = EmbeddingUtils::calculateNDCG(avx2_results, uint8_results);
            double twostep_ndcg = EmbeddingUtils::calculateNDCG(avx2_results, twostep_avx2_results);

            // Get sentences
            const auto &sentences = avx2_searcher->getSentences();

            // Convert results to Python lists
            py::list avx2_results_py, binary_results_py, uint8_results_py, twostep_results_py;

            for (const auto &result : avx2_results)
            {
                avx2_results_py.append(py::make_tuple(result.first, result.second, EmbeddingUtils::sanitize_utf8(sentences[result.second])));
            }
            for (const auto &result : binary_results)
            {
                binary_results_py.append(py::make_tuple(result.first, result.second, EmbeddingUtils::sanitize_utf8(sentences[result.second])));
            }
            for (const auto &result : uint8_results)
            {
                uint8_results_py.append(py::make_tuple(result.first, result.second, EmbeddingUtils::sanitize_utf8(sentences[result.second])));
            }
            for (const auto &result : twostep_avx2_results)
            {
                twostep_results_py.append(py::make_tuple(result.first, result.second, EmbeddingUtils::sanitize_utf8(sentences[result.second])));
            }

            return py::make_tuple(
                avx2_results_py, binary_results_py, uint8_results_py, twostep_results_py,
                avx2_time, binary_time, uint8_time, twostep_time,
                binary_jaccard, uint8_jaccard, twostep_jaccard,
                binary_ndcg, uint8_ndcg, twostep_ndcg);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(std::string("Search failed: ") + e.what());
        }
    }

private:
};

PYBIND11_MODULE(embedding_search, m)
{
    m.doc() = "Python bindings for embedding search implementations";

    py::class_<PyEmbeddingSearch>(m, "EmbeddingSearch")
        .def(py::init<>())
        .def("load", &PyEmbeddingSearch::load, "Load embeddings from file")
        .def("search", &PyEmbeddingSearch::search, "Perform similarity search with all methods")
        .def("benchmark_search", &PyEmbeddingSearch::benchmark_search, "Benchmark similarity search with multiple queries");
}