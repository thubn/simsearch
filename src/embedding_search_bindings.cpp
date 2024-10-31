// embedding_search_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "embedding_search_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_uint8_avx2.h"
#include "embedding_utils.h"
#include <chrono>
#include <immintrin.h>

namespace py = pybind11;

class PyEmbeddingSearch
{
private:
    // EmbeddingSearchFloat float_searcher;
    EmbeddingSearchAVX2 avx2_searcher;
    EmbeddingSearchBinaryAVX2 binary_avx2_searcher;
    EmbeddingSearchUint8AVX2 uint8_avx2_searcher;
    bool is_initialized = false;

public:
    bool load(const std::string &filename)
    {
        EmbeddingSearchFloat float_searcher;
        bool success = float_searcher.load(filename);
        if (success)
        {
            avx2_searcher.setEmbeddings(float_searcher.getEmbeddings());
            avx2_searcher.setSentences(float_searcher.getSentences());
            binary_avx2_searcher.create_binary_embedding_from_float(float_searcher.getEmbeddings());
            // uint8_avx2_searcher.setEmbeddings(float_searcher.getEmbeddings());

            is_initialized = true;
        }
        return success;
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
            std::vector<__m256> queryAvx2(query.size() / 8);
            EmbeddingUtils::convertSingleEmbeddingToAVX2(query, queryAvx2, query.size() / 8);
            std::vector<__m256i> queryBinaryAvx2(query.size() / 8 / 32);
            EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2, query.size(), query.size() / 8 / 32);

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
                auto avx2_results = avx2_searcher.similarity_search(queryAvx2, k);
                auto end = std::chrono::high_resolution_clock::now();
                avx2_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                // avx2_accuracies.push_back(EmbeddingUtils::calculateJaccardIndex(float_results, avx2_results));

                // Binary AVX2 search
                start = std::chrono::high_resolution_clock::now();
                auto binary_results = binary_avx2_searcher.similarity_search(queryBinaryAvx2, k);
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

    py::tuple search(py::array_t<float> query_vector, size_t k)
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
        std::vector<__m256> queryAvx2(query.size() / 8);
        EmbeddingUtils::convertSingleEmbeddingToAVX2(query, queryAvx2, query.size() / 8);
        std::vector<__m256i> queryBinaryAvx2(query.size() / 8 / 32);
        EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2, query.size(), query.size() / 8 / 32);

        // Perform searches with all methods
        // auto float_results = float_searcher.similarity_search(query, k);
        auto avx2_results = avx2_searcher.similarity_search(queryAvx2, k);
        auto binary_results = binary_avx2_searcher.similarity_search(queryBinaryAvx2, k);
        // auto uint8_results = uint8_avx2_searcher.similarity_search(uint8_avx2_searcher.floatToAvx2(query), k);

        // Get sentences for results
        const auto &sentences = avx2_searcher.getSentences();

        // Convert results to Python lists
        py::list float_results_py, avx2_results_py, binary_results_py, uint8_results_py;

        /*for (const auto &result : float_results)
        {
            float_results_py.append(py::make_tuple(result.first, result.second, sentences[result.second]));
        }*/
        for (const auto &result : avx2_results)
        {
            avx2_results_py.append(py::make_tuple(result.first, result.second, sentences[result.second]));
        }
        for (const auto &result : binary_results)
        {
            binary_results_py.append(py::make_tuple(result.first, result.second, sentences[result.second]));
        }
        /*
        for (const auto& result : uint8_results) {
            uint8_results_py.append(py::make_tuple(result.first, result.second, sentences[result.second]));
        }
        */

        return py::make_tuple(float_results_py, avx2_results_py, binary_results_py, uint8_results_py);
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