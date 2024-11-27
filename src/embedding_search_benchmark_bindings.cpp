// embedding_search_bindings.cpp
#include "common_structs.h"
#include "config_manager.h"
#include "embedding_utils.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyEmbeddingSearch {
private:
  std::unique_ptr<simsearch::Searchers> searchers;
  bool is_initialized = false;
  std::string config_path;

public:
  PyEmbeddingSearch() {
    config_path =
        std::filesystem::path(py::module::import("embedding_search_benchmark")
                                  .attr("__file__")
                                  .cast<std::string>())
            .parent_path()
            .string() +
        "/config.json";
  }

  bool load(const std::string &filename) {
    try {
      ConfigManager::getInstance().initialize(config_path);
      searchers = std::make_unique<simsearch::Searchers>();
      simsearch::initializeSearchers(*searchers, filename);
      is_initialized = true;
      return true;
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Error during initialization: ") +
                               e.what());
    }
  }

  // ======================================
  // search()
  // ======================================

  // Base float search
  py::tuple search_float(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->base.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  // AVX2 optimized search
  py::tuple search_avx2(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->oavx2.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  // Binary AVX2 search
  py::tuple search_binary(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);

    auto start = std::chrono::high_resolution_clock::now();
    auto results =
        searchers->obinary_avx2.similarity_search(queryBinaryAvx2, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  // UINT8 AVX2 search
  py::tuple search_uint8(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    avx2i_vector queryUint8Avx2(query.size() / 8 / 4);
    EmbeddingUtils::convertSingleFloatToUint8AVX2(query, queryUint8Avx2,
                                                  query.size() / 8 / 4);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->ouint8_avx2.similarity_search(queryUint8Avx2, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  // PCA variants
  py::tuple search_pca2(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->pca2.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  py::tuple search_pca4(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->pca4.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  py::tuple search_pca8(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->pca8.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  py::tuple search_pca16(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->pca16.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  py::tuple search_pca32(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = searchers->pca32.similarity_search(query, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(results, time);
  }

  // Two-step search
  py::tuple search_two_step(py::array_t<float> query_vector, size_t k,
                            size_t rescoring_factor) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);

    auto start = std::chrono::high_resolution_clock::now();
    auto binary_results = searchers->obinary_avx2.similarity_search(
        queryBinaryAvx2, k * rescoring_factor);
    auto final_results =
        searchers->oavx2.similarity_search(query, k, binary_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(final_results, time);
  }

  // ======================================
  // getEmbedding()
  // ======================================

  // Get base float embedding
  py::array_t<float> get_float_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->base.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->base.getEmbeddings()[index];
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  // Get optimized AVX2 embedding
  py::array_t<float> get_avx2_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->oavx2.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->oavx2.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  // Get PCA embeddings
  py::array_t<float> get_pca2_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->pca2.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->pca2.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  py::array_t<float> get_pca4_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->pca4.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->pca4.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  py::array_t<float> get_pca8_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->pca8.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->pca8.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  py::array_t<float> get_pca16_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->pca16.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->pca16.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  py::array_t<float> get_pca32_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->pca32.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->pca32.getEmbedding(index);
    auto result = py::array_t<float>(embedding.size());
    auto buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);

    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

  // Get binary AVX2 embedding
  py::array_t<uint64_t> get_binary_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->obinary_avx2.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->obinary_avx2.getEmbeddingAVX2(index);
    size_t size =
        embedding.size() * 4; // Each __m256i contains 4 uint64_t values
    auto result = py::array_t<uint64_t>(size);
    auto buf = result.request();
    uint64_t *ptr = static_cast<uint64_t *>(buf.ptr);

    for (size_t i = 0; i < embedding.size(); i++) {
      uint64_t *val_ptr = reinterpret_cast<uint64_t *>(&embedding[i]);
      std::copy(val_ptr, val_ptr + 4, ptr + i * 4);
    }
    return result;
  }

  // Get uint8 AVX2 embedding
  py::array_t<int8_t> get_uint8_embedding(size_t index) {
    check_initialization();
    if (index >= searchers->ouint8_avx2.getNumVectors()) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto embedding = searchers->ouint8_avx2.getEmbeddingAVX2(index);
    size_t size =
        embedding.size() * 32; // Each __m256i contains 32 int8_t values
    auto result = py::array_t<int8_t>(size);
    auto buf = result.request();
    int8_t *ptr = static_cast<int8_t *>(buf.ptr);

    for (size_t i = 0; i < embedding.size(); i++) {
      int8_t *val_ptr = reinterpret_cast<int8_t *>(&embedding[i]);
      std::copy(val_ptr, val_ptr + 32, ptr + i * 32);
    }
    return result;
  }

  // ======================================

  // Get dimensions info
  py::tuple get_dimensions() const {
    check_initialization();
    return py::make_tuple(searchers->base.getNumVectors(),
                          searchers->base.getVectorDim());
  }

private:
  void check_initialization() const {
    if (!is_initialized) {
      throw std::runtime_error("Searcher not initialized. Call load() first.");
    }
  }

  std::vector<float> convert_query(py::array_t<float> &query_vector) {
    auto buf = query_vector.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("Query vector must be 1-dimensional");
    }

    std::vector<float> query(static_cast<float *>(buf.ptr),
                             static_cast<float *>(buf.ptr) + buf.shape[0]);

    // Normalize the query vector
    float norm = EmbeddingUtils::calcNorm(query);
    for (float &val : query) {
      val = val / norm;
    }

    return query;
  }

  template <typename T>
  py::tuple format_results(const std::vector<std::pair<T, size_t>> &results,
                           int64_t time) {
    py::list results_py;
    const auto &sentences = searchers->base.getSentences();

    for (const auto &result : results) {
      results_py.append(py::make_tuple(
          result.first, result.second,
          EmbeddingUtils::sanitize_utf8(sentences[result.second])));
    }

    return py::make_tuple(results_py, time);
  }
};

PYBIND11_MODULE(embedding_search_benchmark, m) {
  m.doc() = "Python bindings for embedding search implementations";

  py::class_<PyEmbeddingSearch>(m, "EmbeddingSearch")
      .def(py::init<>())
      .def("load", &PyEmbeddingSearch::load, "Load embeddings from file",
           py::arg("filename"))
      .def("search_float", &PyEmbeddingSearch::search_float,
           "Base float search", py::arg("query_vector"), py::arg("k"))
      .def("search_avx2", &PyEmbeddingSearch::search_avx2,
           "AVX2 optimized search", py::arg("query_vector"), py::arg("k"))
      .def("search_binary", &PyEmbeddingSearch::search_binary,
           "Binary AVX2 search", py::arg("query_vector"), py::arg("k"))
      .def("search_uint8", &PyEmbeddingSearch::search_uint8,
           "UINT8 AVX2 search", py::arg("query_vector"), py::arg("k"))
      .def("search_pca2", &PyEmbeddingSearch::search_pca2, "PCA2 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca4", &PyEmbeddingSearch::search_pca4, "PCA4 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca8", &PyEmbeddingSearch::search_pca8, "PCA8 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca16", &PyEmbeddingSearch::search_pca16, "PCA16 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca32", &PyEmbeddingSearch::search_pca32, "PCA32 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_two_step", &PyEmbeddingSearch::search_two_step,
           "Two-step binary+float search", py::arg("query_vector"),
           py::arg("k"), py::arg("rescoring_factor"))
      .def("get_dimensions", &PyEmbeddingSearch::get_dimensions,
           "Get number of vectors and vector dimensions")
      .def("get_float_embedding", &PyEmbeddingSearch::get_float_embedding,
           "Get base float embedding at index", py::arg("index"))
      .def("get_avx2_embedding", &PyEmbeddingSearch::get_avx2_embedding,
           "Get AVX2 optimized embedding at index", py::arg("index"))
      .def("get_binary_embedding", &PyEmbeddingSearch::get_binary_embedding,
           "Get binary AVX2 embedding at index", py::arg("index"))
      .def("get_uint8_embedding", &PyEmbeddingSearch::get_uint8_embedding,
           "Get uint8 AVX2 embedding at index", py::arg("index"))
      .def("get_pca2_embedding", &PyEmbeddingSearch::get_pca2_embedding,
           "Get PCA2 embedding at index", py::arg("index"))
      .def("get_pca4_embedding", &PyEmbeddingSearch::get_pca4_embedding,
           "Get PCA4 embedding at index", py::arg("index"))
      .def("get_pca8_embedding", &PyEmbeddingSearch::get_pca8_embedding,
           "Get PCA8 embedding at index", py::arg("index"))
      .def("get_pca16_embedding", &PyEmbeddingSearch::get_pca16_embedding,
           "Get PCA16 embedding at index", py::arg("index"))
      .def("get_pca32_embedding", &PyEmbeddingSearch::get_pca32_embedding,
           "Get PCA32 embedding at index", py::arg("index"));
}