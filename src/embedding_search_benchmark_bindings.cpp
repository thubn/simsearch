// embedding_search_bindings.cpp
#include "common_structs.h"
#include "config_manager.h"
#include "embedding_utils.h"
#include "timing_breakdown.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations
class PyEmbeddingSearch;

namespace {
template <typename T, typename ResultType = float> struct SearcherInfo {
  T &searcher;
  const char *name;
};
} // namespace

// Template specializations declarations
template <typename T, typename ResultType>
py::tuple perform_search_impl(PyEmbeddingSearch *self,
                              const SearcherInfo<T, ResultType> &info,
                              py::array_t<float> query_vector, size_t k);

template <typename T>
py::tuple perform_search_impl(PyEmbeddingSearch *self,
                              const SearcherInfo<T> &info,
                              py::array_t<float> query_vector, size_t k);

class PyEmbeddingSearch {
private:
  std::unique_ptr<simsearch::Searchers> searchers;
  bool is_initialized = false;
  std::string config_path;

  // Helper methods
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

  template <typename T>
  py::tuple format_results_with_timing(
      const std::vector<std::pair<T, size_t>> &results,
      const TimingBreakdown &timing) {
    py::list results_py;
    const auto &sentences = searchers->base.getSentences();

    for (const auto &result : results) {
      results_py.append(py::make_tuple(
          result.first, result.second,
          EmbeddingUtils::sanitize_utf8(sentences[result.second])));
    }

    py::dict timing_py;
    timing_py["T_query_sketch_ms"] = timing.query_sketch_ms;
    timing_py["T_binary_scan_ms"] = timing.binary_scan_ms;
    timing_py["T_candidate_selection_ms"] = timing.candidate_selection_ms;
    timing_py["T_rescore_ms"] = timing.rescore_ms;
    timing_py["T_final_topk_ms"] = timing.final_topk_ms;
    timing_py["T_total_ms"] = timing.total_ms;
    timing_py["num_survivors"] = timing.num_survivors;

    return py::make_tuple(results_py, timing.total_ms * 1000.0, timing_py);
  }

  // Template method that forwards to implementation
  template <typename T, typename ResultType = float>
  py::tuple perform_search(const SearcherInfo<T, ResultType> &info,
                           py::array_t<float> query_vector, size_t k) {
    return perform_search_impl(this, info, query_vector, k);
  }

  // Friend declarations for implementations
  template <typename T, typename R>
  friend py::tuple perform_search_impl(PyEmbeddingSearch *,
                                       const SearcherInfo<T, R> &,
                                       py::array_t<float>, size_t);

  template <typename T>
  friend py::tuple perform_search_impl(PyEmbeddingSearch *,
                                       const SearcherInfo<T> &,
                                       py::array_t<float>, size_t);

  template <typename T>
  py::array_t<T> get_embedding_impl(size_t index, size_t num_vectors,
                                    const std::vector<T> &embedding) {
    check_initialization();
    if (index >= num_vectors) {
      throw std::out_of_range("Embedding index out of range");
    }

    auto result = py::array_t<T>(embedding.size());
    auto buf = result.request();
    T *ptr = static_cast<T *>(buf.ptr);
    std::copy(embedding.begin(), embedding.end(), ptr);
    return result;
  }

public:
  PyEmbeddingSearch() {
    const auto module_dir =
        std::filesystem::path(py::module::import("embedding_search_benchmark")
                                  .attr("__file__")
                                  .cast<std::string>())
            .parent_path();

    // Prefer a config next to the compiled module; fall back to repo roots that
    // are common when running from the python/benchmark folder.
    const std::vector<std::filesystem::path> candidates = {
        module_dir / "config.json",
        module_dir.parent_path() / "config.json",           // repo root
        std::filesystem::current_path() / "config.json"};   // CWD

    for (const auto &candidate : candidates) {
      if (std::filesystem::exists(candidate)) {
        config_path = candidate.string();
        break;
      }
    }

    if (config_path.empty()) {
      // Keep the original default to surface a clear error downstream.
      config_path = (module_dir / "config.json").string();
    }
  }

  bool load(const std::string &filename, const int embedding_dim,
            bool init_pca = true, bool init_avx2 = true,
            bool init_binary = true, bool init_int8 = true,
            bool init_float16 = true, bool init_mf = true,
            size_t max_vectors = 0) {
    try {
      ConfigManager::getInstance().initialize(config_path);
      searchers = std::make_unique<simsearch::Searchers>();
      simsearch::initializeSearchers(*searchers, filename, embedding_dim,
                                     init_pca, init_avx2, init_binary,
                                     init_int8, init_float16, init_mf,
                                     max_vectors);
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

  // Search methods
  py::tuple search_float(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<EmbeddingSearchFloat>{searchers->base, "float"},
        query_vector, k);
  }

  py::tuple search_avx2(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->oavx2, "avx2"},
        query_vector, k);
  }

  py::tuple search_binary(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchBinaryAVX2, int32_t>{
            searchers->obinary_avx2, "binary"},
        query_vector, k);
  }

  py::tuple search_binary_timed(py::array_t<float> query_vector, size_t k) {
    check_initialization();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<float> query = convert_query(query_vector);

    TimingBreakdown timing;
    auto sketch_start = std::chrono::high_resolution_clock::now();
    avx2i_vector queryBinary(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinary,
                                                   query.size() / 8 / 32);
    auto sketch_end = std::chrono::high_resolution_clock::now();

    auto results =
        searchers->obinary_avx2.similarity_search_with_timing(queryBinary, k,
                                                              timing);
    auto total_end = std::chrono::high_resolution_clock::now();

    timing.query_sketch_ms =
        std::chrono::duration<double, std::milli>(sketch_end - sketch_start)
            .count();
    timing.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();

    return format_results_with_timing(results, timing);
  }

  py::tuple search_int8(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchUint8AVX2, int32_t>{
            searchers->ouint8_avx2, "int8"},
        query_vector, k);
  }

  py::tuple search_float16(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<EmbeddingSearchFloat16, float>{searchers->float16,
                                                    "float16"},
        query_vector, k);
  }

  py::tuple search_mf(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<EmbeddingSearchMappedFloat>{searchers->mappedFloat,
                                                 "mapped float"},
        query_vector, k);
  }

  py::tuple search_pca2(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->pca2, "pca2"},
        query_vector, k);
  }

  py::tuple search_pca4(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->pca4, "pca4"},
        query_vector, k);
  }

  py::tuple search_pca8(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->pca8, "pca8"},
        query_vector, k);
  }

  py::tuple search_pca16(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->pca16, "pca16"},
        query_vector, k);
  }

  py::tuple search_pca32(py::array_t<float> query_vector, size_t k) {
    return perform_search(
        SearcherInfo<OptimizedEmbeddingSearchAVX2>{searchers->pca32, "pca32"},
        query_vector, k);
  }

  py::tuple search_twostep(py::array_t<float> query_vector, size_t k,
                           size_t rescoring_factor) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    // Convert query for binary search
    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);

    // Perform two-step search with timing
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

  py::tuple search_twostep_timed(py::array_t<float> query_vector, size_t k,
                                 size_t rescoring_factor) {
    check_initialization();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<float> query = convert_query(query_vector);

    TimingBreakdown timing;
    auto sketch_start = std::chrono::high_resolution_clock::now();
    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);
    auto sketch_end = std::chrono::high_resolution_clock::now();

    auto binary_results = searchers->obinary_avx2.similarity_search_with_timing(
        queryBinaryAvx2, k * rescoring_factor, timing);
    TimingBreakdown rescore_timing;
    auto final_results = searchers->oavx2.similarity_search_with_timing(
        query, k, binary_results, rescore_timing);
    auto total_end = std::chrono::high_resolution_clock::now();

    timing.query_sketch_ms =
        std::chrono::duration<double, std::milli>(sketch_end - sketch_start)
            .count();
    timing.rescore_ms = rescore_timing.rescore_ms;
    timing.final_topk_ms = rescore_timing.final_topk_ms;
    timing.num_survivors = binary_results.size();
    timing.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();

    return format_results_with_timing(final_results, timing);
  }

  py::tuple search_twostep_mf(py::array_t<float> query_vector, size_t k,
                              size_t rescoring_factor) {
    check_initialization();
    std::vector<float> query = convert_query(query_vector);

    // Convert query for binary search
    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);

    // Perform two-step search with timing
    auto start = std::chrono::high_resolution_clock::now();
    auto binary_results = searchers->obinary_avx2.similarity_search(
        queryBinaryAvx2, k * rescoring_factor);
    auto final_results =
        searchers->mappedFloat.similarity_search(query, k, binary_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return format_results(final_results, time);
  }

  py::tuple search_twostep_mf_timed(py::array_t<float> query_vector, size_t k,
                                    size_t rescoring_factor) {
    check_initialization();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<float> query = convert_query(query_vector);

    TimingBreakdown timing;
    auto sketch_start = std::chrono::high_resolution_clock::now();
    avx2i_vector queryBinaryAvx2(query.size() / 8 / 32);
    EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinaryAvx2,
                                                   query.size() / 8 / 32);
    auto sketch_end = std::chrono::high_resolution_clock::now();

    auto binary_results = searchers->obinary_avx2.similarity_search_with_timing(
        queryBinaryAvx2, k * rescoring_factor, timing);
    TimingBreakdown rescore_timing;
    auto final_results = searchers->mappedFloat.similarity_search_with_timing(
        query, k, binary_results, rescore_timing);
    auto total_end = std::chrono::high_resolution_clock::now();

    timing.query_sketch_ms =
        std::chrono::duration<double, std::milli>(sketch_end - sketch_start)
            .count();
    timing.rescore_ms = rescore_timing.rescore_ms;
    timing.final_topk_ms = rescore_timing.final_topk_ms;
    timing.num_survivors = binary_results.size();
    timing.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();

    return format_results_with_timing(final_results, timing);
  }

  // ======================================
  // getEmbedding()
  // ======================================

  py::array_t<float> get_float_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->base.getNumVectors(),
                                     searchers->base.getEmbeddings()[index]);
  }

  py::array_t<float> get_avx2_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->oavx2.getNumVectors(),
                                     searchers->oavx2.getEmbedding(index));
  }

  py::array_t<float> get_pca2_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->pca2.getNumVectors(),
                                     searchers->pca2.getEmbedding(index));
  }

  py::array_t<float> get_pca4_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->pca4.getNumVectors(),
                                     searchers->pca4.getEmbedding(index));
  }

  py::array_t<float> get_pca8_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->pca8.getNumVectors(),
                                     searchers->pca8.getEmbedding(index));
  }

  py::array_t<float> get_pca16_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->pca16.getNumVectors(),
                                     searchers->pca16.getEmbedding(index));
  }

  py::array_t<float> get_pca32_embedding(size_t index) {
    return get_embedding_impl<float>(index, searchers->pca32.getNumVectors(),
                                     searchers->pca32.getEmbedding(index));
  }

  // ======================================

  // Get dimensions info
  py::tuple get_dimensions() const {
    check_initialization();
    return py::make_tuple(searchers->base.getNumVectors(),
                          searchers->base.getVectorDim());
  }

  py::str get_sentence(size_t index) const {
    check_initialization();
    return searchers->base.getSentences()[index];
  }
};

// Implementation for float-based searchers (base and AVX2)
template <typename T>
py::tuple perform_search_impl(PyEmbeddingSearch *self,
                              const SearcherInfo<T> &info,
                              py::array_t<float> query_vector, size_t k) {
  self->check_initialization();
  std::vector<float> query = self->convert_query(query_vector);

  auto start = std::chrono::high_resolution_clock::now();
  auto results = info.searcher.similarity_search(query, k);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

  return self->format_results(results, time);
}

// Specialization for binary AVX2 searcher
template <>
py::tuple perform_search_impl<OptimizedEmbeddingSearchBinaryAVX2, int32_t>(
    PyEmbeddingSearch *self,
    const SearcherInfo<OptimizedEmbeddingSearchBinaryAVX2, int32_t> &info,
    py::array_t<float> query_vector, size_t k) {
  self->check_initialization();
  std::vector<float> query = self->convert_query(query_vector);

  avx2i_vector queryBinary(query.size() / 8 / 32);
  EmbeddingUtils::convertSingleFloatToBinaryAVX2(query, queryBinary,
                                                 query.size() / 8 / 32);

  auto start = std::chrono::high_resolution_clock::now();
  auto results = info.searcher.similarity_search(queryBinary, k);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

  return self->format_results(results, time);
}

// Specialization for int8 AVX2 searcher
template <>
py::tuple perform_search_impl<OptimizedEmbeddingSearchUint8AVX2, int32_t>(
    PyEmbeddingSearch *self,
    const SearcherInfo<OptimizedEmbeddingSearchUint8AVX2, int32_t> &info,
    py::array_t<float> query_vector, size_t k) {
  self->check_initialization();
  std::vector<float> query = self->convert_query(query_vector);

  avx2i_vector queryInt8(query.size() / 8 / 4);
  EmbeddingUtils::convertSingleFloatToUint8AVX2(query, queryInt8,
                                                query.size() / 8 / 4);

  auto start = std::chrono::high_resolution_clock::now();
  auto results = info.searcher.similarity_search(queryInt8, k);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

  return self->format_results(results, time);
}

// Specialization for int8 AVX2 searcher
/*template <>
py::tuple perform_search_impl<EmbeddingSearchUint8AVX2, uint32_t>(
    PyEmbeddingSearch *self,
    const SearcherInfo<EmbeddingSearchUint8AVX2, uint32_t> &info,
    py::array_t<float> query_vector, size_t k) {
  self->check_initialization();
  std::vector<float> query = self->convert_query(query_vector);

  avx2i_vector queryInt8(query.size() / 8 / 4);
  EmbeddingUtils::convertSingleFloatToUint8AVX2(query, queryInt8,
                                                query.size() / 8 / 4);

  auto start = std::chrono::high_resolution_clock::now();
  auto results = info.searcher.similarity_search(queryInt8, k);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

  return self->format_results(results, time);
}*/

// Module definition using a helper macro to reduce repetition
#define ADD_SEARCH_METHOD(name, func)                                          \
  .def(#name, &PyEmbeddingSearch::func, #name " search",                       \
       py::arg("query_vector"), py::arg("k"))

#define ADD_EMBEDDING_METHOD(name, func)                                       \
  .def(#name, &PyEmbeddingSearch::func, "Get " #name " embedding at index",    \
       py::arg("index"))

PYBIND11_MODULE(embedding_search_benchmark, m) {
  m.doc() = "Python bindings for embedding search implementations";

  py::class_<PyEmbeddingSearch>(m, "EmbeddingSearch")
      .def(py::init())
      .def("load", &PyEmbeddingSearch::load, "Load embeddings from file",
           py::arg("filename"), py::arg("embedding_dim"), py::arg("init_pca"),
           py::arg("init_avx2"), py::arg("init_binary"), py::arg("init_int8"),
           py::arg("init_float16"), py::arg("init_mf"),
           py::arg("max_vectors") = 0)
      .def("search_float", &PyEmbeddingSearch::search_float,
           "Base float search", py::arg("query_vector"), py::arg("k"))
      .def("search_avx2", &PyEmbeddingSearch::search_avx2,
           "AVX2 optimized search", py::arg("query_vector"), py::arg("k"))
      .def("search_binary", &PyEmbeddingSearch::search_binary,
           "Binary AVX2 search", py::arg("query_vector"), py::arg("k"))
      .def("search_binary_timed", &PyEmbeddingSearch::search_binary_timed,
           "Binary AVX2 search with timing breakdown",
           py::arg("query_vector"), py::arg("k"))
      .def("search_int8", &PyEmbeddingSearch::search_int8, "INT8 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_float16", &PyEmbeddingSearch::search_float16,
           "float16 search", py::arg("query_vector"), py::arg("k"))
      .def("search_mf", &PyEmbeddingSearch::search_mf, "mapped float search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca2", &PyEmbeddingSearch::search_pca2, "pca2 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca4", &PyEmbeddingSearch::search_pca4, "pca4 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca8", &PyEmbeddingSearch::search_pca8, "pca8 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca16", &PyEmbeddingSearch::search_pca16, "pca16 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_pca32", &PyEmbeddingSearch::search_pca32, "pca32 search",
           py::arg("query_vector"), py::arg("k"))
      .def("search_twostep", &PyEmbeddingSearch::search_twostep,
           "Two-step binary+float search", py::arg("query_vector"),
           py::arg("k"), py::arg("rescoring_factor") = 50)
      .def("search_twostep_timed", &PyEmbeddingSearch::search_twostep_timed,
           "Two-step binary+float search with timing breakdown",
           py::arg("query_vector"), py::arg("k"),
           py::arg("rescoring_factor") = 50)
      .def("search_twostep_mf", &PyEmbeddingSearch::search_twostep_mf,
           "Two-step binary+mf search", py::arg("query_vector"), py::arg("k"),
           py::arg("rescoring_factor") = 50)
      .def("search_twostep_mf_timed",
           &PyEmbeddingSearch::search_twostep_mf_timed,
           "Two-step binary+mf search with timing breakdown",
           py::arg("query_vector"), py::arg("k"),
           py::arg("rescoring_factor") = 50)
      .def("get_float_embedding", &PyEmbeddingSearch::get_float_embedding,
           "Get float embedding at index", py::arg("index"))
      .def("get_avx2_embedding", &PyEmbeddingSearch::get_avx2_embedding,
           "Get AVX2 embedding at index", py::arg("index"))
      .def("get_pca2_embedding", &PyEmbeddingSearch::get_pca2_embedding,
           "Get PCA2 embedding at index", py::arg("index"))
      .def("get_pca4_embedding", &PyEmbeddingSearch::get_pca4_embedding,
           "Get PCA4 embedding at index", py::arg("index"))
      .def("get_pca8_embedding", &PyEmbeddingSearch::get_pca8_embedding,
           "Get PCA8 embedding at index", py::arg("index"))
      .def("get_pca16_embedding", &PyEmbeddingSearch::get_pca16_embedding,
           "Get PCA16 embedding at index", py::arg("index"))
      .def("get_pca32_embedding", &PyEmbeddingSearch::get_pca32_embedding,
           "Get PCA32 embedding at index", py::arg("index"))
      .def("get_dimensions", &PyEmbeddingSearch::get_dimensions,
           "Get number of vectors and vector dimensions")
      .def("get_sentence", &PyEmbeddingSearch::get_sentence, py::arg("index"));
}
