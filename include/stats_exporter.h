#include <common_structs.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace simsearch;

// Structure to hold statistics for each search method
template <typename ScoreType> struct SearchMethodStats {
  std::string method_name;
  std::vector<int64_t> times;
  std::vector<double> jaccard_indexes;
  std::vector<double> ndcg_scores;
  std::vector<std::vector<std::pair<ScoreType, size_t>>> results;

  // Calculate aggregate statistics
  json to_json() const {
    double avg_time =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_jaccard =
        std::accumulate(jaccard_indexes.begin(), jaccard_indexes.end(), 0.0) /
        jaccard_indexes.size();
    double avg_ndcg =
        std::accumulate(ndcg_scores.begin(), ndcg_scores.end(), 0.0) /
        ndcg_scores.size();

    json j;
    j["method"] = method_name;
    j["statistics"] = {{"average_time_us", avg_time},
                       {"average_jaccard_index", avg_jaccard},
                       {"average_ndcg", avg_ndcg},
                       {"total_queries", times.size()}};

    // Detailed per-query results
    json queries;
    for (size_t i = 0; i < times.size(); i++) {
      json query;
      query["query_index"] = i;
      query["time_us"] = times[i];
      if (jaccard_indexes.size() > i + 1) {
        query["jaccard_index"] = jaccard_indexes[i];
        query["ndcg_score"] = ndcg_scores[i];
      } else {
        query["jaccard_index"] = 1;
        query["ndcg_score"] = 1;
      }

      // Top results for this query
      json top_results;
      for (const std::pair<ScoreType, size_t> &result : results[i]) {
        top_results.push_back(
            {{"score", result.first}, {"index", result.second}});
      }
      query["results"] = top_results;
      queries.push_back(query);
    }
    j["queries"] = queries;

    return j;
  }
};

// Function to export search results
void exportQuerySearchResults(const std::vector<Query> &queries,
                              const SearchMethodStats<float> &f32_results,
                              const SearchMethodStats<float> &avx2_results,
                              const SearchMethodStats<int> &binary_results,
                              const SearchMethodStats<uint32_t> &uint8_results,
                              const SearchMethodStats<float> &twostep_results,
                              const std::string &output_file,
                              const BenchmarkConfig &config,
                              const Searchers &searchers) {

  json output;
  // Configuration information
  output["configuration"] = {
      {"total_queries", queries.size()},
      {"k", config.k},
      {"rescoring_factor", config.rescoring_factor},
      {"total_embeddings", searchers.base.getEmbeddings().size()},
      {"embedding_dimensions", searchers.base.getEmbeddings()[0].size()}};

  // Query information
  json query_info;
  for (size_t i = 0; i < queries.size(); i++) {
    query_info.push_back({{"original_query", queries[i].query},
                          {"formatted_query", queries[i].formatted_query}});
  }
  output["queries"] = query_info;

  // Results for each method
  output["methods"] = {f32_results.to_json(), avx2_results.to_json(),
                       binary_results.to_json(), uint8_results.to_json(),
                       twostep_results.to_json()};

  // Write to file
  std::ofstream out_file(output_file);
  if (out_file.is_open()) {
    out_file << std::setw(4) << output << std::endl;
  } else {
    std::cerr << "Failed to open output file: " << output_file << std::endl;
  }
}

// Function to export search results
void exportQuerySearchResults(
    const std::vector<SearchMethodStats<float>> floatResults,
    const std::vector<SearchMethodStats<uint32_t>> uint32Results,
    const std::vector<SearchMethodStats<int>> intResults,
    const std::string &output_file, const BenchmarkConfig &config,
    const Searchers &searchers) {

  json output;
  // Configuration information
  output["configuration"] = {
      {"total_queries", config.runs},
      {"k", config.k},
      {"rescoring_factor", config.rescoring_factor},
      {"total_embeddings", searchers.base.getEmbeddings().size()},
      {"embedding_dimensions", searchers.base.getEmbeddings()[0].size()}};

  output["methods"] = {};

  // Results for each method
  for (const SearchMethodStats<float> &method : floatResults) {
    output["methods"].emplace_back(method.to_json());
  }
  for (const SearchMethodStats<int> &method : intResults) {
    output["methods"].emplace_back(method.to_json());
  }
  for (const SearchMethodStats<uint32_t> &method : uint32Results) {
    output["methods"].emplace_back(method.to_json());
  }

  // Write to file
  std::ofstream out_file(output_file);
  if (out_file.is_open()) {
    out_file << std::setw(4) << output << std::endl;
  } else {
    std::cerr << "Failed to open output file: " << output_file << std::endl;
  }
}

// Helper function to create SearchMethodStats
template <typename ScoreType>
SearchMethodStats<ScoreType> createSearchMethodStats(
    const std::string &name, const std::vector<int64_t> &times,
    const std::vector<double> &jaccard_indexes,
    const std::vector<double> &ndcg_scores,
    const std::vector<std::vector<std::pair<ScoreType, size_t>>> &results) {

  return SearchMethodStats{name, times, jaccard_indexes, ndcg_scores, results};
}