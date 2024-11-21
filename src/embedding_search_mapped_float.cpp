#include "embedding_search_mapped_float.h"
#include "embedding_io.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <stdexcept>

struct PartitionInfo {
  float start;
  float end;
  float average;
};

// Distribution function type for easier switching
enum class DistributionType {
  UNIFORM,
  LINEAR,
  QUADRATIC,
  CUBIC,
  GAUSSIAN,
  COSINE
};

// Calculate weight based on relative position and distribution type
double calculateWeight(double relative_pos, DistributionType dist_type,
                       double factor = 10.0) {
  switch (dist_type) {
  case DistributionType::UNIFORM:
    return 1.0;
  case DistributionType::LINEAR:
    return 1.0 - (std::abs(relative_pos) * 0.5);
  case DistributionType::QUADRATIC:
    return 1.0 - (relative_pos * relative_pos * 0.75);
  case DistributionType::CUBIC:
    return 1.0 - (relative_pos * relative_pos * relative_pos * 0.5);
  case DistributionType::GAUSSIAN:
    return std::exp(-relative_pos * relative_pos * factor);
  case DistributionType::COSINE:
    return std::cos(relative_pos * M_PI_2) * 0.5 + 0.5;
  default:
    return 1.0;
  }
}

std::vector<PartitionInfo>
partitionAndAverage(std::vector<float> &arr, int n_parts,
                    DistributionType dist_type = DistributionType::QUADRATIC,
                    double factor = 10.0) {
  try {
    if (arr.empty()) {
      throw std::invalid_argument("Input array is empty");
    }
    if (n_parts <= 0 || n_parts > arr.size()) {
      throw std::invalid_argument("Invalid number of parts");
    }

    // Sort the array
    std::sort(arr.begin(), arr.end());

    // Find indices for zero crossing
    auto zero_it = std::lower_bound(arr.begin(), arr.end(), 0.0f);
    size_t zero_index = std::distance(arr.begin(), zero_it);

    // Calculate number of negative and positive values
    size_t neg_count = zero_index;
    size_t pos_count = arr.size() - zero_index;

    // Determine partition split based on data distribution
    int neg_parts = static_cast<int>(
        (static_cast<double>(neg_count) / arr.size()) * n_parts);
    int pos_parts = n_parts - neg_parts;

    // Ensure at least one partition for each side if there's data
    if (neg_count > 0 && neg_parts == 0)
      neg_parts = 1;
    if (pos_count > 0 && pos_parts == 0)
      pos_parts = 1;
    pos_parts = n_parts - neg_parts;

    /*
    std::cout << "Distribution info:\n";
    std::cout << "Distribution type: " << static_cast<int>(dist_type) << "\n";
    std::cout << "Total elements: " << arr.size() << "\n";
    std::cout << "Negative elements: " << neg_count << " (" << neg_parts
              << " partitions)\n";
    std::cout << "Positive elements: " << pos_count << " (" << pos_parts
              << " partitions)\n\n";
    */

    // Calculate weights and sizes for both sides
    std::vector<size_t> partition_sizes(n_parts);

    // Handle negative partitions
    if (neg_parts > 0) {
      std::vector<double> weights(neg_parts);
      double total_weight = 0.0;

      // Calculate weights
      for (int i = 0; i < neg_parts; i++) {
        double relative_pos = (double(i) / double(neg_parts - 1)) - 1.0;
        // std::cout << "neg relative_pos: " << i << " " << relative_pos <<
        // std::endl;
        weights[i] = calculateWeight(relative_pos, dist_type, factor);
        total_weight += weights[i];
      }

      // Convert weights to sizes
      size_t elements_assigned = 0;
      for (int i = 0; i < neg_parts - 1; i++) {
        double proportion = weights[i] / total_weight;
        partition_sizes[i] =
            std::max(size_t(1), static_cast<size_t>(neg_count * proportion));
        elements_assigned += partition_sizes[i];
      }
      // Last negative partition gets remaining elements
      if (elements_assigned < neg_count) {
        partition_sizes[neg_parts - 1] = neg_count - elements_assigned;
      }
    }

    // Handle positive partitions
    if (pos_parts > 0) {
      std::vector<double> weights(pos_parts);
      double total_weight = 0.0;

      // Calculate weights
      for (int i = 0; i < pos_parts; i++) {
        double relative_pos = double(i) / double(pos_parts - 1);
        // std::cout << "pos relative_pos: " << i << " " << relative_pos <<
        // std::endl;
        weights[i] = calculateWeight(relative_pos, dist_type, factor);
        total_weight += weights[i];
      }

      // Convert weights to sizes
      size_t elements_assigned = 0;
      for (int i = 0; i < pos_parts - 1; i++) {
        double proportion = weights[i] / total_weight;
        partition_sizes[i + neg_parts] =
            std::max(size_t(1), static_cast<size_t>(pos_count * proportion));
        elements_assigned += partition_sizes[i + neg_parts];
      }
      // Last positive partition gets remaining elements
      if (elements_assigned < pos_count) {
        partition_sizes[n_parts - 1] = pos_count - elements_assigned;
      }
    }

    // Create and fill partitions
    std::vector<PartitionInfo> partitions(n_parts);
    size_t start_pos = 0;

    for (int i = 0; i < n_parts; ++i) {
      if (partition_sizes[i] == 0) {
        continue;
      }

      double sum = 0.0f;
      for (size_t j = 0; j < partition_sizes[i]; ++j) {
        sum += double(arr[start_pos + j]);
      }

      float average = partition_sizes[i] > 0 ? sum / partition_sizes[i] : 0.0f;

      partitions[i] = {arr[start_pos], arr[start_pos + partition_sizes[i] - 1],
                       average};

      /*
      // Print partition details
      std::cout << "Partition " << std::setw(2) << i << ": "
                << "size = " << std::setw(10) << partition_sizes[i]
                << " elements, range [" << std::fixed << std::setprecision(6)
                << partitions[i].start << ", " << partitions[i].end
                << "], avg = " << partitions[i].average << "\n";
      */

      start_pos += partition_sizes[i];
    }

    return partitions;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    throw;
  }
}

std::vector<float> flattenMatrix(const std::vector<std::vector<float>> &input) {
  std::vector<float> flattened;
  // size_t total_size = input.size() * input[0].size();
  try {
    size_t total_size = 0;
    for (const auto &row : input) {
      total_size += row.size();
    }
    flattened.reserve(total_size);
    for (const auto &row : input) {
      flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed in flattenMatrix: " << e.what()
              << '\n';
    std::cerr << "Available system memory might be insufficient\n";
    throw; // Re-throw the exception
  }
}

std::string floatToHex(float value) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = value;
  std::stringstream ss;
  ss << "0x" << std::hex << std::uppercase << u.i;
  return ss.str();
}

void printPartitions(const std::vector<PartitionInfo> &partitions) {
  // Print human-readable results
  std::cout << std::fixed
            << std::setprecision(6); // Increased from 2 to 6 decimal places
  std::cout << "Partition information:\n";
  for (int i = 0; i < partitions.size(); ++i) {
    std::cout << "Part " << (i + 1) << ": [" << std::setw(12)
              << partitions[i].start << ", " // Added setw for alignment
              << std::setw(12) << partitions[i].end << "] "
              << "Average: " << std::setw(12) << partitions[i].average
              << " (Hex: " << floatToHex(partitions[i].average) << ")\n";
  }
  std::cout << std::endl;
}

// Custom exception for out-of-range values
class ValueOutOfRangeException : public std::runtime_error {
public:
  ValueOutOfRangeException(float value, float min, float max)
      : std::runtime_error(
            "Value " + std::to_string(value) + " is outside the valid range [" +
            std::to_string(min) + ", " + std::to_string(max) + "]") {}
};

// Binary search function to find partition index for a value
uint8_t findPartitionIndex(const std::vector<PartitionInfo> &partitions,
                           float value) {
  if (partitions.empty()) {
    throw std::invalid_argument("Partition vector is empty");
  }

  // Check range
  if (value < partitions[0].start || value > partitions.back().end) {
    throw ValueOutOfRangeException(value, partitions[0].start,
                                   partitions.back().end);
  }

  int left = 0;
  int right = partitions.size() - 1;

  while (left <= right) {
    int mid = left + (right - left) / 2;

    // Check if value is in current partition
    if (value >= partitions[mid].start && value <= partitions[mid].end) {
      return static_cast<uint8_t>(mid);
    }

    // If value is less than partition start, search left half
    if (value < partitions[mid].start) {
      right = mid - 1;
    }
    // If value is greater than partition end, search right half
    else {
      left = mid + 1;
    }
  }

  // This should never happen if partitions are contiguous
  throw std::runtime_error(
      "Value falls between partitions - possible gap in partition definitions");
}

bool EmbeddingSearchMappedFloat::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors){return false;}

bool EmbeddingSearchMappedFloat::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors, double distrib_factor) {
  num_vectors = input_vectors.size();
  vector_dim = input_vectors[0].size();

  try {
    std::vector<float> flat_input = flattenMatrix(input_vectors);
    std::vector<PartitionInfo> partitions =
        partitionAndAverage(flat_input, 256, DistributionType::GAUSSIAN, distrib_factor);
    // printPartitions(partitions);

    embeddings.resize(num_vectors, std::vector<uint8_t>(vector_dim));

#pragma omp parallel for
    for (int i = 0; i < num_vectors; i++) {
      for (int j = 0; j < vector_dim; j++) {
        embeddings[i][j] = findPartitionIndex(partitions, input_vectors[i][j]);
      }
    }
    for (int i = 0; i < partitions.size(); i++) {
      mapped_floats[i] = partitions[i].average;
    }

    /*
    Map results for float multiplication
    for (int i = 0; i < MUL_RESULTS_SIZE; i++) {
      for (int j = 0; j < MUL_RESULTS_SIZE; j++) {
        mapped_floats_mul_result[i][j] = mapped_floats[i] * mapped_floats[j];
      }
    }
    */

  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed in setEmbeddings: " << e.what()
              << '\n';
    std::cerr << "Available system memory might be insufficient\n";
    throw; // Re-throw the exception
  }
  // exit(0);

  return true;
}

std::vector<std::pair<float, size_t>>
EmbeddingSearchMappedFloat::similarity_search(const std::vector<uint8_t> &query,
                                              size_t k) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<float, size_t>> similarities;
  similarities.reserve(embeddings.size());

  for (size_t i = 0; i < embeddings.size(); ++i) {
    float sim = cosine_similarity(query, embeddings[i]);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<float, size_t>>(similarities.begin(),
                                               similarities.begin() + k);
}

float EmbeddingSearchMappedFloat::cosine_similarity(
    const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
  float dot_product = 0.0f;

  for (size_t i = 0; i < a.size(); i++) {
    dot_product += mapped_floats[a[i]] * mapped_floats[b[i]];
  }

  return dot_product;
}