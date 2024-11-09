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

std::vector<PartitionInfo> partitionAndAverage(std::vector<float> &arr,
                                               int n_parts) {
  try {
    if (arr.empty()) {
      throw std::invalid_argument("Input array is empty");
    }
    if (n_parts <= 0 || n_parts > arr.size()) {
      throw std::invalid_argument("Invalid number of parts");
    }

    // Sort the array
    std::sort(arr.begin(), arr.end());

    // Calculate dynamic partition sizes
    std::vector<size_t> partition_sizes(n_parts);
    float center = (n_parts - 1) / 2.0f;
    double total_weights = 0.0; // Changed to double for better precision
    std::vector<double> weights(n_parts); // Store weights separately

    // First pass: calculate weights
    for (int i = 0; i < n_parts; i++) {
      float distance = (i - center) / center;
      // cubic
      // double weight = 1.0 - (distance * distance * distance * 0.95); // Changed to double
      // gauss
      double weight = std::exp(-distance * distance * 8.0f);
      weights[i] = weight;
      total_weights += weight;
    }

    // Second pass: convert weights to actual sizes
    size_t elements_assigned = 0;
    for (int i = 0; i < n_parts - 1; i++) {
      // Calculate size and ensure it's at least 1
      double proportion = weights[i] / total_weights;
      partition_sizes[i] =
          std::max(size_t(1), static_cast<size_t>(arr.size() * proportion));

      // Prevent overshooting total size
      if (elements_assigned + partition_sizes[i] >= arr.size()) {
        partition_sizes[i] = arr.size() - elements_assigned;
        // Zero out remaining partitions
        for (int j = i + 1; j < n_parts; j++) {
          partition_sizes[j] = 0;
        }
        break;
      }
      elements_assigned += partition_sizes[i];
    }

    // Last partition gets remaining elements
    if (elements_assigned < arr.size()) {
      partition_sizes[n_parts - 1] = arr.size() - elements_assigned;
    }

    // Pre-allocate result vector
    std::vector<PartitionInfo> partitions(n_parts);

    // Process each partition
    size_t start_pos = 0;
    for (int i = 0; i < n_parts; ++i) {
      if (partition_sizes[i] == 0) {
        // Skip empty partitions
        continue;
      }

      // Calculate sum for current partition
      float sum = 0.0f;
      for (size_t j = 0; j < partition_sizes[i]; ++j) {
        sum += arr[start_pos + j];
      }

      float average = sum / partition_sizes[i];

      // Store partition info
      partitions[i] = {arr[start_pos],                          // start
                       arr[start_pos + partition_sizes[i] - 1], // end
                       average};

      // Print partition details
      std::cout << "Partition " << std::setw(2) << i << ": "
                << "size = " << std::setw(6) << partition_sizes[i]
                << " elements, range [" << std::fixed << std::setprecision(2)
                << partitions[i].start << ", " << partitions[i].end
                << "], avg = " << partitions[i].average << "\n";

      // Update start position for next partition
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
    const std::vector<std::vector<float>> &input_vectors) {
  num_vectors = input_vectors.size();
  vector_dim = input_vectors[0].size();

  try {
    std::vector<float> flat_input = flattenMatrix(input_vectors);
    std::vector<PartitionInfo> partitions =
        partitionAndAverage(flat_input, 16);
    printPartitions(partitions);

    embeddings.resize(num_vectors, std::vector<uint8_t>(vector_dim));

    for (int i = 0; i < num_vectors; i++) {
      for (int j = 0; j < vector_dim; j++) {
        embeddings[i][j] = findPartitionIndex(partitions, input_vectors[i][j]);
      }
    }
    for (int i = 0; i < partitions.size(); i++) {
      mapped_floats[i] = partitions[i].average;
    }

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

  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += mapped_floats[a[i]] * mapped_floats[b[i]];
  }

  return dot_product;
}