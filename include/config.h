// config.h
#pragma once
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace config {

// Forward declarations
class ConfigException;
struct SearchConfig;

// Custom exception for configuration errors
class ConfigException : public std::runtime_error {
public:
  enum class ErrorCode {
    FileNotFound,
    ParseError,
    ValidationError,
    TypeError,
    MissingField
  };

  ConfigException(ErrorCode code, const std::string &message)
      : std::runtime_error(message), code_(code) {}

  ErrorCode code() const { return code_; }

private:
  ErrorCode code_;
};

// Performance-related configuration
struct PerformanceConfig {
  size_t batchSize = 256;
  size_t numThreads = 8;

  // Validate performance settings
  void validate() const {
    if (batchSize == 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "batchSize must be greater than 0");
    if (numThreads == 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "numThreads must be greater than 0");
  }

  // JSON serialization
  nlohmann::json toJson() const {
    return {{"batchSize", batchSize}, {"numThreads", numThreads}};
  }

  // JSON deserialization
  static PerformanceConfig fromJson(const nlohmann::json &j) {
    PerformanceConfig config;
    config.batchSize = j.value("batchSize", config.batchSize);
    config.numThreads = j.value("numThreads", config.numThreads);
    return config;
  }
};

// Algorithm-specific configuration
struct AlgorithmConfig {
  size_t k = 100;
  size_t rescoringFactor = 50;

  // Validate algorithm settings
  void validate() const {
    if (k == 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "k must be greater than 0");
    if (rescoringFactor == 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "rescoringFactor must be greater than 0");
  }

  // JSON serialization
  nlohmann::json toJson() const {
    return {{"k", k}, {"rescoringFactor", rescoringFactor}};
  }

  // JSON deserialization
  static AlgorithmConfig fromJson(const nlohmann::json &j) {
    AlgorithmConfig config;
    config.k = j.value("maxResults", config.k);
    config.rescoringFactor = j.value("rescoringFactor", config.rescoringFactor);
    return config;
  }
};

// Memory-related configuration
struct MemoryConfig {
  size_t maxCacheSize = 1024 * 1024 * 1024; // 1GB
  size_t alignmentSize = 32;
  bool prefetchEnabled = true;
  size_t prefetchDistance = 8;

  // Validate memory settings
  void validate() const {
    if (maxCacheSize == 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "maxCacheSize must be greater than 0");
    if (alignmentSize == 0 || (alignmentSize & (alignmentSize - 1)) != 0)
      throw ConfigException(ConfigException::ErrorCode::ValidationError,
                            "alignmentSize must be a power of 2");
    if (prefetchEnabled && prefetchDistance == 0)
      throw ConfigException(
          ConfigException::ErrorCode::ValidationError,
          "prefetchDistance must be greater than 0 when prefetch is enabled");
  }

  // JSON serialization
  nlohmann::json toJson() const {
    return {{"maxCacheSize", maxCacheSize},
            {"alignmentSize", alignmentSize},
            {"prefetchEnabled", prefetchEnabled},
            {"prefetchDistance", prefetchDistance}};
  }

  // JSON deserialization
  static MemoryConfig fromJson(const nlohmann::json &j) {
    MemoryConfig config;
    config.maxCacheSize = j.value("maxCacheSize", config.maxCacheSize);
    config.alignmentSize = j.value("alignmentSize", config.alignmentSize);
    config.prefetchEnabled = j.value("prefetchEnabled", config.prefetchEnabled);
    config.prefetchDistance =
        j.value("prefetchDistance", config.prefetchDistance);
    return config;
  }
};

// Main configuration struct
struct SearchConfig {
  PerformanceConfig performance;
  AlgorithmConfig algorithm;
  MemoryConfig memory;

  // Validate all configurations
  void validate() const {
    performance.validate();
    algorithm.validate();
    memory.validate();
  }

  // Load configuration from JSON file
  static SearchConfig fromFile(const std::string &filename) {
    if (!std::filesystem::exists(filename)) {
      throw ConfigException(ConfigException::ErrorCode::FileNotFound,
                            "Configuration file not found: " + filename);
    }

    try {
      std::ifstream file(filename);
      nlohmann::json j = nlohmann::json::parse(file);

      SearchConfig config;
      if (j.contains("performance")) {
        config.performance = PerformanceConfig::fromJson(j["performance"]);
      }
      if (j.contains("algorithm")) {
        config.algorithm = AlgorithmConfig::fromJson(j["algorithm"]);
      }
      if (j.contains("memory")) {
        config.memory = MemoryConfig::fromJson(j["memory"]);
      }

      config.validate();
      return config;
    } catch (const nlohmann::json::exception &e) {
      throw ConfigException(ConfigException::ErrorCode::ParseError,
                            "Failed to parse configuration file: " +
                                std::string(e.what()));
    }
  }

  // Save configuration to JSON file
  void saveToFile(const std::string &filename) const {
    try {
      nlohmann::json j = {{"performance", performance.toJson()},
                          {"algorithm", algorithm.toJson()},
                          {"memory", memory.toJson()}};

      std::ofstream file(filename);
      file << j.dump(4);
    } catch (const std::exception &e) {
      throw ConfigException(ConfigException::ErrorCode::ParseError,
                            "Failed to save configuration: " +
                                std::string(e.what()));
    }
  }
};

} // namespace config