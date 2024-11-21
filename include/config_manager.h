// config_manager.h
#pragma once
#include "config.h"
#include <memory>
#include <mutex>

class ConfigManager {
public:
  // Get instance of config manager
  static ConfigManager &getInstance() {
    static ConfigManager instance;
    return instance;
  }

  // Initialize with config file
  void initialize(const std::string &configFile) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_) {
      throw std::runtime_error("ConfigManager already initialized");
    }
    config_ = std::make_unique<config::SearchConfig>(
        config::SearchConfig::fromFile(configFile));
  }

  // Get config
  const config::SearchConfig &getConfig() const {
    if (!config_) {
      throw std::runtime_error("ConfigManager not initialized");
    }
    return *config_;
  }

  // Prevent copying and assignment
  ConfigManager(const ConfigManager &) = delete;
  ConfigManager &operator=(const ConfigManager &) = delete;

private:
  ConfigManager() = default;
  std::unique_ptr<config::SearchConfig> config_;
  std::mutex mutex_;
};

// Example of thread-safe reference wrapper for convenient access
class ConfigRef {
public:
  static const config::SearchConfig &get() {
    return ConfigManager::getInstance().getConfig();
  }
};