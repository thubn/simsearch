#pragma once
#include <string>
#include <vector>

namespace EmbeddingIO {
bool load_safetensors(const std::string &filename,
                      std::vector<std::vector<float>> &embeddings,
                      std::vector<std::string> &sentences);
bool load_json(const std::string &filename,
               std::vector<std::vector<float>> &embeddings,
               std::vector<std::string> &sentences);
bool load_json2(const std::string &filename,
                std::vector<std::vector<float>> &embeddings,
                std::vector<std::string> &sentences);
bool load_parquet(const std::string &filename,
                  std::vector<std::vector<float>> &embeddings,
                  std::vector<std::string> &sentences, const bool set_sentences,
                  const int embedding_dim = 1024);
} // namespace EmbeddingIO