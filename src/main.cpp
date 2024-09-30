#include "embedding_search.h"
#include <iostream>
#include <vector>

int main() {
    EmbeddingSearch searcher;

    std::cout << "Hi World" << std::endl;

    // Load embeddings
    if (!searcher.load_safetensors("../data/embeddings.safetensors")) {
        return 1;
    }

    // Example query vector
    std::vector<float> query = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};  // Adjust size to match your embeddings

    // Perform similarity search
    size_t k = 5;  // Number of similar vectors to retrieve
    std::vector<size_t> results = searcher.similarity_search(query, k);

    // Print results
    std::cout << "Top " << k << " similar vectors:" << std::endl;
    for (size_t idx : results) {
        std::cout << "Index: " << idx << std::endl;
    }

    return 0;
}