#pragma once

#include <cstddef>

struct TimingBreakdown {
  double query_sketch_ms = 0.0;
  double binary_scan_ms = 0.0;
  double candidate_selection_ms = 0.0;
  double rescore_ms = 0.0;
  double final_topk_ms = 0.0;
  double total_ms = 0.0;
  size_t num_survivors = 0;
};
