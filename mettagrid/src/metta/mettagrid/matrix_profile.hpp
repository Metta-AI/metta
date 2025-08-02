#ifndef MATRIX_PROFILE_HPP_
#define MATRIX_PROFILE_HPP_

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "action_distance.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

namespace MatrixProfile {

// Configuration for matrix profile computation
struct MatrixProfileConfig {
  std::vector<int> window_sizes = {10, 25, 50, 100};  // Multiple scales
  int min_window_size = 4;
  int max_window_size = 200;
  bool use_multi_gpu = true;
  int gpu_device_id = 0;  // For single GPU mode
  size_t max_agents_per_gpu = 128;

  // Performance tuning
  int block_size = 256;
  int streams_per_gpu = 2;
  bool use_shared_memory = true;
  bool use_texture_memory = false;  // For LUT access
};

// Results structure for a single agent's matrix profile
struct AgentMatrixProfile {
  GridObjectId agent_id;
  size_t sequence_length;

  // For each window size
  struct WindowResult {
    int window_size;
    std::vector<uint16_t> distances;  // Matrix profile values
    std::vector<uint32_t> indices;    // Nearest neighbor indices

    // Behavioral patterns discovered
    struct Motif {
      uint32_t start_idx;
      uint32_t match_idx;
      uint16_t distance;
      int length;
    };
    std::vector<Motif> top_motifs;
  };

  std::vector<WindowResult> window_results;
};

// Cross-agent analysis results
struct CrossAgentPatterns {
  struct SharedMotif {
    GridObjectId agent1_id;
    GridObjectId agent2_id;
    uint32_t agent1_idx;
    uint32_t agent2_idx;
    int length;
    uint16_t distance;
  };

  std::vector<SharedMotif> shared_motifs;

  // Behavioral clusters
  struct BehaviorCluster {
    std::vector<GridObjectId> agent_ids;
    std::vector<uint8_t> representative_sequence;
    float avg_intra_cluster_distance;
  };

  std::vector<BehaviorCluster> clusters;
};

// Forward declaration of CUDA implementation
class MatrixProfileGPU;

class MatrixProfiler {
public:
  MatrixProfiler(const MatrixProfileConfig& config = MatrixProfileConfig());
  ~MatrixProfiler();

  // Initialize with action distance LUT
  void initialize(const ActionDistance::ActionDistanceLUT& distance_lut);

  // Process agents' action histories
  std::vector<AgentMatrixProfile> compute_profiles(const std::vector<Agent*>& agents,
                                                   const std::vector<int>& window_sizes = {});

  // Cross-agent pattern discovery
  CrossAgentPatterns find_cross_agent_patterns(const std::vector<Agent*>& agents,
                                               int window_size,
                                               float distance_threshold = 5.0f);

  // Real-time analysis during training
  void update_agent(const Agent* agent);
  void batch_update(const std::vector<Agent*>& updated_agents);

  // Performance monitoring
  struct PerformanceStats {
    float total_time_ms = 0;
    float gpu_compute_time_ms = 0;
    float memory_transfer_time_ms = 0;
    float pattern_discovery_time_ms = 0;
    size_t total_comparisons = 0;
    float comparisons_per_second = 0;
  };

  PerformanceStats get_last_performance_stats() const {
    return last_stats_;
  }

  // Memory management
  size_t get_gpu_memory_usage() const;
  void clear_cache();

private:
  MatrixProfileConfig config_;
  std::unique_ptr<MatrixProfileGPU> gpu_impl_;
  PerformanceStats last_stats_{};

  // Action encoding cache
  struct EncodedSequences {
    std::vector<std::vector<uint8_t>> sequences;
    std::vector<size_t> valid_lengths;
    std::vector<GridObjectId> agent_ids;
  };

  EncodedSequences encode_agent_histories(const std::vector<Agent*>& agents) const;

  // Distance LUT storage
  std::array<std::array<uint8_t, 256>, 256> distance_lut_;
  bool lut_initialized_ = false;
};

#ifdef CUDA_DISABLED

// ======================= STUB IMPLEMENTATIONS =======================
// Used when CUDA is not available (e.g., on macOS)

// Stub implementation of MatrixProfileGPU
class MatrixProfileGPU {
public:
  MatrixProfileGPU(const MatrixProfileConfig&) {
    throw std::runtime_error("Matrix Profile GPU implementation requires CUDA support");
  }

  void upload_distance_lut(const uint8_t[256][256]) {
    throw std::runtime_error("CUDA support not available");
  }

  void allocate_memory(size_t, size_t) {
    throw std::runtime_error("CUDA support not available");
  }

  void compute_profiles(const std::vector<std::vector<uint8_t>>&,
                        const std::vector<size_t>&,
                        const std::vector<int>&,
                        std::vector<std::vector<uint16_t>>&,
                        std::vector<std::vector<uint32_t>>&) {
    throw std::runtime_error("CUDA support not available");
  }
};

// MatrixProfiler stub implementation
inline MatrixProfiler::MatrixProfiler(const MatrixProfileConfig& config) : config_(config) {
  std::cerr << "WARNING: MatrixProfiler created without CUDA support. "
            << "Behavioral analysis features will not be available.\n";
}

inline MatrixProfiler::~MatrixProfiler() = default;

inline void MatrixProfiler::initialize(const ActionDistance::ActionDistanceLUT& distance_lut) {
  // Store the LUT even in stub mode
  distance_lut.get_distance_table(distance_lut_.data());
  lut_initialized_ = true;

  std::cerr << "WARNING: MatrixProfiler initialized in stub mode (no CUDA).\n";
}

inline MatrixProfiler::EncodedSequences MatrixProfiler::encode_agent_histories(const std::vector<Agent*>&) const {
  return EncodedSequences{};
}

inline std::vector<AgentMatrixProfile> MatrixProfiler::compute_profiles(const std::vector<Agent*>&,
                                                                        const std::vector<int>&) {
  throw std::runtime_error(
      "Matrix Profile computation requires CUDA support. "
      "Please build with CUDA enabled on a system with NVIDIA GPU.");
}

inline CrossAgentPatterns MatrixProfiler::find_cross_agent_patterns(const std::vector<Agent*>&, int, float) {
  throw std::runtime_error("Cross-agent pattern discovery requires CUDA support");
}

inline void MatrixProfiler::update_agent(const Agent*) {
  // No-op in stub mode
}

inline void MatrixProfiler::batch_update(const std::vector<Agent*>&) {
  // No-op in stub mode
}

inline size_t MatrixProfiler::get_gpu_memory_usage() const {
  return 0;
}

inline void MatrixProfiler::clear_cache() {
  // No-op in stub mode
}

#else  // CUDA_DISABLED

// ======================= CUDA IMPLEMENTATIONS =======================
// These are defined in matrix_profile.cu when CUDA is available
// Just declare them here - the actual implementations are in the .cu file

#endif  // CUDA_DISABLED

// Utility functions for behavioral analysis
namespace Analysis {

// Find recurring patterns within a single agent
inline std::vector<AgentMatrixProfile::WindowResult::Motif> find_top_motifs(
    const std::vector<uint16_t>& matrix_profile,
    const std::vector<uint32_t>& profile_indices,
    int window_size,
    int top_k = 10,
    float exclusion_zone_factor = 0.5f) {
#ifdef CUDA_DISABLED
  return {};
#else
  // Implementation would be in matrix_profile.cu
  std::vector<AgentMatrixProfile::WindowResult::Motif> motifs;
  // TODO: Implement proper motif discovery algorithm
  return motifs;
#endif
}

// Compute behavioral similarity between agents
inline float compute_agent_similarity(const AgentMatrixProfile& profile1,
                                      const AgentMatrixProfile& profile2,
                                      int window_size) {
#ifdef CUDA_DISABLED
  return 0.0f;
#else
  // TODO: Implement similarity computation
  return 0.0f;
#endif
}

// Cluster agents by behavioral patterns
inline std::vector<CrossAgentPatterns::BehaviorCluster>
cluster_by_behavior(const std::vector<AgentMatrixProfile>& profiles, int window_size, int num_clusters = 0) {
#ifdef CUDA_DISABLED
  return {};
#else
  std::vector<CrossAgentPatterns::BehaviorCluster> clusters;
  // TODO: Implement clustering algorithm
  return clusters;
#endif
}

// Detect emergent strategies
struct EmergentStrategy {
  std::string name;
  std::vector<uint8_t> pattern;
  std::vector<GridObjectId> agents_using;
  float prevalence;  // Percentage of agents using this strategy
};

inline std::vector<EmergentStrategy> detect_strategies(const std::vector<AgentMatrixProfile>& profiles,
                                                       const ActionDistance::ActionDistanceLUT& lut) {
#ifdef CUDA_DISABLED
  return {};
#else
  std::vector<EmergentStrategy> strategies;
  // TODO: Implement strategy detection
  return strategies;
#endif
}

}  // namespace Analysis

}  // namespace MatrixProfile

#endif  // MATRIX_PROFILE_HPP_
