#ifndef AGENT_BEHAVIOR_HPP_
#define AGENT_BEHAVIOR_HPP_

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "objects/agent.hpp"
#include "types.hpp"

#ifndef CUDA_DISABLED
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#endif

#include "action_distance.hpp"
#include "matrix_profile.hpp"

namespace AgentBehavior {

// Common structures used by both implementations
struct DominantMotif {
  std::string pattern_description;
  std::vector<uint8_t> encoded_pattern;
  float prevalence;
  int window_size;
  float avg_distance;
  std::vector<GridObjectId> agent_ids;
};

struct BehaviorStats {
  std::map<std::string, float> action_frequencies;
  std::map<std::string, float> pattern_frequencies;
  float behavioral_diversity = 0.0f;
  int num_clusters = 0;
  std::map<std::string, std::vector<GridObjectId>> clusters;
};

class BehaviorAnalyzer {
private:
  ActionDistance::ActionDistanceLUT distance_lut_;
  std::unique_ptr<MatrixProfile::MatrixProfiler> profiler_;
  MatrixProfile::MatrixProfileConfig config_;
  bool initialized_ = false;

#ifndef CUDA_DISABLED
  // Check for GPU availability and configure
  static MatrixProfile::MatrixProfileConfig configure_for_gpus(int num_agents) {
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
      std::cerr << "ERROR: No CUDA devices found. Behavioral analysis requires GPU.\n";
      std::cerr << "Please ensure CUDA is installed and GPUs are available.\n";
      std::exit(1);
    }

    MatrixProfile::MatrixProfileConfig cfg;

    // Use all available GPUs
    cfg.use_multi_gpu = (device_count > 1);

    // Get GPU properties from first device (assume homogeneous)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "BehaviorAnalyzer GPU Configuration:\n"
              << "  GPUs available: " << device_count << "\n"
              << "  GPU model: " << prop.name << "\n"
              << "  GPU memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n"
              << "  Compute capability: " << prop.major << "." << prop.minor << "\n";

    // Calculate agents per GPU
    size_t agents_per_gpu = (num_agents + device_count - 1) / device_count;

    // Verify memory is sufficient
    size_t memory_per_agent = 5 * 1024;  // 5KB per agent (data + workspace)
    size_t required_memory = agents_per_gpu * memory_per_agent;
    size_t available_memory = prop.totalGlobalMem * 0.8;  // Use 80% of memory

    if (required_memory > available_memory) {
      std::cerr << "ERROR: Insufficient GPU memory for " << agents_per_gpu << " agents per GPU\n";
      std::cerr << "Required: " << required_memory / (1024 * 1024) << " MB, "
                << "Available: " << available_memory / (1024 * 1024) << " MB\n";
      std::exit(1);
    }

    cfg.max_agents_per_gpu = agents_per_gpu;

    // Set compute parameters based on GPU architecture
    if (prop.major >= 7) {  // Volta and newer
      cfg.block_size = 256;
      cfg.streams_per_gpu = 4;
    } else if (prop.major >= 6) {  // Pascal
      cfg.block_size = 128;
      cfg.streams_per_gpu = 2;
    } else {
      cfg.block_size = 64;
      cfg.streams_per_gpu = 1;
    }

    // Use shared memory if available
    cfg.use_shared_memory = (prop.sharedMemPerBlock >= 48 * 1024);

    // Default window sizes for behavioral analysis
    cfg.window_sizes = {10, 25, 50, 100};
    cfg.min_window_size = 4;
    cfg.max_window_size = 200;

    std::cout << "  Agents per GPU: " << cfg.max_agents_per_gpu << "\n"
              << "  Block size: " << cfg.block_size << "\n"
              << "  Streams per GPU: " << cfg.streams_per_gpu << "\n"
              << "  Multi-GPU: " << (cfg.use_multi_gpu ? "enabled" : "disabled") << "\n\n";

    return cfg;
  }
#endif

public:
  BehaviorAnalyzer() = default;

  // Initialize with action handlers from MettaGrid
  template <typename ActionHandlerContainer>
  void initialize(const ActionHandlerContainer& action_handlers, int num_agents) {
    if (initialized_) {
      std::cerr << "Warning: BehaviorAnalyzer already initialized\n";
      return;
    }

#ifdef CUDA_DISABLED
    // Configure for CPU
    config_.use_multi_gpu = false;

    std::cout << "BehaviorAnalyzer initialized with CPU fallback\n";
    std::cout << "Note: CPU implementation will be slower than GPU for large datasets\n";
#else
    // Configure for available GPUs (will exit if none found)
    config_ = configure_for_gpus(num_agents);
#endif

    // Build distance LUT
    distance_lut_.register_actions(action_handlers);

    // Create profiler with configuration
    profiler_ = std::make_unique<MatrixProfile::MatrixProfiler>(config_);
    profiler_->initialize(distance_lut_);

    initialized_ = true;

    std::cout << "BehaviorAnalyzer initialized successfully\n";
  }

  // Main analysis function - returns dominant motifs for each window size
  std::vector<DominantMotif> get_dominant_motifs(const std::vector<Agent*>& agents,
                                                 const std::vector<int>& window_sizes) {
    if (!initialized_) {
      std::cerr << "ERROR: BehaviorAnalyzer not initialized\n";
      return {};
    }

    // Use provided window sizes or defaults from config
    const auto& windows = window_sizes.empty() ? config_.window_sizes : window_sizes;

    // Filter agents with sufficient history
    std::vector<Agent*> valid_agents;
    valid_agents.reserve(agents.size());

    for (auto* agent : agents) {
      if (agent->history_count >= static_cast<size_t>(config_.min_window_size)) {
        valid_agents.push_back(agent);
      }
    }

    if (valid_agents.empty()) {
      return {};
    }

    // Log analysis info
    std::cout << "Analyzing " << valid_agents.size() << " agents with window sizes: ";
    for (int ws : windows) std::cout << ws << " ";
#ifdef CUDA_DISABLED
    std::cout << " (CPU mode)\n";
#else
    std::cout << "\n";
#endif

    // Compute matrix profiles
    auto profiles = profiler_->compute_profiles(valid_agents, windows);

    // Extract dominant motifs
    std::vector<DominantMotif> dominant_motifs;

    for (int window_size : windows) {
      auto window_motifs = extract_window_motifs(profiles, window_size, valid_agents);
      dominant_motifs.insert(dominant_motifs.end(), window_motifs.begin(), window_motifs.end());
    }

    // Sort by prevalence (most common patterns first)
    std::sort(dominant_motifs.begin(), dominant_motifs.end(), [](const DominantMotif& a, const DominantMotif& b) {
      return a.prevalence > b.prevalence;
    });

    return dominant_motifs;
  }

  // Get behavioral statistics
  BehaviorStats get_behavior_stats(const std::vector<Agent*>& agents) {
    BehaviorStats stats;

    if (!initialized_ || agents.empty()) {
      return stats;
    }

    // Calculate action frequencies
    std::map<std::string, int> action_counts;
    int total_actions = 0;

    for (const auto* agent : agents) {
      if (agent->history_count == 0) continue;

      // Get action history
      std::vector<ActionType> actions(agent->history_count);
      std::vector<ActionArg> args(agent->history_count);
      agent->copy_history_to_buffers(actions.data(), args.data());

      for (size_t i = 0; i < agent->history_count; i++) {
        auto action_names = distance_lut_.get_action_names();
        if (actions[i] < action_names.size()) {
          action_counts[action_names[actions[i]]]++;
          total_actions++;
        }
      }
    }

    // Convert to frequencies
    if (total_actions > 0) {
      for (const auto& [action, count] : action_counts) {
        stats.action_frequencies[action] = static_cast<float>(count) / total_actions;
      }

      // Compute behavioral diversity (entropy-based)
      float entropy = 0.0f;
      for (const auto& [action, freq] : stats.action_frequencies) {
        if (freq > 0) {
          entropy -= freq * std::log2(freq);
        }
      }
      stats.behavioral_diversity = entropy / std::log2(distance_lut_.get_action_names().size());
    }

    return stats;
  }

  // Get current configuration
  const MatrixProfile::MatrixProfileConfig& get_config() const {
    return config_;
  }

  // Get performance statistics
  using PerformanceStats = MatrixProfile::MatrixProfiler::PerformanceStats;

  PerformanceStats get_performance_stats() const {
    if (!initialized_) {
      return {};
    }
    return profiler_->get_last_performance_stats();
  }

private:
  std::vector<DominantMotif> extract_window_motifs(const std::vector<MatrixProfile::AgentMatrixProfile>& profiles,
                                                   int window_size,
                                                   const std::vector<Agent*>& agents) {
    // Group similar motifs across agents
    struct MotifGroup {
      std::vector<uint8_t> representative_pattern;
      std::vector<GridObjectId> agent_ids;
      std::vector<uint16_t> distances;
    };

    std::vector<MotifGroup> motif_groups;

    // Process each agent's top motifs
    for (size_t agent_idx = 0; agent_idx < profiles.size(); agent_idx++) {
      const auto& profile = profiles[agent_idx];

      // Find window result
      const MatrixProfile::AgentMatrixProfile::WindowResult* window_result = nullptr;
      for (const auto& wr : profile.window_results) {
        if (wr.window_size == window_size) {
          window_result = &wr;
          break;
        }
      }

      if (!window_result || window_result->top_motifs.empty()) continue;

      // Get the agent's action sequence
      std::vector<ActionType> actions(agents[agent_idx]->history_count);
      std::vector<ActionArg> args(agents[agent_idx]->history_count);
      agents[agent_idx]->copy_history_to_buffers(actions.data(), args.data());

      // Process top motif
      const auto& top_motif = window_result->top_motifs[0];

      // Extract and encode the pattern
      std::vector<uint8_t> pattern;
      for (int i = 0; i < window_size; i++) {
        if (top_motif.start_idx + i < actions.size()) {
          uint8_t encoded =
              distance_lut_.encode_action(actions[top_motif.start_idx + i], args[top_motif.start_idx + i]);
          pattern.push_back(encoded);
        }
      }

      // Find or create motif group
      bool found = false;
      for (auto& group : motif_groups) {
        if (distance_lut_.patterns_similar(pattern, group.representative_pattern, 5)) {
          group.agent_ids.push_back(profile.agent_id);
          group.distances.push_back(top_motif.distance);
          found = true;
          break;
        }
      }

      if (!found) {
        MotifGroup new_group;
        new_group.representative_pattern = pattern;
        new_group.agent_ids.push_back(profile.agent_id);
        new_group.distances.push_back(top_motif.distance);
        motif_groups.push_back(new_group);
      }
    }

    // Convert to DominantMotif format
    std::vector<DominantMotif> result;

    for (const auto& group : motif_groups) {
      if (group.agent_ids.size() < 2) continue;  // Skip unique patterns

      DominantMotif motif;
      motif.encoded_pattern = group.representative_pattern;
      motif.pattern_description = distance_lut_.decode_sequence_to_string(group.representative_pattern);
      motif.prevalence = static_cast<float>(group.agent_ids.size()) / agents.size();
      motif.window_size = window_size;
      motif.agent_ids = group.agent_ids;

      // Calculate average distance
      float avg_dist = 0;
      for (uint16_t dist : group.distances) {
        avg_dist += dist;
      }
      motif.avg_distance = avg_dist / group.distances.size();

      result.push_back(motif);
    }

    return result;
  }

  struct AgentCluster {
    std::vector<GridObjectId> agent_ids;
    std::vector<uint8_t> representative_sequence;
  };

  std::vector<AgentCluster> cluster_agents_by_behavior(const std::vector<Agent*>& agents, int window_size) {
    // TODO: Implement proper clustering based on matrix profiles
    // For now, return empty to keep the interface clean
    std::vector<AgentCluster> clusters;
    return clusters;
  }
};

}  // namespace AgentBehavior

// Utility functions available regardless of CUDA support
inline bool is_cuda_available() {
#ifdef CUDA_DISABLED
  return false;
#else
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
#endif
}

inline std::string get_cuda_unavailable_message() {
#ifdef __APPLE__
  return "CUDA is not supported on macOS. Using CPU implementation for behavioral analysis.";
#elif defined(CUDA_DISABLED)
  return "CUDA not found. Using CPU implementation for behavioral analysis (may be slower for large datasets).";
#else
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorString(err) +
           ". Using CPU implementation for behavioral analysis.";
  } else if (device_count == 0) {
    return "No CUDA-capable devices found. Using CPU implementation for behavioral analysis.";
  }

  return "CUDA is available";
#endif
}

inline std::string get_behavior_analysis_info() {
  std::stringstream info;

#ifdef CUDA_DISABLED
  info << "Behavioral Analysis: CPU implementation\n";
#ifdef _OPENMP
  info << "  Parallelization: OpenMP enabled\n";
#else
  info << "  Parallelization: Single-threaded\n";
#endif
  info << "  Performance: Suitable for small to medium datasets (<1000 agents)\n";
  info << "  Note: For large-scale analysis, consider using a CUDA-enabled system\n";
#else
  if (is_cuda_available()) {
    info << "Behavioral Analysis: GPU-accelerated (CUDA)\n";
    info << "  Performance: Optimized for large datasets\n";
  } else {
    info << "Behavioral Analysis: CPU fallback (CUDA runtime not available)\n";
  }
#endif

  return info.str();
}

#endif  // AGENT_BEHAVIOR_HPP_
