// matrix_profile_cpu.cpp - CPU implementation of Matrix Profile for fallback
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "matrix_profile.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace MatrixProfile {

// CPU implementation of distance computation
inline uint32_t compute_distance_cpu(const uint8_t* seq1,
                                     const uint8_t* seq2,
                                     int length,
                                     const std::array<std::array<uint8_t, 256>, 256>& lut) {
  uint32_t total_dist = 0;
  for (int i = 0; i < length; i++) {
    total_dist += lut[seq1[i]][seq2[i]];
  }
  return total_dist;
}

// CPU Matrix Profile computation using STOMP-like algorithm
class MatrixProfileCPU {
private:
  MatrixProfileConfig config_;
  std::array<std::array<uint8_t, 256>, 256> distance_lut_;

  // Performance tracking
  mutable PerformanceStats last_stats_;

public:
  MatrixProfileCPU(const MatrixProfileConfig& config) : config_(config) {}

  void upload_distance_lut(const uint8_t lut[256][256]) {
    std::memcpy(distance_lut_.data(), lut, sizeof(distance_lut_));
  }

  void compute_profiles(const std::vector<std::vector<uint8_t>>& sequences,
                        const std::vector<size_t>& seq_lengths,
                        const std::vector<int>& window_sizes,
                        std::vector<std::vector<uint16_t>>& out_profiles,
                        std::vector<std::vector<uint32_t>>& out_indices) {
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t num_agents = sequences.size();
    out_profiles.resize(num_agents);
    out_indices.resize(num_agents);

    // Track total comparisons
    size_t total_comparisons = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : total_comparisons)
#endif
    for (size_t agent_idx = 0; agent_idx < num_agents; agent_idx++) {
      const auto& seq = sequences[agent_idx];
      size_t seq_length = seq_lengths[agent_idx];

      // Process each window size
      for (int window_size : window_sizes) {
        if (window_size > static_cast<int>(seq_length)) continue;

        size_t profile_length = seq_length - window_size + 1;
        std::vector<uint16_t> profile(profile_length, UINT16_MAX);
        std::vector<uint32_t> indices(profile_length, 0);

        // Compute matrix profile for this sequence and window size
        size_t local_comparisons = 0;
        compute_self_matrix_profile(
            seq.data(), seq_length, window_size, profile.data(), indices.data(), local_comparisons);

        total_comparisons += local_comparisons;

        // Store results (for simplicity, we overwrite with last window size)
        out_profiles[agent_idx] = std::move(profile);
        out_indices[agent_idx] = std::move(indices);
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Update performance stats
    last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    last_stats_.gpu_compute_time_ms = 0;      // No GPU
    last_stats_.memory_transfer_time_ms = 0;  // No memory transfer
    last_stats_.total_comparisons = total_comparisons;
    last_stats_.comparisons_per_second = (total_comparisons / last_stats_.total_time_ms) * 1000.0f;
  }

  PerformanceStats get_last_performance_stats() const {
    return last_stats_;
  }

private:
  void compute_self_matrix_profile(const uint8_t* sequence,
                                   size_t seq_length,
                                   int window_size,
                                   uint16_t* profile,
                                   uint32_t* indices,
                                   size_t& total_comparisons) {
    size_t profile_length = seq_length - window_size + 1;
    int exclusion_zone = window_size / 2;

    // Use STOMP-like approach with sliding window optimization
    // For CPU, we'll use a simpler O(n²) approach with optimizations

    for (size_t i = 0; i < profile_length; i++) {
      uint32_t min_distance = UINT32_MAX;
      uint32_t min_index = 0;

      // Compare with all other subsequences
      for (size_t j = 0; j < profile_length; j++) {
        // Skip self-matches and exclusion zone
        if (std::abs(static_cast<int>(i) - static_cast<int>(j)) < exclusion_zone) {
          continue;
        }

        // Compute distance
        uint32_t dist = compute_distance_cpu(sequence + i, sequence + j, window_size, distance_lut_);
        total_comparisons++;

        if (dist < min_distance) {
          min_distance = dist;
          min_index = j;
        }
      }

      profile[i] = static_cast<uint16_t>(std::min(min_distance, 65535u));
      indices[i] = min_index;
    }
  }
};

// Update MatrixProfiler implementation to support CPU fallback
#ifdef CUDA_BEHAVIORAL_ANALYSIS_DISABLED

// Define PerformanceStats type alias for CPU mode
using PerformanceStats = MatrixProfiler::PerformanceStats;

MatrixProfiler::MatrixProfiler(const MatrixProfileConfig& config) : config_(config) {
  // Use CPU implementation when CUDA is not available
  gpu_impl_ = std::make_unique<MatrixProfileCPU>(config);

  std::cout << "MatrixProfiler initialized with CPU implementation\n";

#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
  std::cout << "  OpenMP enabled with " << num_threads << " threads\n";
#else
  std::cout << "  OpenMP not available - using single-threaded computation\n";
#endif
}

MatrixProfiler::~MatrixProfiler() = default;

void MatrixProfiler::initialize(const ActionDistance::ActionDistanceLUT& distance_lut) {
  // Get the distance table
  uint8_t lut[256][256];
  distance_lut.get_distance_table(lut);

  // Store locally
  std::memcpy(distance_lut_.data(), lut, sizeof(lut));

  // Upload to CPU implementation
  static_cast<MatrixProfileCPU*>(gpu_impl_.get())->upload_distance_lut(lut);

  lut_initialized_ = true;

  std::cout << "MatrixProfiler: Distance LUT initialized for CPU computation\n";
}

MatrixProfiler::EncodedSequences MatrixProfiler::encode_agent_histories(const std::vector<Agent*>& agents) const {
  EncodedSequences encoded;
  encoded.sequences.reserve(agents.size());
  encoded.valid_lengths.reserve(agents.size());
  encoded.agent_ids.reserve(agents.size());

  for (const auto* agent : agents) {
    size_t history_length = agent->history_count;
    if (history_length == 0) continue;

    // Get action history
    std::vector<ActionType> actions(history_length);
    std::vector<ActionArg> args(history_length);
    agent->copy_history_to_buffers(actions.data(), args.data());

    // Encode to uint8
    std::vector<uint8_t> encoded_seq(history_length);
    for (size_t i = 0; i < history_length; i++) {
      // Simple encoding for CPU version
      encoded_seq[i] = static_cast<uint8_t>(actions[i] * 10 + args[i]);
    }

    encoded.sequences.push_back(std::move(encoded_seq));
    encoded.valid_lengths.push_back(history_length);
    encoded.agent_ids.push_back(agent->agent_id);
  }

  return encoded;
}

std::vector<AgentMatrixProfile> MatrixProfiler::compute_profiles(const std::vector<Agent*>& agents,
                                                                 const std::vector<int>& window_sizes) {
  if (!lut_initialized_) {
    throw std::runtime_error("MatrixProfiler not initialized. Call initialize() first.");
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Encode agent histories
  auto encoded = encode_agent_histories(agents);
  if (encoded.sequences.empty()) {
    return {};
  }

  // Use provided window sizes or defaults
  const auto& windows = window_sizes.empty() ? config_.window_sizes : window_sizes;

  // Prepare output structures
  std::vector<std::vector<uint16_t>> profiles(encoded.sequences.size());
  std::vector<std::vector<uint32_t>> indices(encoded.sequences.size());

  // Compute profiles on CPU
  static_cast<MatrixProfileCPU*>(gpu_impl_.get())
      ->compute_profiles(encoded.sequences, encoded.valid_lengths, windows, profiles, indices);

  // Build result structures
  std::vector<AgentMatrixProfile> results;
  results.reserve(encoded.sequences.size());

  for (size_t i = 0; i < encoded.sequences.size(); i++) {
    AgentMatrixProfile agent_profile;
    agent_profile.agent_id = encoded.agent_ids[i];
    agent_profile.sequence_length = encoded.valid_lengths[i];

    for (int window_size : windows) {
      AgentMatrixProfile::WindowResult window_result;
      window_result.window_size = window_size;
      window_result.distances = profiles[i];
      window_result.indices = indices[i];

      // Find top motifs
      window_result.top_motifs = Analysis::find_top_motifs(profiles[i], indices[i], window_size);

      agent_profile.window_results.push_back(std::move(window_result));
    }

    results.push_back(std::move(agent_profile));
  }

  // Update performance stats
  auto end_time = std::chrono::high_resolution_clock::now();
  last_stats_ = static_cast<MatrixProfileCPU*>(gpu_impl_.get())->get_last_performance_stats();
  last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

  // Log performance info
  std::cout << "MatrixProfiler CPU Performance:\n"
            << "  Total time: " << last_stats_.total_time_ms << " ms\n"
            << "  Total comparisons: " << last_stats_.total_comparisons << "\n"
            << "  Comparisons/second: " << last_stats_.comparisons_per_second << "\n";

  return results;
}

// Helper function for cross-agent pattern finding
std::vector<CrossAgentPatterns::SharedMotif> find_shared_motifs_cpu(
    const std::vector<uint8_t>& seq1,
    const std::vector<uint8_t>& seq2,
    int window_size,
    float distance_threshold,
    GridObjectId agent1_id,
    GridObjectId agent2_id,
    const std::array<std::array<uint8_t, 256>, 256>& distance_lut) {
  std::vector<CrossAgentPatterns::SharedMotif> motifs;

  if (seq1.size() < static_cast<size_t>(window_size) || seq2.size() < static_cast<size_t>(window_size)) {
    return motifs;
  }

  // Simple sliding window comparison
  for (size_t i = 0; i <= seq1.size() - window_size; i++) {
    for (size_t j = 0; j <= seq2.size() - window_size; j++) {
      uint32_t dist = compute_distance_cpu(seq1.data() + i, seq2.data() + j, window_size, distance_lut);

      if (dist <= distance_threshold) {
        CrossAgentPatterns::SharedMotif motif;
        motif.agent1_id = agent1_id;
        motif.agent2_id = agent2_id;
        motif.agent1_idx = i;
        motif.agent2_idx = j;
        motif.length = window_size;
        motif.distance = static_cast<uint16_t>(dist);
        motifs.push_back(motif);
      }
    }
  }

  return motifs;
}

CrossAgentPatterns MatrixProfiler::find_cross_agent_patterns(const std::vector<Agent*>& agents,
                                                             int window_size,
                                                             float distance_threshold) {
  CrossAgentPatterns patterns;

  if (!lut_initialized_ || agents.size() < 2) {
    return patterns;
  }

  // Encode agent histories
  auto encoded = encode_agent_histories(agents);

  // Simple CPU implementation: compare all pairs of agents
  for (size_t i = 0; i < encoded.sequences.size(); i++) {
    for (size_t j = i + 1; j < encoded.sequences.size(); j++) {
      auto shared = find_shared_motifs_cpu(encoded.sequences[i],
                                           encoded.sequences[j],
                                           window_size,
                                           distance_threshold,
                                           encoded.agent_ids[i],
                                           encoded.agent_ids[j],
                                           distance_lut_);

      patterns.shared_motifs.insert(patterns.shared_motifs.end(), shared.begin(), shared.end());
    }
  }

  return patterns;
}

void MatrixProfiler::update_agent(const Agent* agent) {
  // For CPU version, we don't maintain incremental state
  // Just log that update was called
  if (agent) {
    std::cout << "MatrixProfiler: Agent " << agent->agent_id << " updated (CPU mode - no incremental update)\n";
  }
}

void MatrixProfiler::batch_update(const std::vector<Agent*>& updated_agents) {
  // For CPU version, we don't maintain incremental state
  std::cout << "MatrixProfiler: Batch update of " << updated_agents.size()
            << " agents (CPU mode - no incremental update)\n";
}

size_t MatrixProfiler::get_gpu_memory_usage() const {
  return 0;  // No GPU memory in CPU mode
}

void MatrixProfiler::clear_cache() {
  // Nothing to clear in CPU mode
  std::cout << "MatrixProfiler: Cache cleared (CPU mode)\n";
}

// Implementation of Analysis functions for CPU
namespace Analysis {

std::vector<AgentMatrixProfile::WindowResult::Motif> find_top_motifs(const std::vector<uint16_t>& matrix_profile,
                                                                     const std::vector<uint32_t>& profile_indices,
                                                                     int window_size,
                                                                     int top_k,
                                                                     float exclusion_zone_factor) {
  if (matrix_profile.empty()) return {};

  std::vector<AgentMatrixProfile::WindowResult::Motif> motifs;
  std::vector<bool> used(matrix_profile.size(), false);

  int exclusion_zone = static_cast<int>(window_size * exclusion_zone_factor);

  // Find top-k motifs
  for (int k = 0; k < top_k && motifs.size() < static_cast<size_t>(top_k); k++) {
    uint16_t min_dist = UINT16_MAX;
    size_t min_idx = 0;

    // Find minimum unused distance
    for (size_t i = 0; i < matrix_profile.size(); i++) {
      if (!used[i] && matrix_profile[i] < min_dist) {
        min_dist = matrix_profile[i];
        min_idx = i;
      }
    }

    if (min_dist == UINT16_MAX) break;  // No more motifs

    // Create motif
    AgentMatrixProfile::WindowResult::Motif motif;
    motif.start_idx = min_idx;
    motif.match_idx = profile_indices[min_idx];
    motif.distance = min_dist;
    motif.length = window_size;
    motifs.push_back(motif);

    // Mark exclusion zone
    for (int i = std::max(0, static_cast<int>(min_idx) - exclusion_zone);
         i < std::min(static_cast<int>(matrix_profile.size()), static_cast<int>(min_idx) + exclusion_zone + 1);
         i++) {
      used[i] = true;
    }

    // Also mark around the match
    for (int i = std::max(0, static_cast<int>(motif.match_idx) - exclusion_zone);
         i < std::min(static_cast<int>(matrix_profile.size()), static_cast<int>(motif.match_idx) + exclusion_zone + 1);
         i++) {
      used[i] = true;
    }
  }

  return motifs;
}

}  // namespace Analysis

#endif  // CUDA_BEHAVIORAL_ANALYSIS_DISABLED

}  // namespace MatrixProfile
