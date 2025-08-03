// matrix_profile.cpp - CPU implementation and main logic
#include "matrix_profile.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef CUDA_DISABLED
#include <cuda_runtime.h>
// Forward declare GPU implementation
namespace MatrixProfile {
class MatrixProfileGPU;
}
#endif

namespace MatrixProfile {

// CPU implementation of distance computation
static inline uint32_t compute_distance_cpu(const uint8_t* seq1,
                                            const uint8_t* seq2,
                                            int length,
                                            const std::array<std::array<uint8_t, 256>, 256>& lut) {
  uint32_t total_dist = 0;
  for (int i = 0; i < length; i++) {
    total_dist += lut[seq1[i]][seq2[i]];
  }
  return total_dist;
}

// Helper function for cross-agent pattern finding
static std::vector<CrossAgentPatterns::SharedMotif> find_shared_motifs_cpu(
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
  for (size_t i = 0; i <= seq1.size() - static_cast<size_t>(window_size); i++) {
    for (size_t j = 0; j <= seq2.size() - static_cast<size_t>(window_size); j++) {
      uint32_t dist = compute_distance_cpu(seq1.data() + i, seq2.data() + j, window_size, distance_lut);

      if (dist <= distance_threshold) {
        CrossAgentPatterns::SharedMotif motif;
        motif.agent1_id = agent1_id;
        motif.agent2_id = agent2_id;
        motif.agent1_idx = static_cast<uint32_t>(i);
        motif.agent2_idx = static_cast<uint32_t>(j);
        motif.length = window_size;
        motif.distance = static_cast<uint16_t>(dist);
        motifs.push_back(motif);
      }
    }
  }

  return motifs;
}

// CPU Matrix Profile implementation
class MatrixProfileCPU : public MatrixProfileImpl {
public:
  // Structure to hold results for all window sizes
  struct AgentProfileResults {
    std::vector<std::vector<uint16_t>> window_profiles;  // One per window size
    std::vector<std::vector<uint32_t>> window_indices;   // One per window size
    std::vector<int> window_sizes_used;
  };

private:
  MatrixProfileConfig config_;
  std::array<std::array<uint8_t, 256>, 256> distance_lut_;
  ActionDistance::ActionDistanceLUT* action_lut_ = nullptr;
  bool lut_initialized_ = false;

  // Performance tracking
  mutable MatrixProfiler::PerformanceStats last_stats_;

  // Store full results for all window sizes
  mutable std::vector<AgentProfileResults> last_full_results_;

public:
  MatrixProfileCPU(const MatrixProfileConfig& config) : config_(config) {
    std::cout << "MatrixProfiler CPU implementation initialized\n";
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "  OpenMP enabled with " << num_threads << " threads\n";
#else
    std::cout << "  OpenMP not available - using single-threaded computation\n";
#endif
  }

  void initialize(const ActionDistance::ActionDistanceLUT& distance_lut) override {
    // Get the distance table
    uint8_t lut[256][256];
    distance_lut.get_distance_table(lut);

    // Store locally
    std::memcpy(distance_lut_.data(), lut, sizeof(lut));

    // Store reference to action LUT for encoding
    action_lut_ = const_cast<ActionDistance::ActionDistanceLUT*>(&distance_lut);

    lut_initialized_ = true;

    std::cout << "MatrixProfiler CPU: Distance LUT initialized\n";
  }

  std::vector<AgentMatrixProfile> compute_profiles(const std::vector<Agent*>& agents,
                                                   const std::vector<uint8_t>& window_sizes) override {
    if (!lut_initialized_) {
      throw std::runtime_error("MatrixProfiler not initialized. Call initialize() first.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Encode agent histories
    auto encoded = encode_agent_histories(agents, *action_lut_);
    if (encoded.sequences.empty()) {
      return {};
    }

    // Use provided window sizes or defaults
    const auto& windows = window_sizes.empty() ? config_.window_sizes : window_sizes;

    // Prepare output structures
    std::vector<std::vector<uint16_t>> profiles(encoded.sequences.size());
    std::vector<std::vector<uint32_t>> indices(encoded.sequences.size());

    // Compute profiles
    compute_profiles_internal(encoded.sequences, encoded.valid_lengths, windows, profiles, indices);

    // Get the full results with all window sizes
    const auto& full_results = last_full_results_;

    // Build result structures
    std::vector<AgentMatrixProfile> results;
    results.reserve(encoded.sequences.size());

    for (size_t i = 0; i < encoded.sequences.size(); i++) {
      AgentMatrixProfile agent_profile;
      agent_profile.agent_id = encoded.agent_ids[i];
      agent_profile.sequence_length = encoded.valid_lengths[i];

      // Process results for each window size that was actually computed
      if (i < full_results.size()) {
        const auto& agent_results = full_results[i];

        for (size_t w = 0; w < agent_results.window_sizes_used.size(); w++) {
          int window_size = agent_results.window_sizes_used[w];

          AgentMatrixProfile::WindowResult window_result;
          window_result.window_size = window_size;
          window_result.distances = agent_results.window_profiles[w];
          window_result.indices = agent_results.window_indices[w];

          // Find top motifs
          window_result.top_motifs =
              Analysis::find_top_motifs(window_result.distances, window_result.indices, window_size);

          agent_profile.window_results.push_back(std::move(window_result));
        }
      }

      results.push_back(std::move(agent_profile));
    }

    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Log performance info
    std::cout << "MatrixProfiler CPU Performance:\n"
              << "  Total time: " << last_stats_.total_time_ms << " ms\n"
              << "  Total comparisons: " << last_stats_.total_comparisons << "\n"
              << "  Comparisons/second: " << last_stats_.comparisons_per_second << "\n";

    return results;
  }

  CrossAgentPatterns find_cross_agent_patterns(const std::vector<Agent*>& agents,
                                               uint8_t window_size,
                                               float distance_threshold) override {
    CrossAgentPatterns patterns;

    if (!lut_initialized_ || agents.size() < 2) {
      return patterns;
    }

    // Encode agent histories
    auto encoded = encode_agent_histories(agents, *action_lut_);

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

  void update_agent(const Agent* agent) override {
    // For CPU version, we don't maintain incremental state
    if (agent) {
      std::cout << "MatrixProfiler: Agent " << agent->agent_id << " updated (CPU mode - no incremental update)\n";
    }
  }

  void batch_update(const std::vector<Agent*>& updated_agents) override {
    // For CPU version, we don't maintain incremental state
    std::cout << "MatrixProfiler: Batch update of " << updated_agents.size()
              << " agents (CPU mode - no incremental update)\n";
  }

  MatrixProfiler::PerformanceStats get_last_performance_stats() const override {
    return last_stats_;
  }

  size_t get_gpu_memory_usage() const override {
    return 0;  // No GPU memory in CPU mode
  }

  void clear_cache() override {
    // Nothing to clear in CPU mode
    std::cout << "MatrixProfiler: Cache cleared (CPU mode)\n";
  }

private:
  void compute_profiles_internal(const std::vector<std::vector<uint8_t>>& sequences,
                                 const std::vector<size_t>& seq_lengths,
                                 const std::vector<uint8_t>& window_sizes,
                                 std::vector<std::vector<uint16_t>>& out_profiles,
                                 std::vector<std::vector<uint32_t>>& out_indices) {
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t num_agents = sequences.size();

    // Store all results per agent
    std::vector<AgentProfileResults> all_results(num_agents);

    // Track total comparisons
    size_t total_comparisons = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : total_comparisons)
#endif
    for (size_t agent_idx = 0; agent_idx < num_agents; agent_idx++) {
      const auto& seq = sequences[agent_idx];
      size_t seq_length = seq_lengths[agent_idx];

      // Process each window size
      for (uint8_t window_size : window_sizes) {
        if (window_size > static_cast<uint8_t>(seq_length)) continue;

        size_t profile_length = seq_length - static_cast<size_t>(window_size) + 1;
        std::vector<uint16_t> profile(profile_length, UINT16_MAX);
        std::vector<uint32_t> indices(profile_length, 0);

        // Compute matrix profile for this sequence and window size
        size_t local_comparisons = 0;
        compute_self_matrix_profile(
            seq.data(), seq_length, window_size, profile.data(), indices.data(), local_comparisons);

        total_comparisons += local_comparisons;

        // Store results for this window size
        all_results[agent_idx].window_profiles.push_back(std::move(profile));
        all_results[agent_idx].window_indices.push_back(std::move(indices));
        all_results[agent_idx].window_sizes_used.push_back(window_size);
      }
    }

    // For compatibility with existing interface, return results for the last valid window size
    out_profiles.resize(num_agents);
    out_indices.resize(num_agents);

    for (size_t i = 0; i < num_agents; i++) {
      if (!all_results[i].window_profiles.empty()) {
        out_profiles[i] = all_results[i].window_profiles.back();
        out_indices[i] = all_results[i].window_indices.back();
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Update performance stats
    last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    last_stats_.gpu_compute_time_ms = 0;      // No GPU
    last_stats_.memory_transfer_time_ms = 0;  // No memory transfer
    last_stats_.total_comparisons = total_comparisons;
    last_stats_.comparisons_per_second = (total_comparisons / last_stats_.total_time_ms) * 1000.0f;

    // Store the full results for later retrieval
    last_full_results_ = std::move(all_results);
  }

  void compute_self_matrix_profile(const uint8_t* sequence,
                                   size_t seq_length,
                                   uint8_t window_size,
                                   uint16_t* profile,
                                   uint32_t* indices,
                                   size_t& total_comparisons) {
    size_t profile_length = seq_length - static_cast<size_t>(window_size) + 1;
    uint8_t exclusion_zone = window_size / 2;

    // Initialize profile with max values
    for (size_t i = 0; i < profile_length; i++) {
      profile[i] = UINT16_MAX;
      indices[i] = static_cast<uint32_t>(i);
    }

    // Use STOMP-like approach with sliding window optimization
    for (size_t i = 0; i < profile_length; i++) {
      uint32_t min_distance = UINT32_MAX;
      uint32_t min_index = static_cast<uint32_t>(i);

      // Compare with all other subsequences
      for (size_t j = 0; j < profile_length; j++) {
        // Skip self-matches and exclusion zone
        if (std::abs(static_cast<int>(i) - static_cast<int>(j)) < exclusion_zone) {
          continue;
        }

        // Compute distance using the LUT
        uint32_t dist = compute_distance_cpu(sequence + i, sequence + j, window_size, distance_lut_);
        total_comparisons++;

        if (dist < min_distance) {
          min_distance = dist;
          min_index = static_cast<uint32_t>(j);
        }
      }

      // Store the minimum distance found
      profile[i] = static_cast<uint16_t>(std::min(min_distance, 65535u));
      indices[i] = min_index;
    }
  }

  MatrixProfiler::EncodedSequences encode_agent_histories(const std::vector<Agent*>& agents,
                                                          const ActionDistance::ActionDistanceLUT& distance_lut) const {
    MatrixProfiler::EncodedSequences encoded;
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

      // Encode to uint8 using the actual distance LUT encoding
      std::vector<uint8_t> encoded_seq(history_length);
      for (size_t i = 0; i < history_length; i++) {
        encoded_seq[i] = distance_lut.encode_action(actions[i], args[i]);
      }

      encoded.sequences.push_back(std::move(encoded_seq));
      encoded.valid_lengths.push_back(history_length);
      encoded.agent_ids.push_back(agent->agent_id);
    }

    return encoded;
  }
};

// Factory function implementation
std::unique_ptr<MatrixProfileImpl> create_matrix_profile_impl(const MatrixProfileConfig& config) {
#ifndef CUDA_DISABLED
  if (!config.force_cpu) {
    // Check for CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err == cudaSuccess && device_count > 0) {
      try {
        // Try to create GPU implementation
        extern std::unique_ptr<MatrixProfileImpl> create_matrix_profile_gpu(const MatrixProfileConfig& config);
        return create_matrix_profile_gpu(config);
      } catch (const std::exception& e) {
        std::cerr << "Failed to initialize GPU: " << e.what() << "\n";
        std::cerr << "Falling back to CPU implementation\n";
      }
    }
  }
#endif

  // Default to CPU implementation
  return std::make_unique<MatrixProfileCPU>(config);
}

// MatrixProfiler implementation
MatrixProfiler::MatrixProfiler(const MatrixProfileConfig& config) : config_(config) {
  impl_ = create_matrix_profile_impl(config);

  // Check if we're using GPU
#ifndef CUDA_DISABLED
  // If impl is not CPU, then it must be GPU
  using_gpu_ = (dynamic_cast<MatrixProfileCPU*>(impl_.get()) == nullptr);
#else
  using_gpu_ = false;
#endif
}

MatrixProfiler::~MatrixProfiler() = default;

void MatrixProfiler::initialize(const ActionDistance::ActionDistanceLUT& distance_lut) {
  impl_->initialize(distance_lut);
}

std::vector<AgentMatrixProfile> MatrixProfiler::compute_profiles(const std::vector<Agent*>& agents,
                                                                 const std::vector<uint8_t>& window_sizes) {
  return impl_->compute_profiles(agents, window_sizes);
}

CrossAgentPatterns MatrixProfiler::find_cross_agent_patterns(const std::vector<Agent*>& agents,
                                                             uint8_t window_size,
                                                             float distance_threshold) {
  return impl_->find_cross_agent_patterns(agents, window_size, distance_threshold);
}

void MatrixProfiler::update_agent(const Agent* agent) {
  impl_->update_agent(agent);
}

void MatrixProfiler::batch_update(const std::vector<Agent*>& updated_agents) {
  impl_->batch_update(updated_agents);
}

MatrixProfiler::PerformanceStats MatrixProfiler::get_last_performance_stats() const {
  return impl_->get_last_performance_stats();
}

size_t MatrixProfiler::get_gpu_memory_usage() const {
  return impl_->get_gpu_memory_usage();
}

void MatrixProfiler::clear_cache() {
  impl_->clear_cache();
}

// Implementation of Analysis functions
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
    motif.start_idx = static_cast<uint32_t>(min_idx);
    motif.match_idx = profile_indices[min_idx];
    motif.distance = min_dist;
    motif.length = window_size;
    motifs.push_back(motif);

    // Mark exclusion zone around this position
    for (int i = std::max(0, static_cast<int>(min_idx) - exclusion_zone);
         i < std::min(static_cast<int>(matrix_profile.size()), static_cast<int>(min_idx) + exclusion_zone + 1);
         i++) {
      used[static_cast<size_t>(i)] = true;
    }

    // Also mark around the match position if it's valid
    if (motif.match_idx < matrix_profile.size()) {
      for (int i = std::max(0, static_cast<int>(motif.match_idx) - exclusion_zone);
           i <
           std::min(static_cast<int>(matrix_profile.size()), static_cast<int>(motif.match_idx) + exclusion_zone + 1);
           i++) {
        used[static_cast<size_t>(i)] = true;
      }
    }
  }

  return motifs;
}

// Coarse similarity - Ultra fast, looks only at top motif statistics
// Complexity: O(1) - just comparing numbers
float compute_agent_similarity_coarse(const AgentMatrixProfile& profile1,
                                      const AgentMatrixProfile& profile2,
                                      uint8_t window_size) {
  // Find window results
  const AgentMatrixProfile::WindowResult* result1 = nullptr;
  const AgentMatrixProfile::WindowResult* result2 = nullptr;

  for (const auto& wr : profile1.window_results) {
    if (wr.window_size == window_size) {
      result1 = &wr;
      break;
    }
  }

  for (const auto& wr : profile2.window_results) {
    if (wr.window_size == window_size) {
      result2 = &wr;
      break;
    }
  }

  if (!result1 || !result2 || result1->top_motifs.empty() || result2->top_motifs.empty()) {
    return 0.0f;
  }

  // Compare only the top motif distances
  const auto& motif1 = result1->top_motifs[0];
  const auto& motif2 = result2->top_motifs[0];

  // Similarity based on how close their best motif distances are
  float dist_diff = std::abs(static_cast<float>(motif1.distance) - static_cast<float>(motif2.distance));
  float dist_similarity = 1.0f - (dist_diff / 15.0f);  // Normalize by max distance

  // Also consider if they have similar number of good motifs
  float motif_count_sim =
      1.0f -
      std::abs(static_cast<float>(result1->top_motifs.size()) - static_cast<float>(result2->top_motifs.size())) / 10.0f;

  // Quick profile statistics comparison
  float profile_mean1 = 0.0f, profile_mean2 = 0.0f;
  if (!result1->distances.empty() && !result2->distances.empty()) {
    // Sample first 100 distances for speed
    size_t sample_size = std::min(size_t(100), std::min(result1->distances.size(), result2->distances.size()));

    for (size_t i = 0; i < sample_size; i++) {
      profile_mean1 += result1->distances[i];
      profile_mean2 += result2->distances[i];
    }
    profile_mean1 /= sample_size;
    profile_mean2 /= sample_size;

    float mean_diff = std::abs(profile_mean1 - profile_mean2);
    float mean_similarity = 1.0f - (mean_diff / 255.0f);

    // Weighted combination
    return 0.5f * dist_similarity + 0.3f * motif_count_sim + 0.2f * mean_similarity;
  }

  return 0.7f * dist_similarity + 0.3f * motif_count_sim;
}

// Fine similarity - Full analysis, compares actual patterns
// This is the original compute_agent_similarity with minor optimizations
float compute_agent_similarity_fine(const AgentMatrixProfile& profile1,
                                    const AgentMatrixProfile& profile2,
                                    uint8_t window_size,
                                    const Agent* agent1,
                                    const Agent* agent2,
                                    const ActionDistance::ActionDistanceLUT& action_lut) {
  // Find window results
  const AgentMatrixProfile::WindowResult* result1 = nullptr;
  const AgentMatrixProfile::WindowResult* result2 = nullptr;

  for (const auto& wr : profile1.window_results) {
    if (wr.window_size == window_size) {
      result1 = &wr;
      break;
    }
  }

  for (const auto& wr : profile2.window_results) {
    if (wr.window_size == window_size) {
      result2 = &wr;
      break;
    }
  }

  if (!result1 || !result2) {
    return 0.0f;
  }

  const auto& motifs1 = result1->top_motifs;
  const auto& motifs2 = result2->top_motifs;

  if (motifs1.empty() || motifs2.empty()) {
    return 0.0f;
  }

  // Get agent histories
  size_t history1_len = agent1->history_count;
  size_t history2_len = agent2->history_count;

  if (history1_len < static_cast<size_t>(window_size) || history2_len < static_cast<size_t>(window_size)) {
    return 0.0f;
  }

  // Extract and encode histories
  std::vector<ActionType> actions1(history1_len), actions2(history2_len);
  std::vector<ActionArg> args1(history1_len), args2(history2_len);

  agent1->copy_history_to_buffers(actions1.data(), args1.data());
  agent2->copy_history_to_buffers(actions2.data(), args2.data());

  std::vector<uint8_t> encoded1 = action_lut.encode_sequence(actions1, args1);
  std::vector<uint8_t> encoded2 = action_lut.encode_sequence(actions2, args2);

  // Compute pairwise distances between motifs
  struct MotifMatch {
    size_t idx1;
    size_t idx2;
    float distance;
    float rank_weight;
  };

  std::vector<MotifMatch> matches;
  matches.reserve(motifs1.size() * motifs2.size());

  for (size_t i = 0; i < motifs1.size(); i++) {
    for (size_t j = 0; j < motifs2.size(); j++) {
      uint32_t start1 = motifs1[i].start_idx;
      uint32_t start2 = motifs2[j].start_idx;

      if (start1 + window_size > encoded1.size() || start2 + window_size > encoded2.size()) {
        continue;
      }

      // Extract subsequences
      std::vector<uint8_t> subseq1(encoded1.begin() + start1, encoded1.begin() + start1 + window_size);
      std::vector<uint8_t> subseq2(encoded2.begin() + start2, encoded2.begin() + start2 + window_size);

      // Use ActionDistance sequence_distance
      uint32_t total_dist = action_lut.sequence_distance(subseq1, subseq2);
      float normalized_dist = static_cast<float>(total_dist) / window_size;

      // Rank weight: more important motifs get higher weight
      float rank_weight = 1.0f / (1.0f + std::sqrt(static_cast<float>(i * j)));

      matches.push_back({i, j, normalized_dist, rank_weight});
    }
  }

  if (matches.empty()) {
    return 0.0f;
  }

  // Sort by distance for greedy matching
  std::sort(
      matches.begin(), matches.end(), [](const MotifMatch& a, const MotifMatch& b) { return a.distance < b.distance; });

  // Greedy matching
  std::vector<bool> used1(motifs1.size(), false);
  std::vector<bool> used2(motifs2.size(), false);

  float total_weight = 0.0f;
  float weighted_similarity = 0.0f;

  for (const auto& match : matches) {
    if (used1[match.idx1] || used2[match.idx2]) continue;

    used1[match.idx1] = true;
    used2[match.idx2] = true;

    float similarity = 1.0f - (match.distance / 15.0f);
    weighted_similarity += match.rank_weight * similarity;
    total_weight += match.rank_weight;
  }

  // Coverage
  float coverage =
      static_cast<float>(std::count(used1.begin(), used1.end(), true) + std::count(used2.begin(), used2.end(), true)) /
      (motifs1.size() + motifs2.size());

  float match_quality = (total_weight > 0) ? (weighted_similarity / total_weight) : 0.0f;
  return 0.7f * match_quality + 0.3f * coverage;
}

// find behavioral patterns using ActionDistance
std::vector<std::string> describe_motif_patterns(const AgentMatrixProfile& profile,
                                                 const Agent* agent,
                                                 const ActionDistance::ActionDistanceLUT& action_lut,
                                                 uint8_t window_size) {
  std::vector<std::string> descriptions;

  // Find the window result
  const AgentMatrixProfile::WindowResult* result = nullptr;
  for (const auto& wr : profile.window_results) {
    if (wr.window_size == window_size) {
      result = &wr;
      break;
    }
  }

  if (!result || result->top_motifs.empty()) {
    return descriptions;
  }

  // Get agent history
  size_t history_len = agent->history_count;
  if (history_len < static_cast<size_t>(window_size)) {
    return descriptions;
  }

  std::vector<ActionType> actions(history_len);
  std::vector<ActionArg> args(history_len);
  agent->copy_history_to_buffers(actions.data(), args.data());

  // Encode the full sequence
  std::vector<uint8_t> encoded = action_lut.encode_sequence(actions, args);

  // Describe each top motif
  for (const auto& motif : result->top_motifs) {
    if (motif.start_idx + window_size <= encoded.size()) {
      // Extract the motif subsequence
      std::vector<uint8_t> motif_seq(encoded.begin() + motif.start_idx,
                                     encoded.begin() + motif.start_idx + window_size);

      // Use ActionDistance to decode the pattern to human-readable form
      std::string pattern_desc = action_lut.decode_sequence_to_string(motif_seq);

      // Add similarity info if there's a match
      std::string match_info = "";
      if (motif.match_idx < encoded.size() - window_size) {
        std::vector<uint8_t> match_seq(encoded.begin() + motif.match_idx,
                                       encoded.begin() + motif.match_idx + window_size);

        // Check if patterns are similar using ActionDistance threshold
        bool similar = action_lut.patterns_similar(motif_seq, match_seq, motif.distance);

        if (similar) {
          match_info = " [Similar pattern at position " + std::to_string(motif.match_idx) + " with distance " +
                       std::to_string(motif.distance) + "]";
        }
      }

      descriptions.push_back("Motif@" + std::to_string(motif.start_idx) + ": " + pattern_desc + match_info);
    }
  }

  return descriptions;
}

// Balanced k-medoids clustering with adaptive coarse/fine similarity computation
std::vector<CrossAgentPatterns::BehaviorCluster> cluster_by_behavior(
    const std::vector<AgentMatrixProfile>& profiles,
    const std::vector<Agent*>& agents,
    const ActionDistance::ActionDistanceLUT& action_lut,
    int window_size,
    int num_clusters = 0,
    float fine_similarity_budget = 0.1f) {  // What fraction of comparisons to do with fine similarity

  if (profiles.empty() || agents.size() != profiles.size() || window_size <= 0) {
    return {};
  }

  size_t n_agents = profiles.size();

  // Auto-determine number of clusters if not specified
  if (num_clusters == 0) {
    num_clusters = std::max(2, static_cast<int>(std::sqrt(n_agents / 2.0)));
    num_clusters = std::min(num_clusters, std::min(30, static_cast<int>(n_agents / 4)));
  }
  num_clusters = std::min(num_clusters, static_cast<int>(n_agents));

  // Calculate fine similarity budget
  size_t total_possible_comparisons = n_agents * (n_agents - 1) / 2;
  size_t fine_similarity_budget_count = static_cast<size_t>(total_possible_comparisons * fine_similarity_budget);
  size_t fine_similarities_used = 0;

  std::cout << "Clustering " << n_agents << " agents into " << num_clusters << " clusters\n";
  std::cout << "Fine similarity budget: " << fine_similarity_budget_count << " comparisons\n";

  // Similarity cache to avoid recomputation
  struct SimilarityCache {
    std::map<std::pair<size_t, size_t>, float> coarse_cache;
    std::map<std::pair<size_t, size_t>, float> fine_cache;

    float get_similarity(size_t i,
                         size_t j,
                         const std::vector<AgentMatrixProfile>& profiles,
                         const std::vector<Agent*>& agents,
                         const ActionDistance::ActionDistanceLUT& lut,
                         int window_size,
                         bool use_fine,
                         size_t& fine_count) {
      if (i > j) std::swap(i, j);
      auto key = std::make_pair(i, j);

      // Check fine cache first
      auto fine_it = fine_cache.find(key);
      if (fine_it != fine_cache.end()) {
        return fine_it->second;
      }

      // Check coarse cache
      auto coarse_it = coarse_cache.find(key);
      if (coarse_it != coarse_cache.end() && !use_fine) {
        return coarse_it->second;
      }

      // Compute similarity
      float sim;
      if (use_fine && agents[i] && agents[j]) {
        sim = compute_agent_similarity_fine(profiles[i], profiles[j], window_size, agents[i], agents[j], lut);
        fine_cache[key] = sim;
        fine_count++;
      } else {
        sim = compute_agent_similarity_coarse(profiles[i], profiles[j], window_size);
        coarse_cache[key] = sim;
      }

      return sim;
    }
  };

  SimilarityCache sim_cache;

  // Step 1: Initialize medoids using k-means++ with smart similarity selection
  std::vector<size_t> medoids;
  std::vector<bool> is_medoid(n_agents, false);
  std::mt19937 rng(42);

  // First medoid: randomly selected
  std::uniform_int_distribution<size_t> dist(0, n_agents - 1);
  medoids.push_back(dist(rng));
  is_medoid[medoids[0]] = true;

  // Remaining medoids: k-means++ strategy
  for (int k = 1; k < num_clusters; k++) {
    std::vector<float> min_distances(n_agents, 0.0f);

// Calculate distance to nearest medoid
#pragma omp parallel for
    for (size_t i = 0; i < n_agents; i++) {
      if (is_medoid[i]) continue;

      float max_sim = 0.0f;
      for (size_t med_idx : medoids) {
        // Use coarse similarity for initialization
        float sim = sim_cache.get_similarity(
            i, med_idx, profiles, agents, action_lut, window_size, false, fine_similarities_used);
        max_sim = std::max(max_sim, sim);
      }

      min_distances[i] = 1.0f - max_sim;
    }

    // Choose next medoid probabilistically
    std::discrete_distribution<size_t> weighted_dist(min_distances.begin(), min_distances.end());
    size_t next_medoid = weighted_dist(rng);
    medoids.push_back(next_medoid);
    is_medoid[next_medoid] = true;
  }

  // Step 2: Initial assignment using coarse similarity
  std::vector<int> assignments(n_agents, -1);
  std::vector<float> assignment_scores(n_agents, 0.0f);

  auto assign_agents = [&](bool use_fine_for_uncertain = false) {
#pragma omp parallel for
    for (size_t i = 0; i < n_agents; i++) {
      float best_sim = -1.0f;
      int best_cluster = -1;
      float second_best_sim = -1.0f;

      for (size_t k = 0; k < medoids.size(); k++) {
        if (i == medoids[k]) {
          best_sim = 1.0f;
          best_cluster = static_cast<int>(k);
          break;
        }

        // Decide whether to use fine similarity
        bool use_fine = false;
        if (use_fine_for_uncertain && fine_similarities_used < fine_similarity_budget_count) {
          // Use fine similarity if previous assignment was uncertain
          use_fine = (assignment_scores[i] < 0.6f && assignment_scores[i] > 0.0f);
        }

        float sim = sim_cache.get_similarity(
            i, medoids[k], profiles, agents, action_lut, window_size, use_fine, fine_similarities_used);

        if (sim > best_sim) {
          second_best_sim = best_sim;
          best_sim = sim;
          best_cluster = static_cast<int>(k);
        } else if (sim > second_best_sim) {
          second_best_sim = sim;
        }
      }

      assignments[i] = best_cluster;
      assignment_scores[i] = best_sim;
    }
  };

  // Initial assignment with coarse similarity
  assign_agents(false);

  // Step 3: Iterative optimization with adaptive similarity
  const int MAX_ITERATIONS = 10;

  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
    bool changed = false;

    // Update medoids
    for (size_t k = 0; k < medoids.size(); k++) {
      // Get cluster members
      std::vector<size_t> members;
      for (size_t i = 0; i < n_agents; i++) {
        if (assignments[i] == static_cast<int>(k)) {
          members.push_back(i);
        }
      }

      if (members.empty()) continue;

      // Find best medoid within cluster
      size_t best_medoid = medoids[k];
      float best_total_sim = -1.0f;

      // Only check a subset of members as potential medoids
      size_t candidates_to_check = std::min(members.size(), size_t(5));

      // Always check current medoid
      float current_total_sim = 0.0f;
      for (size_t member : members) {
        if (member != medoids[k]) {
          current_total_sim += sim_cache.get_similarity(
              medoids[k], member, profiles, agents, action_lut, window_size, false, fine_similarities_used);
        }
      }
      best_total_sim = current_total_sim;

      // Check random candidates
      std::shuffle(members.begin(), members.end(), rng);

      for (size_t c = 0; c < candidates_to_check; c++) {
        size_t candidate = members[c];
        if (candidate == medoids[k]) continue;

        float total_sim = 0.0f;

        // Use coarse similarity for most comparisons
        for (size_t member : members) {
          if (member != candidate) {
            // Use fine similarity for small clusters or if we have budget
            bool use_fine = (members.size() < 10 && fine_similarities_used < fine_similarity_budget_count);

            total_sim += sim_cache.get_similarity(
                candidate, member, profiles, agents, action_lut, window_size, use_fine, fine_similarities_used);
          }
        }

        if (total_sim > best_total_sim) {
          best_total_sim = total_sim;
          best_medoid = candidate;
        }
      }

      if (best_medoid != medoids[k]) {
        medoids[k] = best_medoid;
        changed = true;
      }
    }

    if (!changed) break;

    // Reassign with option to use fine similarity for uncertain cases
    assign_agents(iter > 2);  // Start using fine similarity after iteration 2
  }

  // Step 4: Build final clusters
  std::vector<CrossAgentPatterns::BehaviorCluster> clusters(num_clusters);

  for (size_t i = 0; i < n_agents; i++) {
    int cluster_id = assignments[i];
    if (cluster_id >= 0 && cluster_id < num_clusters) {
      clusters[cluster_id].agent_ids.push_back(profiles[i].agent_id);
    }
  }

  // Set representative sequences and compute statistics
  for (size_t k = 0; k < clusters.size(); k++) {
    if (clusters[k].agent_ids.empty()) continue;

    size_t medoid_idx = medoids[k];

    // Extract representative sequence from medoid's top motif
    const AgentMatrixProfile::WindowResult* result = nullptr;
    for (const auto& wr : profiles[medoid_idx].window_results) {
      if (wr.window_size == window_size) {
        result = &wr;
        break;
      }
    }

    if (result && !result->top_motifs.empty() &&
        agents[medoid_idx]->history_count >= static_cast<size_t>(window_size)) {
      const auto& top_motif = result->top_motifs[0];

      std::vector<ActionType> actions(agents[medoid_idx]->history_count);
      std::vector<ActionArg> args(agents[medoid_idx]->history_count);
      agents[medoid_idx]->copy_history_to_buffers(actions.data(), args.data());

      clusters[k].representative_sequence.clear();
      for (int i = 0; i < window_size; i++) {
        size_t idx = top_motif.start_idx + i;
        if (idx < actions.size()) {
          clusters[k].representative_sequence.push_back(action_lut.encode_action(actions[idx], args[idx]));
        }
      }
    }

    // Compute intra-cluster distance
    float total_dist = 0.0f;
    int count = 0;

    for (size_t i = 0; i < clusters[k].agent_ids.size(); i++) {
      for (size_t j = i + 1; j < clusters[k].agent_ids.size(); j++) {
        // Find indices
        size_t idx_i = 0, idx_j = 0;
        for (size_t a = 0; a < profiles.size(); a++) {
          if (profiles[a].agent_id == clusters[k].agent_ids[i]) idx_i = a;
          if (profiles[a].agent_id == clusters[k].agent_ids[j]) idx_j = a;
        }

        float sim = sim_cache.get_similarity(
            idx_i, idx_j, profiles, agents, action_lut, window_size, false, fine_similarities_used);

        total_dist += (1.0f - sim);
        count++;
      }
    }

    clusters[k].avg_intra_cluster_distance = count > 0 ? total_dist / count : 0.0f;
  }

  // Remove empty clusters
  clusters.erase(std::remove_if(clusters.begin(),
                                clusters.end(),
                                [](const CrossAgentPatterns::BehaviorCluster& c) { return c.agent_ids.empty(); }),
                 clusters.end());

  // Report statistics
  std::cout << "\nClustering complete:\n";
  std::cout << "  Clusters formed: " << clusters.size() << "\n";
  std::cout << "  Fine similarities computed: " << fine_similarities_used << " ("
            << (100.0f * fine_similarities_used / fine_similarity_budget_count) << "% of budget)\n";
  std::cout << "  Coarse similarities computed: " << sim_cache.coarse_cache.size() << "\n";

  for (size_t i = 0; i < clusters.size(); i++) {
    std::cout << "  Cluster " << i << ": " << clusters[i].agent_ids.size()
              << " agents (avg dist: " << clusters[i].avg_intra_cluster_distance << ")\n";
  }

  return clusters;
}

}  // namespace Analysis

}  // namespace MatrixProfile
