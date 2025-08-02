// matrix_profile.cu - GPU-accelerated Matrix Profile implementation
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>

#include "matrix_profile.hpp"

namespace cg = cooperative_groups;

namespace MatrixProfile {

// Constants for GPU computation
constexpr int WARP_SIZE = 32;
constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB per SM
constexpr int MAX_WINDOW_SIZE = 256;

// Distance LUT in constant memory for fast access
__constant__ uint8_t d_distance_lut[256][256];

// Vectorized distance computation using uint32 loads
__device__ __forceinline__ uint32_t compute_distance_vectorized(const uint8_t* seq1, const uint8_t* seq2, int length) {
  uint32_t total_dist = 0;

  // Process 4 elements at a time
  int vec_length = length / 4;
  const uint32_t* vec1 = reinterpret_cast<const uint32_t*>(seq1);
  const uint32_t* vec2 = reinterpret_cast<const uint32_t*>(seq2);

  for (int i = 0; i < vec_length; i++) {
    uint32_t v1 = vec1[i];
    uint32_t v2 = vec2[i];

    // Extract individual bytes and accumulate distances
    total_dist += d_distance_lut[(v1 & 0xFF)][(v2 & 0xFF)];
    total_dist += d_distance_lut[(v1 >> 8) & 0xFF][(v2 >> 8) & 0xFF];
    total_dist += d_distance_lut[(v1 >> 16) & 0xFF][(v2 >> 16) & 0xFF];
    total_dist += d_distance_lut[(v1 >> 24) & 0xFF][(v2 >> 24) & 0xFF];
  }

  // Handle remaining elements
  for (int i = vec_length * 4; i < length; i++) {
    total_dist += d_distance_lut[seq1[i]][seq2[i]];
  }

  return total_dist;
}

// Matrix Profile kernel for multiple agents
template <int BLOCK_SIZE = 256>
__global__ void matrix_profile_multi_agent(const uint8_t* __restrict__ sequences,
                                           const int* __restrict__ seq_lengths,
                                           uint16_t* __restrict__ profiles,
                                           uint32_t* __restrict__ indices,
                                           int num_agents,
                                           int max_seq_length,
                                           int window_size) {
  extern __shared__ uint8_t shared_mem[];

  // Grid-stride loop for agent assignment
  int agent_idx = blockIdx.y;
  if (agent_idx >= num_agents) return;

  int seq_length = seq_lengths[agent_idx];
  int profile_length = seq_length - window_size + 1;
  if (profile_length <= 0) return;

  // Pointers to this agent's data
  const uint8_t* agent_seq = sequences + agent_idx * max_seq_length;
  uint16_t* agent_profile = profiles + agent_idx * max_seq_length;
  uint32_t* agent_indices = indices + agent_idx * max_seq_length;

  // Load query window into shared memory
  uint8_t* query_window = shared_mem;

  // Each thread block processes one query position
  int query_idx = blockIdx.x;
  if (query_idx >= profile_length) return;

  // Cooperative loading of query window
  for (int i = threadIdx.x; i < window_size; i += BLOCK_SIZE) {
    if (i < window_size) {
      query_window[i] = agent_seq[query_idx + i];
    }
  }
  __syncthreads();

  // Initialize minimum distance for this query
  uint32_t min_distance = UINT32_MAX;
  uint32_t min_index = 0;

  // Grid-stride loop through all possible matches
  for (int match_idx = threadIdx.x; match_idx < profile_length; match_idx += BLOCK_SIZE) {
    // Skip self-matches and exclusion zone (half window size)
    int exclusion_zone = window_size / 2;
    if (abs(match_idx - query_idx) < exclusion_zone) continue;

    // Compute distance
    uint32_t dist = compute_distance_vectorized(query_window, agent_seq + match_idx, window_size);

    // Track minimum
    if (dist < min_distance) {
      min_distance = dist;
      min_index = match_idx;
    }
  }

  // Warp-level reduction to find minimum across threads
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

  for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
    uint32_t other_dist = warp.shfl_down(min_distance, offset);
    uint32_t other_idx = warp.shfl_down(min_index, offset);

    if (other_dist < min_distance) {
      min_distance = other_dist;
      min_index = other_idx;
    }
  }

  // First thread in each warp writes to shared memory
  __shared__ uint32_t warp_min_dists[BLOCK_SIZE / 32];
  __shared__ uint32_t warp_min_indices[BLOCK_SIZE / 32];

  if (warp.thread_rank() == 0) {
    warp_min_dists[warp.meta_group_rank()] = min_distance;
    warp_min_indices[warp.meta_group_rank()] = min_index;
  }
  __syncthreads();

  // Final reduction by first warp
  if (threadIdx.x < warp.size()) {
    min_distance = (threadIdx.x < BLOCK_SIZE / 32) ? warp_min_dists[threadIdx.x] : UINT32_MAX;
    min_index = (threadIdx.x < BLOCK_SIZE / 32) ? warp_min_indices[threadIdx.x] : 0;

    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
      uint32_t other_dist = warp.shfl_down(min_distance, offset);
      uint32_t other_idx = warp.shfl_down(min_index, offset);

      if (other_dist < min_distance) {
        min_distance = other_dist;
        min_index = other_idx;
      }
    }

    // Thread 0 writes final result
    if (threadIdx.x == 0) {
      agent_profile[query_idx] = static_cast<uint16_t>(min(min_distance, 65535u));
      agent_indices[query_idx] = min_index;
    }
  }
}

// GPU implementation class
class MatrixProfileGPU : public MatrixProfileImpl {
private:
  struct GPUData {
    uint8_t* d_sequences = nullptr;
    int* d_seq_lengths = nullptr;
    uint16_t* d_profiles = nullptr;
    uint32_t* d_indices = nullptr;

    size_t max_agents = 0;
    size_t max_seq_length = 0;

    cudaStream_t stream;

    ~GPUData() {
      if (d_sequences) cudaFree(d_sequences);
      if (d_seq_lengths) cudaFree(d_seq_lengths);
      if (d_profiles) cudaFree(d_profiles);
      if (d_indices) cudaFree(d_indices);
      cudaStreamDestroy(stream);
    }
  };

  MatrixProfileConfig config_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_;
  std::vector<int> gpu_devices_;
  std::array<std::array<uint8_t, 256>, 256> distance_lut_;
  bool lut_initialized_ = false;
  mutable MatrixProfiler::PerformanceStats last_stats_;

public:
  MatrixProfileGPU(const MatrixProfileConfig& config) : config_(config) {
    // Initialize GPU devices
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (config.use_multi_gpu && device_count > 1) {
      // Use all available GPUs
      for (int i = 0; i < device_count; i++) {
        gpu_devices_.push_back(i);
      }
    } else {
      // Single GPU mode
      gpu_devices_.push_back(config.gpu_device_id);
    }

    // Allocate GPU data structures
    for (int device : gpu_devices_) {
      cudaSetDevice(device);
      auto data = std::make_unique<GPUData>();
      cudaStreamCreate(&data->stream);
      gpu_data_.push_back(std::move(data));
    }

    std::cout << "MatrixProfiler GPU implementation initialized\n";
    std::cout << "  Available GPUs: " << device_count << "\n";

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config.gpu_device_id);
    std::cout << "  Using GPU: " << prop.name << "\n";
    std::cout << "  GPU Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
  }

  void initialize(const ActionDistance::ActionDistanceLUT& distance_lut) override {
    // Get the distance table
    uint8_t lut[256][256];
    distance_lut.get_distance_table(lut);

    // Store locally
    std::memcpy(distance_lut_.data(), lut, sizeof(lut));

    // Upload to constant memory on all devices
    for (int device : gpu_devices_) {
      cudaSetDevice(device);
      cudaMemcpyToSymbol(d_distance_lut, lut, sizeof(d_distance_lut));
    }

    lut_initialized_ = true;
    std::cout << "MatrixProfiler GPU: Distance LUT uploaded to constant memory\n";
  }

  std::vector<AgentMatrixProfile> compute_profiles(const std::vector<Agent*>& agents,
                                                   const std::vector<int>& window_sizes,
                                                   const ActionDistance::ActionDistanceLUT& distance_lut) override {
    if (!lut_initialized_) {
      throw std::runtime_error("MatrixProfiler not initialized. Call initialize() first.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Encode agent histories
    auto encoded = encode_agent_histories(agents, distance_lut);
    if (encoded.sequences.empty()) {
      return {};
    }

    // Use provided window sizes or defaults
    const auto& windows = window_sizes.empty() ? config_.window_sizes : window_sizes;

    // TODO: Implement full GPU computation
    // For now, this is a placeholder that shows the structure
    std::vector<AgentMatrixProfile> results;

    // Log performance info
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    last_stats_.gpu_compute_time_ms = last_stats_.total_time_ms;

    std::cout << "MatrixProfiler GPU Performance:\n"
              << "  Total time: " << last_stats_.total_time_ms << " ms\n";

    return results;
  }

  CrossAgentPatterns find_cross_agent_patterns(const std::vector<Agent*>& agents,
                                               int window_size,
                                               float distance_threshold,
                                               const ActionDistance::ActionDistanceLUT& distance_lut) override {
    // TODO: Implement GPU version
    CrossAgentPatterns patterns;
    return patterns;
  }

  void update_agent(const Agent* agent) override {
    if (agent) {
      std::cout << "MatrixProfiler: Agent " << agent->agent_id << " updated (GPU mode)\n";
    }
  }

  void batch_update(const std::vector<Agent*>& updated_agents) override {
    std::cout << "MatrixProfiler: Batch update of " << updated_agents.size() << " agents (GPU mode)\n";
  }

  MatrixProfiler::PerformanceStats get_last_performance_stats() const override {
    return last_stats_;
  }

  size_t get_gpu_memory_usage() const override {
    size_t total_memory = 0;
    for (const auto& data : gpu_data_) {
      if (data->d_sequences) {
        total_memory += data->max_agents * data->max_seq_length * sizeof(uint8_t);
        total_memory += data->max_agents * sizeof(int);
        total_memory += data->max_agents * data->max_seq_length * sizeof(uint16_t);
        total_memory += data->max_agents * data->max_seq_length * sizeof(uint32_t);
      }
    }
    return total_memory;
  }

  void clear_cache() override {
    for (int device : gpu_devices_) {
      cudaSetDevice(device);
      cudaDeviceSynchronize();
    }
    std::cout << "MatrixProfiler: GPU cache synchronized\n";
  }

private:
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

// Factory function for GPU implementation
std::unique_ptr<MatrixProfileImpl> create_matrix_profile_gpu(const MatrixProfileConfig& config) {
  return std::make_unique<MatrixProfileGPU>(config);
}

}  // namespace MatrixProfile
