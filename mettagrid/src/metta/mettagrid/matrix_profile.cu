// matrix_profile.cu - GPU-accelerated Matrix Profile implementation

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda/atomic>
#include <memory>

#include "matrix_profile.hpp"

namespace cg = cooperative_groups;

namespace MatrixProfile {

// Constants for GPU computation
constexpr int WARP_SIZE = 32;
constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB per SM
constexpr int MAX_WINDOW_SIZE = 256;
constexpr int DISTANCE_LUT_SIZE = 256 * 256;

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
__global__ void matrix_profile_multi_agent(const uint8_t* __restrict__ sequences,  // [num_agents, max_seq_length]
                                           const int* __restrict__ seq_lengths,    // [num_agents]
                                           uint16_t* __restrict__ profiles,        // [num_agents, max_seq_length]
                                           uint32_t* __restrict__ indices,         // [num_agents, max_seq_length]
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

// Cross-agent pattern matching kernel
__global__ void cross_agent_pattern_match(
    const uint8_t* __restrict__ sequences,
    const int* __restrict__ seq_lengths,
    const uint16_t* __restrict__ profiles,
    uint8_t* __restrict__ pattern_matches,  // [num_agents, num_agents, max_matches]
    int num_agents,
    int max_seq_length,
    int window_size,
    uint16_t distance_threshold) {
  int agent1_idx = blockIdx.x;
  int agent2_idx = blockIdx.y;

  if (agent1_idx >= num_agents || agent2_idx >= num_agents || agent1_idx >= agent2_idx) {
    return;
  }

  // Implementation continues...
}

// CUDA implementation class
class MatrixProfileGPU {
public:
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

  std::vector<std::unique_ptr<GPUData>> gpu_data_;
  std::vector<int> gpu_devices_;
  MatrixProfileConfig config_;

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
  }

  void upload_distance_lut(const uint8_t lut[256][256]) {
    // Upload to constant memory on all devices
    for (int device : gpu_devices_) {
      cudaSetDevice(device);
      cudaMemcpyToSymbol(d_distance_lut, lut, sizeof(d_distance_lut));
    }
  }

  void allocate_memory(size_t num_agents, size_t max_seq_length) {
    size_t agents_per_gpu = (num_agents + gpu_devices_.size() - 1) / gpu_devices_.size();

    for (size_t i = 0; i < gpu_data_.size(); i++) {
      cudaSetDevice(gpu_devices_[i]);
      auto& data = gpu_data_[i];

      // Free old allocations if needed
      if (data->d_sequences) cudaFree(data->d_sequences);
      if (data->d_seq_lengths) cudaFree(data->d_seq_lengths);
      if (data->d_profiles) cudaFree(data->d_profiles);
      if (data->d_indices) cudaFree(data->d_indices);

      // Allocate new memory
      data->max_agents = agents_per_gpu;
      data->max_seq_length = max_seq_length;

      size_t seq_size = agents_per_gpu * max_seq_length * sizeof(uint8_t);
      size_t len_size = agents_per_gpu * sizeof(int);
      size_t profile_size = agents_per_gpu * max_seq_length * sizeof(uint16_t);
      size_t index_size = agents_per_gpu * max_seq_length * sizeof(uint32_t);

      cudaMalloc(&data->d_sequences, seq_size);
      cudaMalloc(&data->d_seq_lengths, len_size);
      cudaMalloc(&data->d_profiles, profile_size);
      cudaMalloc(&data->d_indices, index_size);
    }
  }

  void compute_profiles(const std::vector<std::vector<uint8_t>>& sequences,
                        const std::vector<size_t>& seq_lengths,
                        const std::vector<int>& window_sizes,
                        std::vector<std::vector<uint16_t>>& out_profiles,
                        std::vector<std::vector<uint32_t>>& out_indices) {
    size_t num_agents = sequences.size();
    size_t max_seq_length = 0;
    for (const auto& seq : sequences) {
      max_seq_length = std::max(max_seq_length, seq.size());
    }

    // Ensure memory is allocated
    allocate_memory(num_agents, max_seq_length);

// Distribute agents across GPUs
#pragma omp parallel for num_threads(gpu_devices_.size())
    for (size_t gpu_idx = 0; gpu_idx < gpu_devices_.size(); gpu_idx++) {
      cudaSetDevice(gpu_devices_[gpu_idx]);
      auto& data = gpu_data_[gpu_idx];

      // Calculate agent range for this GPU
      size_t agents_per_gpu = (num_agents + gpu_devices_.size() - 1) / gpu_devices_.size();
      size_t start_agent = gpu_idx * agents_per_gpu;
      size_t end_agent = std::min(start_agent + agents_per_gpu, num_agents);
      size_t gpu_num_agents = end_agent - start_agent;

      if (gpu_num_agents == 0) continue;

      // Prepare host data
      std::vector<uint8_t> h_sequences(gpu_num_agents * max_seq_length, 0);
      std::vector<int> h_seq_lengths(gpu_num_agents);

      for (size_t i = 0; i < gpu_num_agents; i++) {
        size_t agent_idx = start_agent + i;
        h_seq_lengths[i] = static_cast<int>(seq_lengths[agent_idx]);
        std::memcpy(&h_sequences[i * max_seq_length], sequences[agent_idx].data(), sequences[agent_idx].size());
      }

      // Upload to GPU
      cudaMemcpyAsync(data->d_sequences,
                      h_sequences.data(),
                      h_sequences.size() * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      data->stream);
      cudaMemcpyAsync(data->d_seq_lengths,
                      h_seq_lengths.data(),
                      h_seq_lengths.size() * sizeof(int),
                      cudaMemcpyHostToDevice,
                      data->stream);

      // Process each window size
      for (int window_size : window_sizes) {
        // Calculate grid dimensions
        int max_profile_length = max_seq_length - window_size + 1;
        dim3 grid(max_profile_length, gpu_num_agents);
        dim3 block(config_.block_size);

        size_t shared_mem_size = window_size * sizeof(uint8_t) + (config_.block_size / 32) * 2 * sizeof(uint32_t);

        // Launch kernel
        matrix_profile_multi_agent<256><<<grid, block, shared_mem_size, data->stream>>>(data->d_sequences,
                                                                                        data->d_seq_lengths,
                                                                                        data->d_profiles,
                                                                                        data->d_indices,
                                                                                        gpu_num_agents,
                                                                                        max_seq_length,
                                                                                        window_size);
      }

      // Copy results back asynchronously
      std::vector<uint16_t> h_profiles(gpu_num_agents * max_seq_length);
      std::vector<uint32_t> h_indices(gpu_num_agents * max_seq_length);

      cudaMemcpyAsync(h_profiles.data(),
                      data->d_profiles,
                      h_profiles.size() * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost,
                      data->stream);
      cudaMemcpyAsync(
          h_indices.data(), data->d_indices, h_indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, data->stream);

      // Synchronize this GPU
      cudaStreamSynchronize(data->stream);

      // Copy to output structures
      for (size_t i = 0; i < gpu_num_agents; i++) {
        size_t agent_idx = start_agent + i;
        out_profiles[agent_idx].assign(h_profiles.begin() + i * max_seq_length,
                                       h_profiles.begin() + i * max_seq_length + seq_lengths[agent_idx]);
        out_indices[agent_idx].assign(h_indices.begin() + i * max_seq_length,
                                      h_indices.begin() + i * max_seq_length + seq_lengths[agent_idx]);
      }
    }
  }
};

// MatrixProfiler implementation
MatrixProfiler::MatrixProfiler(const MatrixProfileConfig& config) : config_(config) {
  gpu_impl_ = std::make_unique<MatrixProfileGPU>(config);
}

MatrixProfiler::~MatrixProfiler() = default;

void MatrixProfiler::initialize(const ActionDistance::ActionDistanceLUT& distance_lut) {
  // Get the distance table
  uint8_t lut[256][256];
  distance_lut.get_distance_table(lut);

  // Store locally
  std::memcpy(distance_lut_.data(), lut, sizeof(lut));

  // Upload to GPU constant memory
  gpu_impl_->upload_distance_lut(lut);

  lut_initialized_ = true;
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
      // Note: You'll need to expose encode_action from your LUT
      // encoded_seq[i] = distance_lut_.encode_action(actions[i], args[i]);
      // For now, simple encoding:
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

  // Compute profiles on GPU
  gpu_impl_->compute_profiles(encoded.sequences, encoded.valid_lengths, windows, profiles, indices);

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
  last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

  return results;
}

}  // namespace MatrixProfile
