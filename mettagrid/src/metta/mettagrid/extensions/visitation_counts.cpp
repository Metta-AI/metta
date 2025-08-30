// extensions/visitation_counts.cpp
#include "extensions/visitation_counts.hpp"

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "packed_coordinate.hpp"

namespace {
constexpr size_t DENSE_THRESHOLD = 100 * 100;

// Pack row/col into uint32_t for hash map key
inline uint32_t packCoord(GridCoord r, GridCoord c) {
  return (static_cast<uint32_t>(r) << 16) | static_cast<uint32_t>(c);
}
}  // namespace

void VisitationCounts::registerObservations(ObservationEncoder* enc) {
  _visitation_count_feature = enc->register_feature("agent:visitation_counts");
}

void VisitationCounts::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();
  _grid_width = env->grid().width;
  _grid_height = env->grid().height;

  // Decide whether to use dense or sparse storage
  _use_dense = (_grid_width * _grid_height <= DENSE_THRESHOLD);

  if (_use_dense) {
    // Allocate dense storage: single contiguous array for better cache performance
    size_t total_size = _num_agents * _grid_width * _grid_height;
    _dense_visits.resize(total_size, 0);
  } else {
    // Allocate sparse storage
    _sparse_visits.resize(_num_agents);
  }
}

void VisitationCounts::onReset(MettaGrid* env) {
  if (_use_dense) {
    // Clear dense storage
    std::fill(_dense_visits.begin(), _dense_visits.end(), 0);

    // Mark initial positions
    for (size_t i = 0; i < _num_agents; i++) {
      const Agent* agent = getAgent(env, i);
      size_t idx = getDenseIndex(i, agent->location.r, agent->location.c);
      _dense_visits[idx] = 1;
    }
  } else {
    // Clear sparse storage
    for (auto& map : _sparse_visits) {
      map.clear();
    }

    // Mark initial positions
    for (size_t i = 0; i < _num_agents; i++) {
      const Agent* agent = getAgent(env, i);
      uint32_t key = packCoord(agent->location.r, agent->location.c);
      _sparse_visits[i][key] = 1;
    }
  }

  // Add initial visitation counts to observations
  addVisitationCountsToObservations(env);
}

void VisitationCounts::onStep(MettaGrid* env) {
  // Update visit counts
  for (size_t i = 0; i < _num_agents; i++) {
    const Agent* agent = getAgent(env, i);

    if (_use_dense) {
      size_t idx = getDenseIndex(i, agent->location.r, agent->location.c);
      // Saturate at 255 instead of wrapping
      if (_dense_visits[idx] < 255) {
        _dense_visits[idx]++;
      }
    } else {
      uint32_t key = packCoord(agent->location.r, agent->location.c);
      auto& count = _sparse_visits[i][key];
      // Saturate at 255
      if (count < 255) {
        count++;
      }
    }
  }

  // Add visitation counts to observations
  addVisitationCountsToObservations(env);
}

std::array<unsigned int, 5> VisitationCounts::computeVisitationCounts(const MettaGrid* env, size_t agent_idx) const {
  std::array<unsigned int, 5> counts = {0, 0, 0, 0, 0};

  if (agent_idx >= _num_agents) {
    return counts;
  }

  const Agent* agent = getAgent(env, agent_idx);
  GridCoord r = agent->location.r;
  GridCoord c = agent->location.c;

  if (_use_dense) {
    // Center
    counts[0] = _dense_visits[getDenseIndex(agent_idx, r, c)];

    // North
    if (r > 0) {
      counts[1] = _dense_visits[getDenseIndex(agent_idx, r - 1, c)];
    }

    // South
    if (r + 1 < _grid_height) {
      counts[2] = _dense_visits[getDenseIndex(agent_idx, r + 1, c)];
    }

    // East
    if (c + 1 < _grid_width) {
      counts[3] = _dense_visits[getDenseIndex(agent_idx, r, c + 1)];
    }

    // West
    if (c > 0) {
      counts[4] = _dense_visits[getDenseIndex(agent_idx, r, c - 1)];
    }
  } else {
    const auto& agent_map = _sparse_visits[agent_idx];

    // Center
    auto it = agent_map.find(packCoord(r, c));
    if (it != agent_map.end()) {
      counts[0] = it->second;
    }

    // North
    if (r > 0) {
      it = agent_map.find(packCoord(r - 1, c));
      if (it != agent_map.end()) {
        counts[1] = it->second;
      }
    }

    // South
    if (r + 1 < _grid_height) {
      it = agent_map.find(packCoord(r + 1, c));
      if (it != agent_map.end()) {
        counts[2] = it->second;
      }
    }

    // East
    if (c + 1 < _grid_width) {
      it = agent_map.find(packCoord(r, c + 1));
      if (it != agent_map.end()) {
        counts[3] = it->second;
      }
    }

    // West
    if (c > 0) {
      it = agent_map.find(packCoord(r, c - 1));
      if (it != agent_map.end()) {
        counts[4] = it->second;
      }
    }
  }

  return counts;
}

void VisitationCounts::addVisitationCountsToObservations(MettaGrid* env) {
  uint8_t center_r = env->obs_height / 2;
  uint8_t center_c = env->obs_width / 2;

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    auto counts = computeVisitationCounts(env, agent_idx);

    // Create observation tokens for the 5 visitation counts
    std::vector<ObservationToken> tokens;
    tokens.reserve(5);  // Pre-allocate for efficiency

    // Center position
    ObservationToken center_token;
    center_token.location = PackedCoordinate::pack(center_r, center_c);
    center_token.feature_id = _visitation_count_feature;
    center_token.value = static_cast<uint8_t>(counts[0]);  // Already limited to 255
    tokens.push_back(center_token);

    // North (r-1) - check bounds
    if (center_r > 0) {
      ObservationToken north_token;
      north_token.location = PackedCoordinate::pack(center_r - 1, center_c);
      north_token.feature_id = _visitation_count_feature;
      north_token.value = static_cast<uint8_t>(counts[1]);
      tokens.push_back(north_token);
    }

    // South (r+1) - check bounds
    if (center_r + 1 < env->obs_height) {
      ObservationToken south_token;
      south_token.location = PackedCoordinate::pack(center_r + 1, center_c);
      south_token.feature_id = _visitation_count_feature;
      south_token.value = static_cast<uint8_t>(counts[2]);
      tokens.push_back(south_token);
    }

    // East (c+1) - check bounds
    if (center_c + 1 < env->obs_width) {
      ObservationToken east_token;
      east_token.location = PackedCoordinate::pack(center_r, center_c + 1);
      east_token.feature_id = _visitation_count_feature;
      east_token.value = static_cast<uint8_t>(counts[3]);
      tokens.push_back(east_token);
    }

    // West (c-1) - check bounds
    if (center_c > 0) {
      ObservationToken west_token;
      west_token.location = PackedCoordinate::pack(center_r, center_c - 1);
      west_token.feature_id = _visitation_count_feature;
      west_token.value = static_cast<uint8_t>(counts[4]);
      tokens.push_back(west_token);
    }

    // Write all tokens at once
    writeObservations(env, agent_idx, tokens);
  }
}

ExtensionStats VisitationCounts::getStats() const {
  ExtensionStats stats;

  // Calculate average stats across all agents
  float avg_total_steps = 0.0f;
  float avg_unique_cells = 0.0f;

  for (size_t i = 0; i < _num_agents; i++) {
    size_t unique_cells = 0;
    size_t total_visits = 0;

    if (_use_dense) {
      // Count non-zero entries for this agent
      size_t agent_offset = i * _grid_width * _grid_height;
      for (size_t j = 0; j < _grid_width * _grid_height; j++) {
        uint8_t visits = _dense_visits[agent_offset + j];
        if (visits > 0) {
          unique_cells++;
          total_visits += visits;
        }
      }
    } else {
      unique_cells = _sparse_visits[i].size();
      for (const auto& [coord, visits] : _sparse_visits[i]) {
        total_visits += visits;
      }
    }

    avg_total_steps += static_cast<float>(total_visits);
    avg_unique_cells += static_cast<float>(unique_cells);
  }

  if (_num_agents > 0) {
    avg_total_steps /= static_cast<float>(_num_agents);
    avg_unique_cells /= static_cast<float>(_num_agents);
  }

  stats["avg_agent_total_steps"] = avg_total_steps;
  stats["avg_agent_unique_cells_visited"] = avg_unique_cells;
  stats["storage_mode_is_dense"] = _use_dense ? 1.0f : 0.0f;

  return stats;
}

REGISTER_EXTENSION("visitation_counts", VisitationCounts)
