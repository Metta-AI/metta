// extensions/mettagrid_extension.cpp
#include "extensions/mettagrid_extension.hpp"

#include "mettagrid_c.hpp"
#include "observation_tokens.hpp"
#include "packed_coordinate.hpp"

const Agent* MettaGridExtension::getAgent(const MettaGrid* env, size_t agent_idx) const {
  return env->agent(static_cast<uint32_t>(agent_idx));
}

std::span<const uint8_t> MettaGridExtension::getAgentActions(const MettaGrid* env, size_t agent_idx) const {
  const auto& actions = env->_actions;
  size_t actions_per_agent = 2;  // action + action_arg
  size_t offset = agent_idx * actions_per_agent;

  return std::span<const uint8_t>(static_cast<const uint8_t*>(actions.data()) + offset, actions_per_agent);
}

std::span<const uint8_t> MettaGridExtension::getAgentObservations(const MettaGrid* env, size_t agent_idx) const {
  const auto& obs = env->_observations;
  size_t obs_per_agent = env->num_observation_tokens * 3;  // 3 bytes per token: packed_coord, feature, value
  size_t offset = agent_idx * obs_per_agent;

  return std::span<const uint8_t>(static_cast<const uint8_t*>(obs.data()) + offset, obs_per_agent);
}

std::span<uint8_t> MettaGridExtension::getAgentObservationsMutable(MettaGrid* env, size_t agent_idx) {
  auto& obs = env->_observations;
  size_t obs_per_agent = env->num_observation_tokens * 3;  // 3 bytes per token: packed_coord, feature, value
  size_t offset = agent_idx * obs_per_agent;

  return std::span<uint8_t>(static_cast<uint8_t*>(obs.mutable_data()) + offset, obs_per_agent);
}

size_t MettaGridExtension::getObservationSize(const MettaGrid* env) const {
  return env->num_observation_tokens;
}

std::optional<size_t> MettaGridExtension::findEmptyObservationSlot(const MettaGrid* env, size_t agent_idx) const {
  auto agent_obs = getAgentObservations(env, agent_idx);
  size_t num_tokens = env->num_observation_tokens;
  size_t num_channels = 3;  // packed_coord, feature, value

  for (size_t token_idx = 0; token_idx < num_tokens; token_idx++) {
    size_t base_idx = token_idx * num_channels;
    if (agent_obs[base_idx] == OBSERVATION_EMPTY_TOKEN && agent_obs[base_idx + 1] == OBSERVATION_EMPTY_TOKEN &&
        agent_obs[base_idx + 2] == OBSERVATION_EMPTY_TOKEN) {
      return token_idx;
    }
  }
  return std::nullopt;
}

size_t MettaGridExtension::writeObservations(MettaGrid* env,
                                             size_t agent_idx,
                                             const std::vector<ObservationToken>& tokens) {
  size_t written = 0;

  // Get mutable observations once
  auto agent_obs = getAgentObservationsMutable(env, agent_idx);
  size_t num_tokens = env->num_observation_tokens;
  size_t num_channels = 3;

  // Find first empty slot
  size_t current_slot = 0;
  bool found_start = false;
  for (size_t token_idx = 0; token_idx < num_tokens; token_idx++) {
    size_t base_idx = token_idx * num_channels;
    if (agent_obs[base_idx] == OBSERVATION_EMPTY_TOKEN && agent_obs[base_idx + 1] == OBSERVATION_EMPTY_TOKEN &&
        agent_obs[base_idx + 2] == OBSERVATION_EMPTY_TOKEN) {
      current_slot = token_idx;
      found_start = true;
      break;
    }
  }

  if (!found_start) {
    return 0;
  }

  // Write tokens sequentially from the found slot
  for (const auto& token : tokens) {
    if (current_slot >= num_tokens) {
      break;
    }

    size_t base_idx = current_slot * num_channels;
    agent_obs[base_idx] = token.location;
    agent_obs[base_idx + 1] = token.feature_id;
    agent_obs[base_idx + 2] = token.value;

    written++;
    current_slot++;
  }

  return written;
}

size_t MettaGridExtension::writeGlobalObservations(MettaGrid* env,
                                                   size_t agent_idx,
                                                   const std::vector<ObservationType>& features,
                                                   const std::vector<ObservationType>& values) {
  // Check that features and values vectors have the same size
  if (features.size() != values.size()) {
    return 0;
  }

  // Get center position of observation window
  uint8_t center_r = env->obs_height / 2;
  uint8_t center_c = env->obs_width / 2;
  uint8_t packed_center = PackedCoordinate::pack(center_r, center_c);

  // Create observation tokens for all feature-value pairs
  std::vector<ObservationToken> tokens;
  tokens.reserve(features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    ObservationToken token;
    token.location = packed_center;
    token.feature_id = features[i];
    token.value = values[i];
    tokens.push_back(token);
  }

  // Write all tokens at once and return the number written
  return writeObservations(env, agent_idx, tokens);
}
