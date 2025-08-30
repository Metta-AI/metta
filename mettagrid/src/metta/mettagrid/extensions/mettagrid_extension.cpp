// extensions/mettagrid_extension.cpp
#include "extensions/mettagrid_extension.hpp"

#include "mettagrid_c.hpp"

std::span<const uint8_t> MettaGridExtension::getAgentObservations(const MettaGrid* env, size_t agent_idx) const {
  const auto& obs = env->_observations;
  size_t obs_per_agent = env->obs_height * env->obs_width * 5;  // 5 channels
  size_t offset = agent_idx * obs_per_agent;

  return std::span<const uint8_t>(static_cast<const uint8_t*>(obs.data()) + offset, obs_per_agent);
}

std::span<uint8_t> MettaGridExtension::getAgentObservationsMutable(MettaGrid* env, size_t agent_idx) {
  auto& obs = env->_observations;
  size_t obs_per_agent = env->obs_height * env->obs_width * 5;  // 5 channels
  size_t offset = agent_idx * obs_per_agent;

  return std::span<uint8_t>(static_cast<uint8_t*>(obs.mutable_data()) + offset, obs_per_agent);
}
