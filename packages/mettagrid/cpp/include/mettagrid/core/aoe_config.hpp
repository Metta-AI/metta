#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_CONFIG_HPP_

#include <vector>

#include "core/types.hpp"

namespace mettagrid {

enum class AOEAlignmentFilter {
  any,                  // All objects in range
  same_collective,      // Only objects aligned to same collective as source
  different_collective  // Only objects aligned to different collective
};

struct AOEResourceDelta {
  InventoryItem resource_id;  // Resource index
  InventoryDelta delta;       // Amount to change per tick

  AOEResourceDelta() : resource_id(0), delta(0) {}
  AOEResourceDelta(InventoryItem res_id, InventoryDelta d) : resource_id(res_id), delta(d) {}
};

struct AOEConfig {
  int radius = 0;                        // L-infinity (Chebyshev) distance
  std::vector<AOEResourceDelta> deltas;  // Resource changes per tick
  std::vector<int> target_tag_ids;       // Filter by object tag IDs (empty = all)
  AOEAlignmentFilter alignment_filter = AOEAlignmentFilter::any;

  AOEConfig() = default;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_CONFIG_HPP_
