#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

// Forward declaration
class MettaGrid;

using InventoryItem = uint8_t;

class StatsTracker {
private:
  std::unordered_map<std::string, float> _stats;
  // StatsTracker holds a reference to resource_names to make it easier to track stats for each resource.
  // The environment owns this reference, so it should live as long as we're going to use it.
  const std::vector<std::string>* _resource_names;

  // Test class needs access for testing
  friend class StatsTrackerTest;

public:
  explicit StatsTracker(const std::vector<std::string>* resource_names) : _stats(), _resource_names(resource_names) {
    if (resource_names == nullptr) {
      throw std::invalid_argument("resource_names cannot be null");
    }
  }

  const std::string& resource_name(InventoryItem item) const {
    return (*_resource_names)[item];
  }

  void add(const std::string& key, float amount) {
    _stats[key] += amount;
  }

  // Increment by 1 (convenience method)
  void incr(const std::string& key) {
    add(key, 1);
  }

  void set(const std::string& key, float value) {
    _stats[key] = value;
  }

  float get(const std::string& key) const {
    auto it = _stats.find(key);
    if (it == _stats.end()) {
      return 0.0f;
    }
    return it->second;
  }

  // Convert to map for Python API
  const std::unordered_map<std::string, float>& to_dict() const {
    return _stats;
  }

  // Reset all statistics
  void reset() {
    _stats.clear();
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_
