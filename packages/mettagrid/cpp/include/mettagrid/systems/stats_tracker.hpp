#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <variant>

// Forward declaration
class MettaGrid;

using InventoryItem = uint8_t;

class StatsTracker {
private:
  std::map<std::string, float> _stats;
  // StatsTracker holds a reference to resource_names to make it easier to track stats for each resource.
  // The environment owns this reference, so it should live as long as we're going to use it.
  const std::vector<std::string>* _resource_names;

  // Test class needs access for testing
  friend class StatsTrackerTest;

  // Use a static function to avoid global destructor
  static const std::string& get_unknown_resource_name() {
    static const std::string name = "[unknown -- StatsTracker resource names not initialized]";
    return name;
  }

public:
  StatsTracker() : _stats(), _resource_names(nullptr) {}

  void set_resource_names(const std::vector<std::string>* resource_names) {
    _resource_names = resource_names;
  }

  const std::string& resource_name(InventoryItem item) const {
    if (_resource_names == nullptr) {
      return get_unknown_resource_name();
    }
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

  float get(const std::string& key) {
    return _stats[key];
  }

  // Convert to map for Python API
  const std::map<std::string, float>& to_dict() const {
    return _stats;
  }

  // Reset all statistics
  void reset() {
    _stats.clear();
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_
