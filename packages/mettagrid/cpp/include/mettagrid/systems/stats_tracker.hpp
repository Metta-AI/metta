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
  MettaGrid* _env;

  // Test class needs access for testing
  friend class StatsTrackerTest;

  // Use a static function to avoid global destructor
  static const std::string& get_no_env_resource_name() {
    static const std::string name = "[unknown -- stats tracker not initialized]";
    return name;
  }

public:
  StatsTracker() : _stats(), _env(nullptr) {}

  void set_environment(MettaGrid* env) {
    _env = env;
  }

  const std::string& resource_name(InventoryItem item) const;

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
  std::map<std::string, float> to_dict() const {
    return _stats;
  }

  // Reset all statistics
  void reset() {
    _stats.clear();
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_STATS_TRACKER_HPP_
