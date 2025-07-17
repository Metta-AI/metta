#ifndef STATS_TRACKER_HPP_
#define STATS_TRACKER_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <unordered_set>
#include <vector>

// Forward declaration
class MettaGrid;

using InventoryItem = uint8_t;

class StatsTracker {
private:
  std::map<std::string, float> _stats;
  std::map<std::string, unsigned int> _first_seen_at;
  std::map<std::string, unsigned int> _last_seen_at;
  std::map<std::string, float> _min_value;
  std::map<std::string, float> _max_value;
  std::map<std::string, unsigned int> _update_count;
  MettaGrid* _env;

  // Exploration tracking
  std::vector<std::unordered_set<uint16_t>> _agent_explored_pixels;
  bool _track_exploration;

  // Track timing for any update
  void track_timing(const std::string& key) {
    if (_env) {
      unsigned int step = get_current_step();
      if (_first_seen_at.find(key) == _first_seen_at.end()) {
        _first_seen_at[key] = step;
      }
      _last_seen_at[key] = step;
      _update_count[key]++;
    }
  }

  // Helper to get current step - implemented where MettaGrid is complete
  unsigned int get_current_step() const;

  // Track min/max values automatically
  void track_bounds(const std::string& key, float value) {
    if (_min_value.find(key) == _min_value.end()) {
      _min_value[key] = value;
      _max_value[key] = value;
    } else {
      if (value < _min_value[key]) _min_value[key] = value;
      if (value > _max_value[key]) _max_value[key] = value;
    }
  }

  // MettaGrid needs access to implement get_current_step
  friend class MettaGrid;

  // Test class needs access for testing
  friend class StatsTrackerTest;

public:
  inline static const std::string NO_ENV_INVENTORY_ITEM_NAME = "[unknown -- stats tracker not initialized]";

  StatsTracker() : _env(nullptr), _track_exploration(false) {}

  void set_environment(MettaGrid* env) {
    _env = env;
  }

  void set_track_exploration(bool track, size_t num_agents) {
    _track_exploration = track;
    if (track) {
      _agent_explored_pixels.resize(num_agents);
    } else {
      _agent_explored_pixels.clear();
    }
  }

  void track_pixel(size_t agent_idx, uint16_t pixel_coord) {
    if (_track_exploration && agent_idx < _agent_explored_pixels.size()) {
      _agent_explored_pixels[agent_idx].insert(pixel_coord);
    }
  }

  float get_exploration_rate(size_t agent_idx) const {
    if (!_track_exploration || agent_idx >= _agent_explored_pixels.size()) {
      return 0.0f;
    }

    unsigned int steps = get_current_step();
    if (steps == 0) return 0.0f;

    return static_cast<float>(_agent_explored_pixels[agent_idx].size()) / steps;
  }

  void reset_exploration() {
    for (auto& agent_pixels : _agent_explored_pixels) {
      agent_pixels.clear();
    }
  }

  const std::string& inventory_item_name(InventoryItem item) const;

  void add(const std::string& key, float amount) {
    _stats[key] += amount;
    track_timing(key);
    track_bounds(key, _stats[key]);
  }

  // Increment by 1 (convenience method)
  void incr(const std::string& key) {
    add(key, 1);
  }

  void set(const std::string& key, float value) {
    _stats[key] = value;
    track_timing(key);
    track_bounds(key, value);
  }

  // Calculate rate (updates per step)
  float rate(const std::string& key) const {
    if (!_env) return 0.0f;

    auto it = _update_count.find(key);
    if (it == _update_count.end()) return 0.0f;

    unsigned int steps = get_current_step();
    return (steps > 0) ? static_cast<float>(it->second) / static_cast<float>(steps) : 0.0f;
  }

  // Convert to map for Python API (all values as floats)
  std::map<std::string, float> to_dict() const {
    std::map<std::string, float> result;

    // Add all stats
    for (const auto& [key, value] : _stats) {
      result[key] = value;
    }

    // Add timing metadata and calculated stats
    for (const auto& [key, step] : _first_seen_at) {
      result[key + ".first_step"] = static_cast<float>(step);
    }

    for (const auto& [key, step] : _last_seen_at) {
      result[key + ".last_step"] = static_cast<float>(step);
    }

    for (const auto& [key, count] : _update_count) {
      result[key + ".updates"] = static_cast<float>(count);
      result[key + ".rate"] = rate(key);
      result[key + ".avg"] = result[key] / count;

      // Also calculate activity rate if there's a time span
      auto first_it = _first_seen_at.find(key);
      auto last_it = _last_seen_at.find(key);
      if (first_it != _first_seen_at.end() && last_it != _last_seen_at.end()) {
        int duration = static_cast<int>(last_it->second) - static_cast<int>(first_it->second);
        if (duration > 0 && count > 1) {
          result[key + ".activity_rate"] = static_cast<float>(count - 1) / static_cast<float>(duration);
        }
      }
    }

    // Add min/max values
    for (const auto& [key, min_val] : _min_value) {
      result[key + ".min"] = min_val;
    }

    for (const auto& [key, max_val] : _max_value) {
      result[key + ".max"] = max_val;
    }

    return result;
  }

  // Reset all statistics
  void reset() {
    _stats.clear();
    _first_seen_at.clear();
    _last_seen_at.clear();
    _min_value.clear();
    _max_value.clear();
    _update_count.clear();
    reset_exploration();
  }
};

#endif  // STATS_TRACKER_HPP_
