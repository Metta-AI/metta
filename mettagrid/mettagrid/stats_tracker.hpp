#ifndef STATS_TRACKER_HPP
#define STATS_TRACKER_HPP

#include <map>
#include <stdexcept>
#include <string>

// Custom exception class for stats-related errors
class StatsException : public std::runtime_error {
public:
  explicit StatsException(const std::string& message) : std::runtime_error(message) {}
};

class StatsTracker {
private:
  std::map<std::string, int> _stats;

public:
  StatsTracker() = default;

  inline std::map<std::string, int> stats() const {
    return _stats;
  }

  inline void incr(const std::string& key) {
    if (key.empty()) {
      throw StatsException("Cannot increment with empty key");
    }
    try {
      _stats[key] += 1;
    } catch (const std::exception& e) {
      throw StatsException("Failed to increment stat '" + key + "': " + e.what());
    }
  }

  inline void incr(const std::string& key1, const std::string& key2) {
    if (key1.empty()) {
      throw StatsException("Cannot increment with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      _stats[combinedKey] += 1;
    } catch (const std::exception& e) {
      throw StatsException("Failed to increment combined stat '" + key1 + "." + key2 + "': " + e.what());
    }
  }

  inline void incr(const std::string& key1, const std::string& key2, const std::string& key3) {
    if (key1.empty()) {
      throw StatsException("Cannot increment with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      _stats[combinedKey] += 1;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to increment multi-key stat: ") + e.what());
    }
  }

  inline void incr(const std::string& key1, const std::string& key2, const std::string& key3, const std::string& key4) {
    if (key1.empty()) {
      throw StatsException("Cannot increment with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      if (!key4.empty()) {
        combinedKey += "." + key4;
      }
      _stats[combinedKey] += 1;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to increment multi-key stat: ") + e.what());
    }
  }

  inline void add(const std::string& key, int value) {
    if (key.empty()) {
      throw StatsException("Cannot add to empty key");
    }
    try {
      _stats[key] += value;
    } catch (const std::exception& e) {
      throw StatsException("Failed to add to stat '" + key + "': " + e.what());
    }
  }

  inline void add(const std::string& key1, const std::string& key2, int value) {
    if (key1.empty()) {
      throw StatsException("Cannot add with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      _stats[combinedKey] += value;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to add to combined stat: ") + e.what());
    }
  }

  inline void add(const std::string& key1, const std::string& key2, const std::string& key3, int value) {
    if (key1.empty()) {
      throw StatsException("Cannot add with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      _stats[combinedKey] += value;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to add to multi-key stat: ") + e.what());
    }
  }

  inline void add(const std::string& key1,
                  const std::string& key2,
                  const std::string& key3,
                  const std::string& key4,
                  int value) {
    if (key1.empty()) {
      throw StatsException("Cannot add with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      if (!key4.empty()) {
        combinedKey += "." + key4;
      }
      _stats[combinedKey] += value;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to add to multi-key stat: ") + e.what());
    }
  }

  inline void set_once(const std::string& key, int value) {
    if (key.empty()) {
      throw StatsException("Cannot set_once with empty key");
    }
    try {
      if (_stats.find(key) == _stats.end()) {
        _stats[key] = value;
      }
    } catch (const std::exception& e) {
      throw StatsException("Failed in set_once for '" + key + "': " + e.what());
    }
  }

  inline void set_once(const std::string& key1, const std::string& key2, int value) {
    if (key1.empty()) {
      throw StatsException("Cannot set_once with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (_stats.find(combinedKey) == _stats.end()) {
        _stats[combinedKey] = value;
      }
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed in set_once for combined key: ") + e.what());
    }
  }

  inline void set_once(const std::string& key1, const std::string& key2, const std::string& key3, int value) {
    if (key1.empty()) {
      throw StatsException("Cannot set_once with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      if (_stats.find(combinedKey) == _stats.end()) {
        _stats[combinedKey] = value;
      }
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed in set_once for multi-key stat: ") + e.what());
    }
  }

  inline void set_once(const std::string& key1,
                       const std::string& key2,
                       const std::string& key3,
                       const std::string& key4,
                       int value) {
    if (key1.empty()) {
      throw StatsException("Cannot set_once with empty primary key");
    }
    try {
      std::string combinedKey = key1;
      if (!key2.empty()) {
        combinedKey += "." + key2;
      }
      if (!key3.empty()) {
        combinedKey += "." + key3;
      }
      if (!key4.empty()) {
        combinedKey += "." + key4;
      }
      if (_stats.find(combinedKey) == _stats.end()) {
        _stats[combinedKey] = value;
      }
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed in set_once for multi-key stat: ") + e.what());
    }
  }

  // Helper method for debugging that returns a string rather than writing to stream
  std::string dump_stats() const {
    std::string result = "StatsTracker state:\n";
    for (const auto& pair : _stats) {
      result += "  " + pair.first + ": " + std::to_string(pair.second) + "\n";
    }
    return result;
  }
};

#endif
