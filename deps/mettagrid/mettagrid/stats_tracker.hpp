#ifndef STATS_TRACKER_HPP
#define STATS_TRACKER_HPP

#include <map>
#include <string>

class StatsTracker {
private:
  std::map<std::string, int> _stats;

public:
  StatsTracker() = default;

  inline std::map<std::string, int> stats() {
    return _stats;
  }

  inline void incr(const std::string& key) {
    _stats[key] += 1;
  }

  inline void incr(const std::string& key1, const std::string& key2) {
    _stats[key1 + "." + key2] += 1;
  }

  inline void incr(const std::string& key1, const std::string& key2, const std::string& key3) {
    _stats[key1 + "." + key2 + "." + key3] += 1;
  }

  inline void incr(const std::string& key1, const std::string& key2, const std::string& key3, const std::string& key4) {
    _stats[key1 + "." + key2 + "." + key3 + "." + key4] += 1;
  }

  inline void add(const std::string& key, int value) {
    _stats[key] += value;
  }

  inline void add(const std::string& key1, const std::string& key2, int value) {
    _stats[key1 + "." + key2] += value;
  }

  inline void add(const std::string& key1, const std::string& key2, const std::string& key3, int value) {
    _stats[key1 + "." + key2 + "." + key3] += value;
  }

  inline void add(const std::string& key1,
                  const std::string& key2,
                  const std::string& key3,
                  const std::string& key4,
                  int value) {
    _stats[key1 + "." + key2 + "." + key3 + "." + key4] += value;
  }

  inline void set_once(const std::string& key, int value) {
    if (_stats.find(key) == _stats.end()) {
      _stats[key] = value;
    }
  }

  inline void set_once(const std::string& key1, const std::string& key2, int value) {
    std::string key = key1 + "." + key2;
    if (_stats.find(key) == _stats.end()) {
      _stats[key] = value;
    }
  }

  inline void set_once(const std::string& key1, const std::string& key2, const std::string& key3, int value) {
    std::string key = key1 + "." + key2 + "." + key3;
    if (_stats.find(key) == _stats.end()) {
      _stats[key] = value;
    }
  }

  inline void set_once(const std::string& key1,
                       const std::string& key2,
                       const std::string& key3,
                       const std::string& key4,
                       int value) {
    std::string key = key1 + "." + key2 + "." + key3 + "." + key4;
    if (_stats.find(key) == _stats.end()) {
      _stats[key] = value;
    }
  }
};

#endif
