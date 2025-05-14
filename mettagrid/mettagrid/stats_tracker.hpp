#ifndef STATS_TRACKER_HPP
#define STATS_TRACKER_HPP

#include <initializer_list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Custom exception class for stats-related errors
class StatsException : public std::runtime_error {
public:
  explicit StatsException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief A thread-safe statistics tracking class that maintains counts using hierarchical keys
 *
 * StatsTracker allows incrementing, adding, and setting of statistical values
 * using dot-separated hierarchical keys, e.g., "category.subcategory.action"
 */
class StatsTracker {
private:
  std::map<std::string, int> _stats;
  mutable std::mutex _mutex;  // For thread safety

  /**
   * @brief Builds a combined key from multiple key segments
   *
   * @param keys A list of key segments to join with dots
   * @return std::string The combined hierarchical key
   */
  inline std::string buildKey(const std::vector<std::string_view>& keys) const noexcept {
    if (keys.empty() || keys[0].empty()) {
      return "";
    }

    std::string result;
    size_t totalLength = 0;

    // Calculate required space to avoid reallocations
    for (const auto& key : keys) {
      totalLength += key.length();
    }
    // Add space for the dots
    totalLength += keys.size() - 1;

    result.reserve(totalLength);

    // Add first key (always required)
    result.append(keys[0]);

    // Add any subsequent non-empty keys with dot separators
    for (size_t i = 1; i < keys.size(); ++i) {
      if (keys[i].empty()) {
        continue;
      }
      result.append(".");
      result.append(keys[i]);
    }

    return result;
  }

public:
  // Default constructor
  StatsTracker() = default;

  // Copy operations are expensive for large maps, so delete them
  // to prevent accidental copies
  StatsTracker(const StatsTracker&) = delete;
  StatsTracker& operator=(const StatsTracker&) = delete;

  // Move operations
  StatsTracker(StatsTracker&& other) noexcept {
    std::lock_guard<std::mutex> lock(other._mutex);
    _stats = std::move(other._stats);
  }

  StatsTracker& operator=(StatsTracker&& other) noexcept {
    if (this != &other) {
      std::lock_guard<std::mutex> lock1(_mutex);
      std::lock_guard<std::mutex> lock2(other._mutex);
      _stats = std::move(other._stats);
    }
    return *this;
  }

  // Default destructor is sufficient since no manual resource management
  ~StatsTracker() = default;

  /**
   * @brief Get a copy of all current statistics
   *
   * @return std::map<std::string, int> A copy of the statistics map
   */
  inline std::map<std::string, int> stats() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _stats;  // Return a copy for thread safety
  }

  /**
   * @brief Get a const reference to all current statistics (more efficient but not thread-safe)
   *
   * @return const std::map<std::string, int>& A reference to the statistics map
   * @warning Only use this method if you're already handling thread safety externally
   */
  inline const std::map<std::string, int>& stats_unsafe_ref() const noexcept {
    return _stats;
  }

  /**
   * @brief Increment a stat by 1
   *
   * @param keys A list of key segments to form the hierarchical key
   * @throws StatsException if the primary key is empty
   */
  inline void incr(std::initializer_list<std::string_view> keys) {
    add(keys, 1);
  }

  /**
   * @brief Add a value to a stat
   *
   * @param keys A list of key segments to form the hierarchical key
   * @param value The value to add
   * @throws StatsException if the primary key is empty
   */
  inline void add(std::initializer_list<std::string_view> keys, int value) {
    std::vector<std::string_view> keyVector(keys);
    if (keyVector.empty() || keyVector[0].empty()) {
      throw StatsException("Cannot add with empty primary key");
    }

    try {
      std::string combinedKey = buildKey(keyVector);

      std::lock_guard<std::mutex> lock(_mutex);
      _stats[combinedKey] += value;
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed to add to multi-key stat: ") + e.what());
    }
  }

  /**
   * @brief Set a value once if the key doesn't exist yet
   *
   * @param keys A list of key segments to form the hierarchical key
   * @param value The value to set
   * @throws StatsException if the primary key is empty
   */
  inline void set_once(std::initializer_list<std::string_view> keys, int value) {
    std::vector<std::string_view> keyVector(keys);
    if (keyVector.empty() || keyVector[0].empty()) {
      throw StatsException("Cannot set_once with empty primary key");
    }

    try {
      std::string combinedKey = buildKey(keyVector);

      std::lock_guard<std::mutex> lock(_mutex);
      // Just call try_emplace without capturing the return value
      // try_emplace only inserts if the key doesn't exist
      _stats.try_emplace(combinedKey, value);
    } catch (const std::exception& e) {
      throw StatsException(std::string("Failed in set_once for multi-key stat: ") + e.what());
    }
  }

  /**
   * @brief Clear all statistics
   */
  inline void clear() noexcept {
    std::lock_guard<std::mutex> lock(_mutex);
    _stats.clear();
  }

  /**
   * @brief Get a specific stat value
   *
   * @param keys A list of key segments to form the hierarchical key
   * @return int The value (0 if not found)
   */
  inline int get(std::initializer_list<std::string_view> keys) const {
    std::vector<std::string_view> keyVector(keys);
    if (keyVector.empty()) {
      return 0;
    }

    std::string combinedKey = buildKey(keyVector);

    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _stats.find(combinedKey);
    return (it != _stats.end()) ? it->second : 0;
  }

  /**
   * @brief Check if a specific stat exists
   *
   * @param keys A list of key segments to form the hierarchical key
   * @return bool True if the stat exists
   */
  inline bool has(std::initializer_list<std::string_view> keys) const {
    std::vector<std::string_view> keyVector(keys);
    if (keyVector.empty()) {
      return false;
    }

    std::string combinedKey = buildKey(keyVector);

    std::lock_guard<std::mutex> lock(_mutex);
    return _stats.find(combinedKey) != _stats.end();
  }

  /**
   * @brief Generate a string representation of all stats
   *
   * @return std::string Formatted string with all stat keys and values
   */
  std::string dump_stats() const {
    std::lock_guard<std::mutex> lock(_mutex);

    std::string result = "StatsTracker state:\n";
    for (const auto& [key, value] : _stats) {
      result += "  " + key + ": " + std::to_string(value) + "\n";
    }
    return result;
  }

  // Backward compatibility methods for transition
  inline void incr(const std::string_view& key) {
    incr({key});
  }
  inline void incr(const std::string_view& key1, const std::string_view& key2) {
    incr({key1, key2});
  }
  inline void incr(const std::string_view& key1, const std::string_view& key2, const std::string_view& key3) {
    incr({key1, key2, key3});
  }
  inline void incr(const std::string_view& key1,
                   const std::string_view& key2,
                   const std::string_view& key3,
                   const std::string_view& key4) {
    incr({key1, key2, key3, key4});
  }

  inline void add(const std::string_view& key, int value) {
    add({key}, value);
  }
  inline void add(const std::string_view& key1, const std::string_view& key2, int value) {
    add({key1, key2}, value);
  }
  inline void add(const std::string_view& key1, const std::string_view& key2, const std::string_view& key3, int value) {
    add({key1, key2, key3}, value);
  }
  inline void add(const std::string_view& key1,
                  const std::string_view& key2,
                  const std::string_view& key3,
                  const std::string_view& key4,
                  int value) {
    add({key1, key2, key3, key4}, value);
  }

  inline void set_once(const std::string_view& key, int value) {
    set_once({key}, value);
  }
  inline void set_once(const std::string_view& key1, const std::string_view& key2, int value) {
    set_once({key1, key2}, value);
  }
  inline void set_once(const std::string_view& key1,
                       const std::string_view& key2,
                       const std::string_view& key3,
                       int value) {
    set_once({key1, key2, key3}, value);
  }
  inline void set_once(const std::string_view& key1,
                       const std::string_view& key2,
                       const std::string_view& key3,
                       const std::string_view& key4,
                       int value) {
    set_once({key1, key2, key3, key4}, value);
  }
};

#endif