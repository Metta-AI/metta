#ifndef GRID_HPP_
#define GRID_HPP_

#include <algorithm>
#include <bitset>
#include <map>
#include <memory>
#include <vector>

#include "grid_object.hpp"
#include "objects/constants.hpp"
#include "objects/agent.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<std::vector<GridObjectId>>>;

// Maximum grid size for exploration tracking (adjust as needed)
constexpr size_t MAX_GRID_SIZE = 10000;  // 100x100 grid max

struct ExplorationTracker {
  std::vector<std::bitset<MAX_GRID_SIZE>> agent_visited_positions;
  std::vector<std::bitset<MAX_GRID_SIZE>> agent_observed_pixels;
  std::bitset<MAX_GRID_SIZE> total_observed_pixels;
  std::vector<int> threshold_times;  // [33%, 66%, 100%]
  int total_grid_cells;
  int current_step;
  int max_steps;
  int num_agents;
  bool enabled;
  int width;  // Store grid width for index calculation

  ExplorationTracker() : enabled(false), total_grid_cells(0), current_step(0), max_steps(0), num_agents(0), width(0) {
    threshold_times.resize(3, -1);  // Initialize to -1 (not reached)
  }

  void initialize(int width, int height, int num_agents, int max_steps, bool enable_tracking) {
    this->width = width;
    this->num_agents = num_agents;
    this->max_steps = max_steps;
    this->total_grid_cells = width * height;
    this->enabled = enable_tracking;

    if (enabled) {
      agent_visited_positions.resize(num_agents);
      agent_observed_pixels.resize(num_agents);
      // Reset all bitsets
      for (int i = 0; i < num_agents; ++i) {
        agent_visited_positions[i].reset();
        agent_observed_pixels[i].reset();
      }
      total_observed_pixels.reset();
      threshold_times.assign(3, -1);
    }
  }

  void reset() {
    if (!enabled) return;

    for (int i = 0; i < num_agents; ++i) {
      agent_visited_positions[i].reset();
      agent_observed_pixels[i].reset();
    }
    total_observed_pixels.reset();
    threshold_times.assign(3, -1);
  }

  void track_visit(unsigned char agent_id, int r, int c) {
    if (!enabled || agent_id >= num_agents) return;

    int index = r * width + c;
    if (index < MAX_GRID_SIZE) {
      agent_visited_positions[agent_id].set(index);
    }
  }

  void track_observation(unsigned char agent_id, int r, int c) {
    if (!enabled || agent_id >= num_agents) return;

    int index = r * width + c;
    if (index < MAX_GRID_SIZE) {
      agent_observed_pixels[agent_id].set(index);
      total_observed_pixels.set(index);
    }
  }

  void track_observation_window(unsigned char agent_id, int center_r, int center_c,
                              int obs_width, int obs_height, int grid_width, int grid_height) {
    if (!enabled || agent_id >= num_agents) return;

    int radius_w = obs_width / 2;
    int radius_h = obs_height / 2;

    for (int r = center_r - radius_h; r <= center_r + radius_h; r++) {
      for (int c = center_c - radius_w; c <= center_c + radius_w; c++) {
        if (r >= 0 && r < grid_height && c >= 0 && c < grid_width) {
          track_observation(agent_id, r, c);
        }
      }
    }
  }

  void update_threshold_times() {
    if (!enabled || total_grid_cells == 0) return;

    int total_observed = total_observed_pixels.count();
    float coverage_percentage = (float(total_observed) / total_grid_cells) * 100.0f;

    // Check thresholds in order
    if (coverage_percentage >= 33.0f && threshold_times[0] == -1) {
      threshold_times[0] = current_step;
    }
    if (coverage_percentage >= 66.0f && threshold_times[1] == -1) {
      threshold_times[1] = current_step;
    }
    if (coverage_percentage >= 100.0f && threshold_times[2] == -1) {
      threshold_times[2] = current_step;
    }
  }

  std::map<std::string, float> get_metrics() {
    std::map<std::string, float> metrics;

    if (!enabled) return metrics;

    // Calculate visited locations metrics
    std::vector<int> visited_counts;
    std::bitset<MAX_GRID_SIZE> all_visited;

    for (int i = 0; i < num_agents; ++i) {
      visited_counts.push_back(agent_visited_positions[i].count());
      all_visited |= agent_visited_positions[i];
    }

    int total_unique_visited = all_visited.count();
    float avg_visited = num_agents > 0 ? float(std::accumulate(visited_counts.begin(), visited_counts.end(), 0)) / num_agents : 0.0f;

    // Calculate observed pixels metrics
    std::vector<int> observed_counts;
    for (int i = 0; i < num_agents; ++i) {
      observed_counts.push_back(agent_observed_pixels[i].count());
    }

    int total_unique_observed = total_observed_pixels.count();
    float avg_observed = num_agents > 0 ? float(std::accumulate(observed_counts.begin(), observed_counts.end(), 0)) / num_agents : 0.0f;

    // Core metrics
    metrics["explore/total_unique_visits"] = float(total_unique_visited);
    metrics["explore/total_unique_observations"] = float(total_unique_observed);
    metrics["explore/avg_unique_visits_per_agent"] = avg_visited;
    metrics["explore/avg_unique_observations_per_agent"] = avg_observed;

    // Volume normalized metrics
    if (total_grid_cells > 0) {
      metrics["explore/unique_visits_normalized_by_volume"] = float(total_unique_visited) / total_grid_cells;
      metrics["explore/unique_observations_normalized_by_volume"] = float(total_unique_observed) / total_grid_cells;
    }

    // Agent and volume normalized metrics
    if (total_grid_cells > 0 && num_agents > 0) {
      metrics["explore/unique_visits_normalized_by_agent_volume"] = float(total_unique_visited) / (total_grid_cells * num_agents);
      metrics["explore/unique_observations_normalized_by_agent_volume"] = float(total_unique_observed) / (total_grid_cells * num_agents);
    }

    // Effective rate of exploration
    if (num_agents > 0 && max_steps > 0) {
      int obs_area = 11 * 11;  // 11x11 observation window
      int initial_obs_total = obs_area * num_agents;
      int new_pixels_discovered = total_unique_observed - initial_obs_total;
      float exploration_rate = float(new_pixels_discovered) / (max_steps * num_agents);
      metrics["explore/effective_exploration_rate"] = exploration_rate;
    }

    // Median time to threshold percentage of observations
    for (int i = 0; i < 3; ++i) {
      std::string threshold_name = (i == 0) ? "33%" : (i == 1) ? "66%" : "100%";
      if (threshold_times[i] >= 0) {
        metrics["explore/median_time_to_" + threshold_name + "_coverage"] = float(threshold_times[i]);
      } else {
        metrics["explore/median_time_to_" + threshold_name + "_coverage"] = float(max_steps);
      }
    }

    // Standard deviations
    if (visited_counts.size() > 1) {
      float mean_visited = std::accumulate(visited_counts.begin(), visited_counts.end(), 0.0f) / visited_counts.size();
      float variance_visited = 0.0f;
      for (int count : visited_counts) {
        variance_visited += (count - mean_visited) * (count - mean_visited);
      }
      variance_visited /= visited_counts.size();
      metrics["explore/std_dev_agent_visits"] = std::sqrt(variance_visited);
    } else {
      metrics["explore/std_dev_agent_visits"] = 0.0f;
    }

    if (observed_counts.size() > 1) {
      float mean_observed = std::accumulate(observed_counts.begin(), observed_counts.end(), 0.0f) / observed_counts.size();
      float variance_observed = 0.0f;
      for (int count : observed_counts) {
        variance_observed += (count - mean_observed) * (count - mean_observed);
      }
      variance_observed /= observed_counts.size();
      metrics["explore/std_dev_agent_observations"] = std::sqrt(variance_observed);
    } else {
      metrics["explore/std_dev_agent_observations"] = 0.0f;
    }

    return metrics;
  }
};

class Grid {
public:
  unsigned int width;
  unsigned int height;

  GridType grid;
  vector<std::unique_ptr<GridObject>> objects;
  ExplorationTracker exploration_tracker;

  inline Grid(unsigned int width, unsigned int height) : width(width), height(height) {
    grid.resize(height, vector<vector<GridObjectId>>(width, vector<GridObjectId>(GridLayer::GridLayerCount, 0)));

    // 0 is reserved for empty space
    objects.push_back(nullptr);
  }

  virtual ~Grid() = default;

  inline char add_object(GridObject* obj) {
    if (obj->location.r >= height || obj->location.c >= width || obj->location.layer >= GridLayer::GridLayerCount) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != 0) {
      return false;
    }

    obj->id = this->objects.size();
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
    return true;
  }

  // Removes and object from the grid and gives ownership of the object to the caller.
  // Since the caller is now the owner, this can make the raw pointer invalid, if the
  // returned unique_ptr is destroyed.
  inline unique_ptr<GridObject> remove_object(GridObject* obj) {
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    auto obj_ptr = this->objects[obj->id].release();
    this->objects[obj->id] = nullptr;
    return std::unique_ptr<GridObject>(obj_ptr);
  }

  inline char move_object(GridObjectId id, const GridLocation& loc) {
    if (loc.r >= height || loc.c >= width || loc.layer >= GridLayer::GridLayerCount) {
      return false;
    }

    if (grid[loc.r][loc.c][loc.layer] != 0) {
      return false;
    }

    GridObject* obj = object(id);

    // Track exploration for agents
    if (obj->type_id == 0) {  // Agent type ID is 0
      // Cast to Agent to get agent_id
      Agent* agent = dynamic_cast<Agent*>(obj);
      if (agent) {
        exploration_tracker.track_visit(agent->agent_id, loc.r, loc.c);
      }
    }

    grid[loc.r][loc.c][loc.layer] = id;
    grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    obj->location = loc;
    return true;
  }

  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = object(id1);
    GridLocation loc1 = obj1->location;
    Layer layer1 = loc1.layer;
    grid[loc1.r][loc1.c][loc1.layer] = 0;

    GridObject* obj2 = object(id2);
    GridLocation loc2 = obj2->location;
    Layer layer2 = loc2.layer;
    grid[loc2.r][loc2.c][loc2.layer] = 0;

    // Keep the layer the same
    obj1->location = loc2;
    obj1->location.layer = layer1;
    obj2->location = loc1;
    obj2->location.layer = layer2;

    grid[obj1->location.r][obj1->location.c][obj1->location.layer] = id1;
    grid[obj2->location.r][obj2->location.c][obj2->location.layer] = id2;
  }

  inline GridObject* object(GridObjectId obj_id) {
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) {
    if (loc.r >= height || loc.c >= width || loc.layer >= GridLayer::GridLayerCount) {
      return nullptr;
    }
    if (grid[loc.r][loc.c][loc.layer] == 0) {
      return nullptr;
    }
    return object(grid[loc.r][loc.c][loc.layer]);
  }

  inline const GridLocation location(GridObjectId id) {
    return object(id)->location;
  }

  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation facing,
                                              short forward_distance,  // + is forward, - is backward
                                              short lateral_offset) {  // + is relative right, - is relative left
    const int r = static_cast<int>(loc.r);
    const int c = static_cast<int>(loc.c);

    int new_r;
    int new_c;

    switch (facing) {
      case Orientation::Up:
        new_r = r - forward_distance;  // Positive dist = go up (decrease row)
        new_c = c + lateral_offset;    // Positive offset = go right (increase col)
        break;
      case Orientation::Down:
        new_r = r + forward_distance;  // Positive dist = go down (increase row)
        new_c = c - lateral_offset;    // Positive offset = go left (decrease col)
        break;
      case Orientation::Left:
        new_c = c - forward_distance;  // Positive dist = go left (decrease col)
        new_r = r - lateral_offset;    // Positive offset = go up (decrease row)
        break;
      case Orientation::Right:
        new_c = c + forward_distance;  // Positive dist = go right (increase col)
        new_r = r + lateral_offset;    // Positive offset = go down (increase row)
        break;
      default:
        assert(false && "Invalid orientation passed to relative_location()");
    }
    new_r = std::clamp(new_r, 0, static_cast<int>(this->height - 1));
    new_c = std::clamp(new_c, 0, static_cast<int>(this->width - 1));
    return GridLocation(static_cast<GridCoord>(new_r), static_cast<GridCoord>(new_c), loc.layer);
  }

  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation) {
    return this->relative_location(loc, orientation, 1, 0);
  }

  inline char is_empty(unsigned int row, unsigned int col) {
    GridLocation loc;
    loc.r = row;
    loc.c = col;
    for (int layer = 0; layer < GridLayer::GridLayerCount; ++layer) {
      loc.layer = layer;
      if (object_at(loc) != nullptr) {
        return 0;
      }
    }
    return 1;
  }
};

#endif  // GRID_HPP_
