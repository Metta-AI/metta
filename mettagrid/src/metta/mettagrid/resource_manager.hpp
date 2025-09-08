#ifndef RESOURCE_MANAGER_HPP_
#define RESOURCE_MANAGER_HPP_

#include <map>
#include <vector>

#include "grid.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

// Forward declarations
class Grid;
class Agent;
class Converter;

class ResourceManager {
public:
  explicit ResourceManager(Grid* grid);

  // Main processing method called once per step
  void step();

  // Team-based object tracking
  void register_agent(Agent* agent);
  void register_converter(Converter* converter);
  void unregister_agent(GridObjectId agent_id);
  void unregister_converter(GridObjectId converter_id);

  // Team-based access methods
  const std::vector<Agent*>& get_agents_by_team(unsigned int team_id) const;
  const std::vector<Converter*>& get_converters_by_team(unsigned int team_id) const;
  std::vector<unsigned int> get_all_teams() const;

  // Agent inventory management
  InventoryDelta modify_agent_inventory(GridObjectId agent_id, InventoryItem item, InventoryDelta delta);
  InventoryQuantity get_agent_inventory(GridObjectId agent_id, InventoryItem item) const;
  const std::map<InventoryItem, InventoryQuantity>& get_agent_inventory(GridObjectId agent_id) const;

  // Converter inventory management
  InventoryDelta modify_converter_inventory(GridObjectId converter_id, InventoryItem item, InventoryDelta delta);
  InventoryQuantity get_converter_inventory(GridObjectId converter_id, InventoryItem item) const;
  const std::map<InventoryItem, InventoryQuantity>& get_converter_inventory(GridObjectId converter_id) const;

  // Utility methods
  std::vector<GridObjectId> get_all_agent_ids() const;
  std::vector<GridObjectId> get_all_converter_ids() const;
  bool is_agent(GridObjectId id) const;
  bool is_converter(GridObjectId id) const;

  // Resource transfer between objects
  InventoryDelta transfer_resource(GridObjectId from_id, GridObjectId to_id, InventoryItem item, InventoryQuantity amount);

  // Team-based resource operations
  InventoryQuantity get_team_total_inventory(unsigned int team_id, InventoryItem item) const;
  void distribute_resources_to_team(unsigned int team_id, InventoryItem item, InventoryQuantity total_amount);

private:
  Grid* _grid;

  // Team-based tracking maps
  std::map<unsigned int, std::vector<Agent*>> _agents_by_team;
  std::map<unsigned int, std::vector<Converter*>> _converters_by_team;

  // Helper methods
  Agent* _get_agent(GridObjectId agent_id) const;
  Converter* _get_converter(GridObjectId converter_id) const;
  void _validate_object_id(GridObjectId id) const;
  void _update_team_maps();
};

#endif  // RESOURCE_MANAGER_HPP_
