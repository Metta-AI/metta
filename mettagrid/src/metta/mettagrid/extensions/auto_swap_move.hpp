#ifndef EXTENSIONS_AUTO_SWAP_MOVE_HPP_
#define EXTENSIONS_AUTO_SWAP_MOVE_HPP_

#include <cmath>
#include <limits>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"

class AutoSwapMove : public MettaGridExtension {
public:
  bool overridesAction(const std::string& action_name) const override {
    return action_name == "move";
  }

  bool handleAction(const std::string& action_name, Agent* actor, ActionArg arg) override {
    if (action_name != "move") return false;

    Grid& grid = const_cast<Grid&>(_env->grid());

    // Get orientation from arg
    Orientation move_direction = static_cast<Orientation>(arg);

    // Validate direction based on diagonal support
    if (!isValidOrientation(move_direction, _config->allow_diagonals)) {
      return false;
    }

    // Get movement deltas
    int dc, dr;
    getOrientationDelta(move_direction, dc, dr);

    GridLocation target;
    target.r = static_cast<GridCoord>(static_cast<int>(actor->location.r) + dr);
    target.c = static_cast<GridCoord>(static_cast<int>(actor->location.c) + dc);
    target.layer = actor->location.layer;

    // Update orientation (even if movement fails)
    actor->orientation = move_direction;

    // Check if target is valid
    if (!grid.is_valid_location(target)) {
      return false;
    }

    // Check what's at target
    auto occupant = grid.object_at(target);

    if (!occupant) {
      // Empty - regular move
      return grid.move_object(actor->id, target);
    }

    // Check if blocked by non-agent object
    if (!dynamic_cast<Agent*>(occupant)) {
      return false;  // Blocked by wall/object
    }

    // Agent blocking - perform swap
    grid.swap_objects(actor->id, occupant->id);
    return true;
  }

  void onInit(const MettaGrid* env, const GameConfig* config) override {
    _env = const_cast<MettaGrid*>(env);
    _config = config;
  }

private:
  MettaGrid* _env;
  const GameConfig* _config;
};

REGISTER_EXTENSION("auto_swap_move", AutoSwapMoves)

#endif  // EXTENSIONS_AUTO_SWAP_MOVE_HPP_
