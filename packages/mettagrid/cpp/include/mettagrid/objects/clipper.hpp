#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPER_HPP_

#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/clipped_grid_object.hpp"
#include "objects/clipped_grid_object_config.hpp"

class Clipper {
public:
  Grid* grid;
  std::vector<std::shared_ptr<Recipe>> unclipping_recipes;
  std::mt19937* _rng{};

  Clipper(Grid* grid, std::mt19937* rng) : grid(grid), _rng(rng) {}

  GridObject* pick_building_to_clip() {}

  void unclip(ClippedGridObject* clipped) {
    grid->add_object(clipped->clipped_object.release());
    grid->remove_object(clipped->id);
  }

  // replaces a building with a Clipped
  void clip(GridObject& to_clip) {
    // Keep the same type_id and name as the original building. So it'll look the same, but have an extra
    // "clipped" observation.
    const TypeId type_id = to_clip.type_id;
    const std::string type_name = to_clip.type_name;
    std::vector<int> tag_ids = to_clip.tag_ids;
    ClippedGridObjectConfig config(type_id, type_name, tag_ids);

    if (!unclipping_recipes.empty() && _rng != nullptr) {
      std::uniform_int_distribution<size_t> dist(0, unclipping_recipes.size() - 1);
      std::shared_ptr<Recipe> selected_recipe = unclipping_recipes[dist(*_rng)];

      // Fill all slots with the same recipe. For now the recipe can be done however agents want.
      config.recipes.resize(256);
      for (int i = 0; i < 256; i++) {
        config.recipes[i] = selected_recipe;
      }
    }
    ClippedGridObject* clipped = new ClippedGridObject(to_clip.location.r, to_clip.location.c, config, this, &to_clip);
    grid->add_object(clipped);
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPER_HPP_
