#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_

#include <memory>
#include <vector>

#include "core/grid.hpp"
#include "objects/assembler.hpp"

class Clipper {
public:
  std::shared_ptr<Recipe> recipe;
  float clip_rate;

  Clipper(std::shared_ptr<Recipe> recipe_ptr, float rate) : recipe(std::move(recipe_ptr)), clip_rate(rate) {}

  void clip_at_random(Grid& grid, std::mt19937& rng) {
    if (!recipe || clip_rate <= 0.0f) return;
    for (size_t obj_id = 1; obj_id < grid.objects.size(); obj_id++) {
      auto* obj = grid.object(static_cast<GridObjectId>(obj_id));
      if (!obj) continue;
      if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
        if (assembler->is_clipped) continue;
        if (std::generate_canonical<float, 10>(rng) < clip_rate) {
          assembler->becomeClipped(std::vector<std::shared_ptr<Recipe>>{recipe});
        }
      }
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
