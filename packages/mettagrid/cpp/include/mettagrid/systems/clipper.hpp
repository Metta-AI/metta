#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_

#include <cmath>
#include <memory>
#include <vector>

#include "core/grid.hpp"
#include "objects/assembler.hpp"

class Clipper {
public:
  std::shared_ptr<Recipe> recipe;
  std::map<Assembler*, float> assembler_infection_weight;
  std::set<Assembler*> unclipped_assemblers;
  float length_scale;
  float cutoff_distance;
  Grid& grid;
  float clip_rate;

  Clipper(Grid& grid, std::shared_ptr<Recipe> recipe_ptr, float length_scale, float cutoff_distance, float clip_rate)
      : recipe(std::move(recipe_ptr)),
        length_scale(length_scale),
        cutoff_distance(cutoff_distance),
        grid(grid),
        clip_rate(clip_rate) {
    for (size_t obj_id = 1; obj_id < grid.objects.size(); obj_id++) {
      auto* obj = grid.object(static_cast<GridObjectId>(obj_id));
      if (!obj) continue;
      if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
        // Skip clip-immune assemblers
        if (assembler->clip_immune) continue;
        assembler_infection_weight[assembler] = 0.0f;
        unclipped_assemblers.insert(assembler);
      }
    }
  }

  float infection_weight(Assembler& from, Assembler& to) const {
    float distance = this->distance(from, to);
    if (cutoff_distance > 0.0f && distance > cutoff_distance) return 0.0f;
    return std::exp(-distance / length_scale);
  }

  // it's a little funky to use L2 distance here, since everywhere else we use L1 or Linf.
  float distance(Assembler& assembler_a, Assembler& assembler_b) const {
    GridLocation location_a = assembler_a.location;
    GridLocation location_b = assembler_b.location;
    return std::sqrt(std::pow(location_a.r - location_b.r, 2) + std::pow(location_a.c - location_b.c, 2));
  }

  void clip_assembler(Assembler& to_infect) {
    for (auto& [other, weight] : assembler_infection_weight) {
      if (other == &to_infect) continue;
      weight += infection_weight(to_infect, *other);
    }
    unclipped_assemblers.erase(&to_infect);
    std::vector<std::shared_ptr<Recipe>> unclip_recipes;
    unclip_recipes.assign(256, recipe);
    to_infect.become_clipped(unclip_recipes, this);
  }

  void on_unclip_assembler(Assembler& to_unclip) {
    for (auto& [other, weight] : assembler_infection_weight) {
      if (other == &to_unclip) continue;
      weight -= infection_weight(to_unclip, *other);
    }
    unclipped_assemblers.insert(&to_unclip);
  }

  Assembler* pick_assembler_to_clip(std::mt19937& rng) {
    float total_weight = 0.0f;
    for (auto& candidate_assembler : unclipped_assemblers) {
      total_weight += assembler_infection_weight.at(candidate_assembler);
    }
    float random_weight = std::generate_canonical<float, 10>(rng) * total_weight;
    for (auto& candidate_assembler : unclipped_assemblers) {
      random_weight -= assembler_infection_weight.at(candidate_assembler);
      if (random_weight <= 0.0f) {
        return candidate_assembler;
      }
    }
    return nullptr;
  }

  Assembler* pick_initial_assembler_to_clip(std::mt19937& rng) {
    // Pick an assembler uniformly at random.
    float total_weight = unclipped_assemblers.size();
    float random_weight = std::generate_canonical<float, 10>(rng) * total_weight;
    // Clearly we just want to get a random index and the pull that assembler, but we have a map, and we don't
    // expect to have to do this too often.
    for (auto& candidate_assembler : unclipped_assemblers) {
      random_weight -= 1.0f;
      if (random_weight <= 0.0f) {
        return candidate_assembler;
      }
    }
    return nullptr;
  }

  void maybe_clip_new_assembler(std::mt19937& rng) {
    if (std::generate_canonical<float, 10>(rng) < clip_rate) {
      Assembler* assembler = pick_assembler_to_clip(rng);
      if (assembler) clip_assembler(*assembler);
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
