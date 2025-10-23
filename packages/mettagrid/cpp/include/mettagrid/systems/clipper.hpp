#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../../third_party/libfixmath/fix16.hpp"
#include "core/grid.hpp"
#include "objects/assembler.hpp"

class Clipper {
public:
  std::vector<std::shared_ptr<Recipe>> unclipping_recipes;
  // A map from assembler to its adjacent assemblers. This should be constant once computed.
  std::unordered_map<Assembler*, std::vector<Assembler*>> adjacent_assemblers;
  // A map of all assemblers to their current infection weight. This is the weight at which they'll be selected
  // for clipping (if currently unclipped).
  std::unordered_map<Assembler*, Fix16> assembler_infection_weight;
  // A set of assemblers that are adjacent to the set of clipped assemblers. Any unclipped assembler with
  // non-zero infection weight will be in this set.
  std::set<Assembler*> border_assemblers;
  // A set of all unclipped assemblers. Any assembler that is not in this set will be clipped.
  std::set<Assembler*> unclipped_assemblers;

  Fix16 length_scale;
  Fix16 cutoff_distance;
  Grid& grid;
  Fix16 clip_rate;
  std::mt19937 rng;

  Clipper(Grid& grid,
          std::vector<std::shared_ptr<Recipe>> recipe_ptrs,
          float length_scale,
          float cutoff_distance,
          float clip_rate,
          std::mt19937 rng_init)
      : unclipping_recipes(std::move(recipe_ptrs)),
        length_scale(Fix16(length_scale)),
        cutoff_distance(Fix16(cutoff_distance)),
        grid(grid),
        clip_rate(Fix16(clip_rate)),
        rng(std::move(rng_init)) {
    std::vector<Assembler*> starting_clipped_assemblers;
    for (size_t obj_id = 1; obj_id < grid.objects.size(); obj_id++) {
      auto* obj = grid.object(static_cast<GridObjectId>(obj_id));
      if (!obj) continue;
      if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
        // Skip clip-immune assemblers
        if (assembler->clip_immune) continue;

        assembler_infection_weight[assembler] = Fix16(0);
        unclipped_assemblers.insert(assembler);

        if (assembler->start_clipped) {
          // We'll actually do the clipping in a later step.
          starting_clipped_assemblers.push_back(assembler);
        }
      }
    }

    // Auto-calculate length_scale based on percolation theory if length_scale <= 0
    if (this->length_scale <= Fix16(0) && !assembler_infection_weight.empty()) {
      // Get grid dimensions
      GridCoord grid_width = grid.width;
      GridCoord grid_height = grid.height;
      Fix16 grid_size = Fix16((std::max(grid_width, grid_height)));

      // Calculate percolation-based length scale
      // The constant 4.51 is the critical percolation density λ_c for 2D continuum percolation,
      // empirically determined through Monte Carlo simulations (not analytically derivable).
      // Reference:
      // https://en.wikipedia.org/wiki/Percolation_threshold#Thresholds_for_2D_continuum_models
      // Note: Wikipedia provides a value of ~1.127 when defined in terms of radius,
      // We use diameter basis which becomes 4.51

      Fix16 PERCOLATION_CONSTANT = Fix16(4.51);
      Fix16 PI = Fix16(3.14159265359);
      Fix16 assembler_count = Fix16(static_cast<int16_t>(assembler_infection_weight.size()));
      Fix16 sqrt_count = assembler_count.sqrt();
      Fix16 sqrt_term = (PERCOLATION_CONSTANT / (Fix16(static_cast<int16_t>(4)) * PI)).sqrt();
      this->length_scale = (grid_size / sqrt_count) * sqrt_term;
    }
    // else: use the provided positive length_scale value as-is

    // Auto-calculate cutoff_distance if not provided (cutoff_distance <= 0)
    // At 3*length_scale, exp(-3) ≈ 0.05, so weights beyond this are negligible
    if (this->cutoff_distance <= Fix16(0)) {
      this->cutoff_distance = Fix16(static_cast<int16_t>(3)) * this->length_scale;
    }

    // This can be expensive, so only do it if the clipper is active. Note that having a Clipper with
    // a zero clip rate is value, since we can still clip assemblers that start clipped.
    if (this->clip_rate > Fix16(0)) {
      compute_adjacencies();
    }

    // Clip all starting clipped assemblers
    for (auto* assembler : starting_clipped_assemblers) {
      clip_assembler(*assembler);
    }
  }

  Fix16 infection_weight(Assembler& from, Assembler& to) const {
    Fix16 distance = this->distance(from, to);
    if (cutoff_distance > Fix16(0) && distance > cutoff_distance) return Fix16(0);
    // Using fix16_exp from libfixmath
    Fix16 neg_ratio = -(distance / length_scale);
    return Fix16(fix16_exp(neg_ratio.value));
  }

  // it's a little funky to use L2 distance here, since everywhere else we use L1 or Linf.
  Fix16 distance(Assembler& assembler_a, Assembler& assembler_b) const {
    GridLocation location_a = assembler_a.location;
    GridLocation location_b = assembler_b.location;
    // Cast to int to handle potential negative values from subtraction
    Fix16 dr = Fix16(static_cast<int16_t>(location_a.r - location_b.r));
    Fix16 dc = Fix16(static_cast<int16_t>(location_a.c - location_b.c));
    Fix16 sum_squares = (dr * dr) + (dc * dc);
    return sum_squares.sqrt();
  }

  void compute_adjacencies() {
    // Clear existing adjacencies
    adjacent_assemblers.clear();

    // Collect all assemblers and sort by x coordinate (column)
    std::vector<Assembler*> sorted_assemblers;
    for (auto& [assembler, _] : assembler_infection_weight) {
      sorted_assemblers.push_back(assembler);
    }

    // Sort assemblers by their column. In the future we could consider sorting by row instead when
    // height > width.
    std::sort(sorted_assemblers.begin(), sorted_assemblers.end(), [](Assembler* a, Assembler* b) {
      return a->location.c < b->location.c;
    });

    // For each assembler, find adjacent assemblers within cutoff_distance
    for (size_t i = 0; i < sorted_assemblers.size(); ++i) {
      Assembler* assembler_a = sorted_assemblers[i];
      GridCoord a_x = assembler_a->location.c;

      // Check assemblers with x coordinates within cutoff_distance
      for (size_t j = i + 1; j < sorted_assemblers.size(); ++j) {
        Assembler* assembler_b = sorted_assemblers[j];
        GridCoord b_x = assembler_b->location.c;

        // If x difference exceeds cutoff_distance, no need to check further
        if (Fix16(static_cast<int16_t>(b_x - a_x)) > cutoff_distance) {
          break;
        }

        // Calculate actual distance
        Fix16 dist = distance(*assembler_a, *assembler_b);

        // If within cutoff_distance, they are adjacent
        if (dist <= cutoff_distance) {
          // Add both directions of connection
          adjacent_assemblers[assembler_a].push_back(assembler_b);
          adjacent_assemblers[assembler_b].push_back(assembler_a);
        }
      }
    }
  }

  void clip_assembler(Assembler& to_infect) {
    assert(!to_infect.is_clipped);
    // Update infection weights only for adjacent assemblers
    for (auto* adjacent : adjacent_assemblers[&to_infect]) {
      // Track this even for clipped assemblers, so we'll have an accurate number if they become unclipped.
      assembler_infection_weight[adjacent] += infection_weight(to_infect, *adjacent);
      if (adjacent->is_clipped) continue;
      border_assemblers.insert(adjacent);
    }
    border_assemblers.erase(&to_infect);
    unclipped_assemblers.erase(&to_infect);

    // Randomly select one recipe from the list
    std::uniform_int_distribution<size_t> dist(0, unclipping_recipes.size() - 1);
    size_t selected_idx = dist(rng);
    std::shared_ptr<Recipe> selected_recipe = unclipping_recipes[selected_idx];

    // Create a map with the selected recipe for all possible vibes
    // For now, we'll just map vibe 0 to the recipe (no glyphs case)
    // TODO: This might need adjustment based on how unclipping should work with glyphs
    std::unordered_map<uint64_t, std::shared_ptr<Recipe>> unclip_recipes;
    unclip_recipes[0] = selected_recipe;
    to_infect.become_clipped(unclip_recipes, this);
  }

  void on_unclip_assembler(Assembler& to_unclip) {
    // Update infection weights only for adjacent assemblers
    for (auto* adjacent : adjacent_assemblers[&to_unclip]) {
      assembler_infection_weight[adjacent] -= infection_weight(to_unclip, *adjacent);
    }
    unclipped_assemblers.insert(&to_unclip);
    if (assembler_infection_weight[&to_unclip] > Fix16(0)) {
      border_assemblers.insert(&to_unclip);
    }
  }

  Assembler* pick_assembler_to_clip() {
    Fix16 total_weight = Fix16(0);
    for (auto& candidate_assembler : border_assemblers) {
      total_weight += assembler_infection_weight.at(candidate_assembler);
    }
    Fix16 random_weight = Fix16(std::generate_canonical<float, 10>(rng)) * total_weight;
    for (auto& candidate_assembler : border_assemblers) {
      random_weight -= assembler_infection_weight.at(candidate_assembler);
      if (random_weight <= Fix16(0)) {
        return candidate_assembler;
      }
    }
    // We failed to find an assembler to clip. That _should_ be because there are no border assemblers. But maybe
    // it's a floating point error. In any case, pick a random assembler from the unclipped assemblers.
    if (!unclipped_assemblers.empty()) {
      auto it = unclipped_assemblers.begin();
      std::advance(it, std::uniform_int_distribution<size_t>(0, unclipped_assemblers.size() - 1)(rng));
      return *it;
    }

    return nullptr;
  }

  void maybe_clip_new_assembler() {
    if (Fix16(std::generate_canonical<float, 10>(rng)) < clip_rate) {
      Assembler* assembler = pick_assembler_to_clip();
      if (assembler) {
        clip_assembler(*assembler);
      }
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
