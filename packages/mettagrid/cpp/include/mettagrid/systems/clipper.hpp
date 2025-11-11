#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "core/grid.hpp"
#include "objects/assembler.hpp"
#include "objects/protocol.hpp"

class Clipper {
public:
  std::vector<std::shared_ptr<Protocol>> unclipping_protocols;
  // A map from assembler to its adjacent assemblers. This should be constant once computed.
  std::unordered_map<Assembler*, std::vector<Assembler*>> adjacent_assemblers;
  // A map of all assemblers to their current infection weight. This is the weight at which they'll be selected
  // for clipping (if currently unclipped).
  std::unordered_map<Assembler*, uint32_t> assembler_infection_weight;
  // A set of assemblers that are adjacent to the set of clipped assemblers. Any unclipped assembler with
  // non-zero infection weight will be in this set.
  std::set<Assembler*> border_assemblers;
  // A set of all unclipped assemblers. Any assembler that is not in this set will be clipped.
  std::set<Assembler*> unclipped_assemblers;

  float length_scale;
  float cutoff_distance;
  Grid& grid;
  uint32_t clip_period;
  std::mt19937 rng;

  Clipper(Grid& grid,
          std::vector<std::shared_ptr<Protocol>> protocol_ptrs,
          float length_scale,
          float cutoff_distance,
          uint32_t clip_period,
          std::mt19937 rng_init)
      : unclipping_protocols(std::move(protocol_ptrs)),
        length_scale(length_scale),
        cutoff_distance(cutoff_distance),
        grid(grid),
        clip_period(clip_period),
        rng(std::move(rng_init)) {
    std::vector<Assembler*> starting_clipped_assemblers;
    for (size_t obj_id = 1; obj_id < grid.objects.size(); obj_id++) {
      auto* obj = grid.object(static_cast<GridObjectId>(obj_id));
      if (!obj) continue;
      if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
        // Skip clip-immune assemblers
        if (assembler->clip_immune) continue;

        assembler_infection_weight[assembler] = 0;
        unclipped_assemblers.insert(assembler);

        if (assembler->start_clipped) {
          // We'll actually do the clipping in a later step.
          starting_clipped_assemblers.push_back(assembler);
        }
      }
    }

    // Auto-calculate length_scale based on percolation theory if length_scale <= 0
    if (length_scale <= 0.0f && !assembler_infection_weight.empty()) {
      // Get grid dimensions
      GridCoord grid_width = grid.width;
      GridCoord grid_height = grid.height;
      float grid_size = static_cast<float>(std::max(grid_width, grid_height));

      // Calculate percolation-based length scale
      // The constant 4.51 is the critical percolation density λ_c for 2D continuum percolation,
      // empirically determined through Monte Carlo simulations (not analytically derivable).
      // Reference:
      // https://en.wikipedia.org/wiki/Percolation_threshold#Thresholds_for_2D_continuum_models
      // Note: Wikipedia provides a value of ~1.127 when defined in terms of radius,
      // We use diameter basis which becomes 4.51

      constexpr float PERCOLATION_CONSTANT = 4.51f;
      this->length_scale = (grid_size / std::sqrt(static_cast<float>(assembler_infection_weight.size()))) *
                           std::sqrt(PERCOLATION_CONSTANT / (4.0f * std::numbers::pi_v<float>));
    }
    // else: use the provided positive length_scale value as-is

    // Auto-calculate cutoff_distance if not provided (cutoff_distance <= 0)
    // At 3*length_scale, exp(-3) ≈ 0.05, so weights beyond this are negligible
    if (cutoff_distance <= 0.0f) {
      this->cutoff_distance = 3.0f * this->length_scale;
    }

    // This can be expensive, so only do it if the clipper is active. Note that having a Clipper with
    // a zero clip rate is value, since we can still clip assemblers that start clipped.
    if (clip_period > 0) {
      compute_adjacencies();
    }

    // Clip all starting clipped assemblers
    for (auto* assembler : starting_clipped_assemblers) {
      clip_assembler(*assembler);
    }
  }

  uint32_t infection_weight(Assembler& from, Assembler& to) const {
    float distance = this->distance(from, to);
    if (cutoff_distance > 0.0f && distance > cutoff_distance) return 0;
    // The * 1000000.0f is a hack to get us from float to uint32_t, as we move to get rid of floats. We only care
    // about relative weights, so scaling them linearly (like this) doesn't matter.
    return static_cast<uint32_t>(std::exp(-distance / length_scale) * 1000000.0f);
  }

  // it's a little funky to use L2 distance here, since everywhere else we use L1 or Linf.
  float distance(Assembler& assembler_a, Assembler& assembler_b) const {
    GridLocation location_a = assembler_a.location;
    GridLocation location_b = assembler_b.location;
    return std::sqrt(std::pow(location_a.r - location_b.r, 2) + std::pow(location_a.c - location_b.c, 2));
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
        if (b_x - a_x > cutoff_distance) {
          break;
        }

        // Calculate actual distance
        float dist = distance(*assembler_a, *assembler_b);

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
      assembler_infection_weight.at(adjacent) += infection_weight(to_infect, *adjacent);
      if (adjacent->is_clipped) continue;
      border_assemblers.insert(adjacent);
    }
    border_assemblers.erase(&to_infect);
    unclipped_assemblers.erase(&to_infect);

    // Randomly select one protocol from the list
    std::uniform_int_distribution<size_t> dist(0, unclipping_protocols.size() - 1);
    size_t selected_idx = dist(rng);
    std::shared_ptr<Protocol> selected_protocol = unclipping_protocols[selected_idx];

    std::vector<std::shared_ptr<Protocol>> unclip_protocols;
    unclip_protocols.push_back(selected_protocol);
    to_infect.become_clipped(unclip_protocols, this);
  }

  void on_unclip_assembler(Assembler& to_unclip) {
    // Update infection weights only for adjacent assemblers
    for (auto* adjacent : adjacent_assemblers[&to_unclip]) {
      assembler_infection_weight.at(adjacent) -= infection_weight(to_unclip, *adjacent);
    }
    unclipped_assemblers.insert(&to_unclip);
    if (assembler_infection_weight.at(&to_unclip) > 0) {
      border_assemblers.insert(&to_unclip);
    }
  }

  Assembler* pick_assembler_to_clip() {
    if (unclipped_assemblers.size() == 0) {
      return nullptr;
    }
    uint32_t total_weight = 0;
    for (auto& candidate_assembler : border_assemblers) {
      total_weight += assembler_infection_weight.at(candidate_assembler);
    }
    if (total_weight == 0) {
      // If there are no border assemblers, pick a random assembler from the unclipped assemblers.
      auto it = unclipped_assemblers.begin();
      std::advance(it, std::uniform_int_distribution<size_t>(0, unclipped_assemblers.size() - 1)(rng));
      return *it;
    }
    uint32_t random_weight = std::uniform_int_distribution<>(1, total_weight)(rng);
    for (auto& candidate_assembler : border_assemblers) {
      if (random_weight <= assembler_infection_weight.at(candidate_assembler)) {
        return candidate_assembler;
      }
      random_weight -= assembler_infection_weight.at(candidate_assembler);
    }
    throw std::runtime_error("Failed to pick an assembler to clip");
  }

  void maybe_clip_new_assembler() {
    if (std::uniform_int_distribution<uint32_t>(1, clip_period)(rng) == 1) {
      Assembler* assembler = pick_assembler_to_clip();
      if (assembler) {
        clip_assembler(*assembler);
      }
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
