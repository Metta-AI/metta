#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
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

  GridCoord length_scale;
  // This is the cutoff distance in units of length_scale. Assemblers further from this will be considered disconnected.
  uint32_t scaled_cutoff_distance;
  Grid& grid;
  uint32_t clip_period;
  std::mt19937 rng;

  Clipper(Grid& grid,
          std::vector<std::shared_ptr<Protocol>> protocol_ptrs,
          GridCoord length_scale,
          uint32_t scaled_cutoff_distance,
          uint32_t clip_period,
          std::mt19937 rng_init)
      : unclipping_protocols(std::move(protocol_ptrs)),
        length_scale(length_scale),
        scaled_cutoff_distance(scaled_cutoff_distance),
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
    if (length_scale <= 0u && !assembler_infection_weight.empty()) {
      // Get grid dimensions
      GridCoord grid_width = grid.width;
      GridCoord grid_height = grid.height;
      // 1 / density. The sparser the grid, the larger the length_scale.
      uint32_t sparsity = (grid_width * grid_height) / assembler_infection_weight.size();
      // Take an approximate square root to get a linearized sparsity.
      int32_t root_sparsity = 1;
      // This is a guess for something close to the root for most maps we have.
      int32_t root_sparsity_next = 10;
      // We use a for loop because it makes it very clear that this loop terminates. I _think_ that the
      // Newton's method method will always converge, but I haven't proven it.
      for (int32_t i = 0; i < 10; i++) {
        if (root_sparsity_next == root_sparsity || root_sparsity == 0) {
          break;
        }
        root_sparsity = root_sparsity_next;
        // Do a step of Newton's method. The +(2 * root_sparsity - 2) is to help with convergence. This causes us to
        // converge to the ceiling of the root.
        root_sparsity_next = (sparsity + root_sparsity * root_sparsity + (2 * root_sparsity - 2)) / (2 * root_sparsity);
      }
      // A more disciplined approach would be use percolation theory to calculate the length_scale.
      // E.g., using the L_inf norm for distance, we should use an n_c of 1.0988 (since L_inf means that the shapes
      // we're using are aligned squares). In particular, this would let us pick a length_scale that's just at
      // the critical threshold for percolation. But in practice, we're then using a scaled_cutoff_distance of
      // something like 3, which would mean that we're well into the supercritical regime. So we don't worry about it.
      // Instead, we really just want to make sure that we're doing something reasonable with scaling as the
      // sparsity changes.

      // The division by 2 basically to move us from diameter to radius. Or to move us from "linear sparsity" to
      // "average distance".
      this->length_scale = static_cast<GridCoord>(std::max(root_sparsity / 2, 1));
    }
    // else: use the provided positive length_scale value as-is

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
    uint32_t scaled_distance = static_cast<uint32_t>(this->distance(from, to)) / length_scale;
    if (scaled_distance > scaled_cutoff_distance) return 0;
    // A cheap rendition of c * exp(-scaled_distance). Note that the value of c doesn't matter, since we only care
    // about relative weights. So we set c to 2**scaled_cutoff_distance, since that lets us distinguish between
    // values of scaled_distance up to scaled_cutoff_distance.
    return 1 << (scaled_cutoff_distance - scaled_distance);
  }

  // Use L_inf distance here, since it's more natural for our grid world that L_2. L_1 could also be a reasonable
  // contender.
  GridCoord distance(Assembler& assembler_a, Assembler& assembler_b) const {
    GridLocation location_a = assembler_a.location;
    GridLocation location_b = assembler_b.location;
    return std::max(std::abs(location_a.r - location_b.r), std::abs(location_a.c - location_b.c));
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

    // For each assembler, find adjacent assemblers within scaled_cutoff_distance of each other.
    for (size_t i = 0; i < sorted_assemblers.size(); ++i) {
      Assembler* assembler_a = sorted_assemblers[i];
      GridCoord a_x = assembler_a->location.c;

      // Check assemblers with x coordinates within cutoff_distance
      for (size_t j = i + 1; j < sorted_assemblers.size(); ++j) {
        Assembler* assembler_b = sorted_assemblers[j];
        GridCoord b_x = assembler_b->location.c;

        // If x difference exceeds cutoff_distance, no need to check further
        if (b_x - a_x > scaled_cutoff_distance * length_scale) {
          break;
        }

        // Calculate actual distance
        GridCoord dist = distance(*assembler_a, *assembler_b);

        // If within cutoff_distance, they are adjacent
        if (dist <= scaled_cutoff_distance * length_scale) {
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
