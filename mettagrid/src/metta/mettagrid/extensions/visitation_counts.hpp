// extensions/visitation_counts.hpp
#ifndef EXTENSIONS_VISITATION_COUNTS_HPP_
#define EXTENSIONS_VISITATION_COUNTS_HPP_

#include <array>
#include <set>
#include <vector>

#include "extensions/mettagrid_extension.hpp"
#include "types.hpp"

// Forward declarations
class MettaGrid;
class Agent;

class VisitationCounts : public MettaGridExtension {
private:
  std::vector<std::vector<GridCoord>> _position_history_r;
  std::vector<std::vector<GridCoord>> _position_history_c;
  size_t _num_agents = 0;

  void addVisitationCountsToObservations(MettaGrid* env);

public:
  std::string getName() const override {
    return "visitation_counts";
  }

  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;
  py::dict getStats() const override;

private:
  std::array<unsigned int, 5> computeVisitationCounts(const MettaGrid* env, size_t agent_idx) const;
};

#endif  // EXTENSIONS_VISITATION_COUNTS_HPP_
