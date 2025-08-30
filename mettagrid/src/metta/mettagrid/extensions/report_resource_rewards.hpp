// extensions/report_resource_rewards.hpp
#ifndef EXTENSIONS_REPORT_RESOURCE_REWARDS_HPP_
#define EXTENSIONS_REPORT_RESOURCE_REWARDS_HPP_

#include <vector>

#include "extensions/mettagrid_extension.hpp"

class ReportResourceRewards : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override;
  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;

  std::string getName() const override {
    return "report_resource_rewards";
  }

  ExtensionStats getStats() const override;

private:
  ObservationType _resource_rewards_feature;
  size_t _num_agents;

  void addResourceRewardsToObservations(MettaGrid* env);
};

#endif  // EXTENSIONS_REPORT_RESOURCE_REWARDS_HPP_
