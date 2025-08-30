// extensions/resource_rewards.hpp
#ifndef EXTENSIONS_RESOURCE_REWARDS_HPP_
#define EXTENSIONS_RESOURCE_REWARDS_HPP_

#include <vector>

#include "extensions/mettagrid_extension.hpp"

class ResourceRewards : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override;
  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;

  std::string getName() const override {
    return "resource_rewards";
  }

  ExtensionStats getStats() const override;

private:
  ObservationType _resource_rewards_feature;
  size_t _num_agents;
  size_t _num_inventory_items;

  // Store packed resource rewards for each agent
  std::vector<uint8_t> _resource_rewards;

  void updateResourceRewards(MettaGrid* env);
  void addResourceRewardsToObservations(MettaGrid* env);
};

#endif  // EXTENSIONS_RESOURCE_REWARDS_HPP_
