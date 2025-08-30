// extensions/episode_completion.hpp
#ifndef EXTENSIONS_EPISODE_COMPLETION_HPP_
#define EXTENSIONS_EPISODE_COMPLETION_HPP_

#include "extensions/mettagrid_extension.hpp"

class EpisodeCompletion : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override;
  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;

  std::string getName() const override {
    return "episode_completion";
  }

  ExtensionStats getStats() const override;

private:
  ObservationType _episode_completion_pct_feature;
  size_t _max_steps;
  size_t _current_step;
  size_t _num_agents;

  void addEpisodeCompletionToObservations(MettaGrid* env);
};

#endif  // EXTENSIONS_EPISODE_COMPLETION_HPP_
