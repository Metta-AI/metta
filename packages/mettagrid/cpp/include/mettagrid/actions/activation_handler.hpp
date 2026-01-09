#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "actions/activation_context.hpp"
#include "actions/activation_handler_config.hpp"
#include "actions/filters/filter.hpp"
#include "actions/mutations/mutation.hpp"

namespace mettagrid {

/**
 * ActivationHandler processes object activations through configurable
 * filter chains and mutation chains.
 *
 * Usage:
 *   1. Create handler with ActivationHandlerConfig
 *   2. Call try_apply() with actor and target
 *   3. Returns true if all filters passed and mutations were applied
 */
class ActivationHandler {
public:
  explicit ActivationHandler(const ActivationHandlerConfig& config);

  // Get handler name
  const std::string& name() const {
    return _name;
  }

  // Try to apply this handler to the given activation
  // Returns true if all filters passed and mutations were applied
  bool try_apply(HasInventory* actor, HasInventory* target);

  // Check if all filters pass without applying mutations
  bool check_filters(HasInventory* actor, HasInventory* target) const;

private:
  // Create a filter from its config
  static std::unique_ptr<Filter> create_filter(const FilterConfig& config);

  // Create a mutation from its config
  static std::unique_ptr<Mutation> create_mutation(const MutationConfig& config);

  std::string _name;
  std::vector<std::unique_ptr<Filter>> _filters;
  std::vector<std::unique_ptr<Mutation>> _mutations;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_
