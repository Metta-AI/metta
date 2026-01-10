#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_HPP_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "handler/filters/filter.hpp"
#include "handler/handler_config.hpp"
#include "handler/handler_context.hpp"
#include "handler/mutations/mutation.hpp"

namespace mettagrid {

/**
 * Handler processes object activations through configurable
 * filter chains and mutation chains.
 *
 * Usage:
 *   1. Create handler with HandlerConfig
 *   2. Call try_apply() with actor and target
 *   3. Returns true if all filters passed and mutations were applied
 */
class Handler {
public:
  explicit Handler(const HandlerConfig& config);

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

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_HPP_
