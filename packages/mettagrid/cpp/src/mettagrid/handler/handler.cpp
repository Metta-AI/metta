#include "handler/handler.hpp"

namespace mettagrid {

Handler::Handler(const HandlerConfig& config) : _name(config.name) {
  // Create filters from config
  for (const auto& filter_config : config.filters) {
    auto filter = create_filter(filter_config);
    if (filter) {
      _filters.push_back(std::move(filter));
    }
  }

  // Create mutations from config
  for (const auto& mutation_config : config.mutations) {
    auto mutation = create_mutation(mutation_config);
    if (mutation) {
      _mutations.push_back(std::move(mutation));
    }
  }
}

bool Handler::try_apply(HasInventory* actor, HasInventory* target) {
  if (!check_filters(actor, target)) {
    return false;
  }

  HandlerContext ctx(actor, target);
  for (auto& mutation : _mutations) {
    mutation->apply(ctx);
  }

  return true;
}

bool Handler::check_filters(HasInventory* actor, HasInventory* target) const {
  HandlerContext ctx(actor, target);

  for (const auto& filter : _filters) {
    if (!filter->passes(ctx)) {
      return false;
    }
  }

  return true;
}

// By using a visitor pattern here, we can keep the configs and the creation of the filters/mutations separate.
std::unique_ptr<Filter> Handler::create_filter(const FilterConfig& config) {
  return std::visit(
      [](auto&& cfg) -> std::unique_ptr<Filter> {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, VibeFilterConfig>) {
          return std::make_unique<VibeFilter>(cfg);
        } else if constexpr (std::is_same_v<T, ResourceFilterConfig>) {
          return std::make_unique<ResourceFilter>(cfg);
        } else if constexpr (std::is_same_v<T, AlignmentFilterConfig>) {
          return std::make_unique<AlignmentFilter>(cfg);
        } else if constexpr (std::is_same_v<T, TagFilterConfig>) {
          return std::make_unique<TagFilter>(cfg);
        } else {
          return nullptr;
        }
      },
      config);
}

std::unique_ptr<Mutation> Handler::create_mutation(const MutationConfig& config) {
  return std::visit(
      [](auto&& cfg) -> std::unique_ptr<Mutation> {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, ResourceDeltaMutationConfig>) {
          return std::make_unique<ResourceDeltaMutation>(cfg);
        } else if constexpr (std::is_same_v<T, ResourceTransferMutationConfig>) {
          return std::make_unique<ResourceTransferMutation>(cfg);
        } else if constexpr (std::is_same_v<T, AlignmentMutationConfig>) {
          return std::make_unique<AlignmentMutation>(cfg);
        } else if constexpr (std::is_same_v<T, FreezeMutationConfig>) {
          return std::make_unique<FreezeMutation>(cfg);
        } else if constexpr (std::is_same_v<T, ClearInventoryMutationConfig>) {
          return std::make_unique<ClearInventoryMutation>(cfg);
        } else if constexpr (std::is_same_v<T, AttackMutationConfig>) {
          return std::make_unique<AttackMutation>(cfg);
        } else {
          return nullptr;
        }
      },
      config);
}

}  // namespace mettagrid
