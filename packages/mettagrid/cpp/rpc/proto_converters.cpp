#include "rpc/proto_converters.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "actions/resource_mod.hpp"
#include "objects/agent_config.hpp"
#include "objects/assembler_config.hpp"
#include "objects/chest_config.hpp"
#include "objects/converter_config.hpp"
#include "objects/recipe.hpp"
#include "objects/wall.hpp"
#include "systems/clipper_config.hpp"

namespace mettagrid::rpc {
namespace {

using RequiredMap = std::unordered_map<InventoryItem, InventoryQuantity>;
using ConsumedMap = std::unordered_map<InventoryItem, InventoryProbability>;

RequiredMap ConvertQuantityMap(const ::google::protobuf::Map<uint32_t, uint32_t>& proto_map) {
  RequiredMap result;
  for (const auto& [item, quantity] : proto_map) {
    result.emplace(static_cast<InventoryItem>(item), static_cast<InventoryQuantity>(quantity));
  }
  return result;
}

ConsumedMap ConvertProbabilityMap(const ::google::protobuf::Map<uint32_t, float>& proto_map) {
  ConsumedMap result;
  for (const auto& [item, value] : proto_map) {
    result.emplace(static_cast<InventoryItem>(item), static_cast<InventoryProbability>(value));
  }
  return result;
}

std::pair<RequiredMap, ConsumedMap> ConvertActionResources(const v1::ActionResources* resources) {
  if (resources == nullptr) {
    return {};
  }
  return {ConvertQuantityMap(resources->required()), ConvertProbabilityMap(resources->consumed())};
}

std::shared_ptr<ActionConfig> MakeBaseActionConfig(const v1::ActionResources* resources) {
  auto [required, consumed] = ConvertActionResources(resources);
  return std::make_shared<ActionConfig>(required, consumed);
}

std::shared_ptr<ActionConfig> ConvertActionDefinition(const v1::ActionDefinition& proto_action) {
  switch (proto_action.type()) {
    case v1::ActionDefinition::ACTION_NOOP:
      return MakeBaseActionConfig(proto_action.has_noop() ? &proto_action.noop().base() : nullptr);
    case v1::ActionDefinition::ACTION_MOVE:
      return MakeBaseActionConfig(proto_action.has_move() ? &proto_action.move().base() : nullptr);
    case v1::ActionDefinition::ACTION_ROTATE:
      return MakeBaseActionConfig(proto_action.has_rotate() ? &proto_action.rotate().base() : nullptr);
    case v1::ActionDefinition::ACTION_SWAP:
      return MakeBaseActionConfig(proto_action.has_swap() ? &proto_action.swap().base() : nullptr);
    case v1::ActionDefinition::ACTION_PUT_ITEMS:
      return MakeBaseActionConfig(proto_action.has_put_items() ? &proto_action.put_items().base() : nullptr);
    case v1::ActionDefinition::ACTION_GET_ITEMS:
      return MakeBaseActionConfig(proto_action.has_get_items() ? &proto_action.get_items().base() : nullptr);
    case v1::ActionDefinition::ACTION_ATTACK: {
      const auto* attack_proto = proto_action.has_attack() ? &proto_action.attack() : nullptr;
      auto [required, consumed] = ConvertActionResources(attack_proto ? &attack_proto->base() : nullptr);
      RequiredMap defence = attack_proto ? ConvertQuantityMap(attack_proto->defense()) : RequiredMap{};
      return std::make_shared<AttackActionConfig>(required, consumed, defence);
    }
    case v1::ActionDefinition::ACTION_CHANGE_GLYPH: {
      const auto* glyph_proto = proto_action.has_change_glyph() ? &proto_action.change_glyph() : nullptr;
      if (!glyph_proto) {
        throw std::runtime_error("change_glyph action requires configuration");
      }
      auto [required, consumed] = ConvertActionResources(&glyph_proto->base());
      return std::make_shared<ChangeGlyphActionConfig>(
          required, consumed, static_cast<ObservationType>(glyph_proto->num_glyphs()));
    }
    case v1::ActionDefinition::ACTION_RESOURCE_MOD: {
      const auto* mod_proto = proto_action.has_resource_mod() ? &proto_action.resource_mod() : nullptr;
      if (!mod_proto) {
        throw std::runtime_error("resource_mod action requires configuration");
      }
      auto [required, consumed] = ConvertActionResources(&mod_proto->base());
      ConsumedMap modifies = ConvertProbabilityMap(mod_proto->modifies());
      return std::make_shared<ResourceModConfig>(required,
                                                 consumed,
                                                 modifies,
                                                 static_cast<GridCoord>(mod_proto->agent_radius()),
                                                 static_cast<GridCoord>(mod_proto->converter_radius()),
                                                 mod_proto->scales());
    }
    default:
      throw std::runtime_error("Unsupported action type in protobuf definition");
  }
}

std::vector<int> ConvertTagIds(const ::google::protobuf::RepeatedField<uint32_t>& ids) {
  std::vector<int> tag_ids;
  tag_ids.reserve(ids.size());
  for (auto id : ids) {
    tag_ids.push_back(static_cast<int>(id));
  }
  return tag_ids;
}

InventoryConfig ConvertInventoryConfig(const v1::InventoryConfig& proto_cfg) {
  std::vector<std::pair<std::vector<InventoryItem>, InventoryQuantity>> limits;
  limits.reserve(proto_cfg.limits_size());
  for (const auto& limit_proto : proto_cfg.limits()) {
    std::vector<InventoryItem> items;
    items.reserve(limit_proto.items_size());
    for (auto item : limit_proto.items()) {
      items.push_back(static_cast<InventoryItem>(item));
    }
    limits.emplace_back(std::move(items), static_cast<InventoryQuantity>(limit_proto.max_quantity()));
  }
  return InventoryConfig(limits);
}

std::shared_ptr<GridObjectConfig> ConvertAgentConfig(const v1::AgentConfig& proto_cfg) {
  InventoryConfig inventory_cfg;
  if (proto_cfg.has_inventory()) {
    inventory_cfg = ConvertInventoryConfig(proto_cfg.inventory());
  }

  auto initial_inventory = ConvertQuantityMap(proto_cfg.initial_inventory());
  auto stat_rewards =
      std::unordered_map<std::string, RewardType>(proto_cfg.stat_rewards().begin(), proto_cfg.stat_rewards().end());
  auto stat_reward_max = std::unordered_map<std::string, RewardType>(proto_cfg.stat_reward_max().begin(),
                                                                     proto_cfg.stat_reward_max().end());

  std::vector<InventoryItem> soul_bound;
  soul_bound.reserve(proto_cfg.soul_bound_resources_size());
  for (auto item : proto_cfg.soul_bound_resources()) {
    soul_bound.push_back(static_cast<InventoryItem>(item));
  }

  std::vector<InventoryItem> shareable;
  shareable.reserve(proto_cfg.shareable_resources_size());
  for (auto item : proto_cfg.shareable_resources()) {
    shareable.push_back(static_cast<InventoryItem>(item));
  }

  auto regen_amounts = ConvertQuantityMap(proto_cfg.inventory_regen_amounts());

  return std::make_shared<AgentConfig>(static_cast<TypeId>(proto_cfg.type_id()),
                                       proto_cfg.type_name(),
                                       static_cast<unsigned char>(proto_cfg.group_id()),
                                       proto_cfg.group_name(),
                                       static_cast<unsigned char>(proto_cfg.freeze_duration()),
                                       proto_cfg.action_failure_penalty(),
                                       inventory_cfg,
                                       stat_rewards,
                                       stat_reward_max,
                                       proto_cfg.group_reward_pct(),
                                       initial_inventory,
                                       ConvertTagIds(proto_cfg.tag_ids()),
                                       soul_bound,
                                       shareable,
                                       regen_amounts);
}

std::shared_ptr<GridObjectConfig> ConvertWallConfig(const v1::WallConfig& proto_cfg) {
  return std::make_shared<WallConfig>(
      static_cast<TypeId>(proto_cfg.type_id()), proto_cfg.type_name(), proto_cfg.swappable(), ConvertTagIds(proto_cfg.tag_ids()));
}

std::vector<std::shared_ptr<Recipe>> ConvertRecipesVector(const ::google::protobuf::RepeatedPtrField<v1::ConverterRecipe>& recipes) {
  std::vector<std::shared_ptr<Recipe>> result;
  result.reserve(recipes.size());
  for (const auto& recipe_proto : recipes) {
    auto inputs = ConvertQuantityMap(recipe_proto.inputs());
    auto outputs = ConvertQuantityMap(recipe_proto.outputs());
    result.push_back(std::make_shared<Recipe>(inputs, outputs, static_cast<unsigned short>(recipe_proto.cooldown())));
  }
  return result;
}

std::unordered_map<uint64_t, std::shared_ptr<Recipe>> ConvertRecipesMap(const ::google::protobuf::RepeatedPtrField<v1::ConverterRecipe>& recipes) {
  std::unordered_map<uint64_t, std::shared_ptr<Recipe>> result;
  // Use sequential indices as temporary keys since protobuf doesn't include vibe information
  // In the full system, vibes are computed from glyph patterns in the Python config
  uint64_t index = 0;
  for (const auto& recipe_proto : recipes) {
    auto inputs = ConvertQuantityMap(recipe_proto.inputs());
    auto outputs = ConvertQuantityMap(recipe_proto.outputs());
    result.emplace(index++, std::make_shared<Recipe>(inputs, outputs, static_cast<unsigned short>(recipe_proto.cooldown())));
  }
  return result;
}

std::shared_ptr<GridObjectConfig> ConvertAssemblerConfig(const v1::AssemblerConfig& proto_cfg) {
  auto config = std::make_shared<AssemblerConfig>(
      static_cast<TypeId>(proto_cfg.type_id()), proto_cfg.type_name(), ConvertTagIds(proto_cfg.tag_ids()));
  config->recipes = ConvertRecipesMap(proto_cfg.recipes());
  config->allow_partial_usage = proto_cfg.allow_partial_usage();
  config->max_uses = proto_cfg.max_uses();
  config->exhaustion = proto_cfg.exhaustion();
  config->clip_immune = proto_cfg.clip_immune();
  config->start_clipped = proto_cfg.start_clipped();
  return config;
}

std::shared_ptr<GridObjectConfig> ConvertConverterConfig(const v1::ConverterConfig& proto_cfg) {
  auto input_resources = ConvertQuantityMap(proto_cfg.input_resources());
  auto output_resources = ConvertQuantityMap(proto_cfg.output_resources());
  short max_output = static_cast<short>(proto_cfg.max_output());
  short max_conversions = static_cast<short>(proto_cfg.max_conversions());
  unsigned short conversion_ticks = static_cast<unsigned short>(proto_cfg.conversion_ticks());

  std::vector<unsigned short> cooldown_schedule;
  cooldown_schedule.reserve(proto_cfg.cooldown_schedule_size());
  for (auto value : proto_cfg.cooldown_schedule()) {
    cooldown_schedule.push_back(static_cast<unsigned short>(value));
  }

  auto config = std::make_shared<ConverterConfig>(static_cast<TypeId>(proto_cfg.type_id()),
                                                  proto_cfg.type_name(),
                                                  input_resources,
                                                  output_resources,
                                                  max_output,
                                                  max_conversions,
                                                  conversion_ticks,
                                                  cooldown_schedule,
                                                  static_cast<unsigned char>(proto_cfg.initial_resource_count()),
                                                  proto_cfg.recipe_details_obs(),
                                                  ConvertTagIds(proto_cfg.tag_ids()));
  return config;
}

std::shared_ptr<GridObjectConfig> ConvertChestConfig(const v1::ChestConfig& proto_cfg) {
  std::unordered_map<int, int> deltas;
  for (const auto& [position, delta] : proto_cfg.position_deltas()) {
    deltas.emplace(static_cast<int>(position), delta);
  }

  return std::make_shared<ChestConfig>(static_cast<TypeId>(proto_cfg.type_id()),
                                       proto_cfg.type_name(),
                                       static_cast<InventoryItem>(proto_cfg.resource_type()),
                                       deltas,
                                       proto_cfg.initial_inventory(),
                                       proto_cfg.max_inventory(),
                                       ConvertTagIds(proto_cfg.tag_ids()));
}

std::shared_ptr<GridObjectConfig> ConvertObjectDefinition(const v1::ObjectDefinition& proto_object) {
  switch (proto_object.config_case()) {
    case v1::ObjectDefinition::kAgent:
      return ConvertAgentConfig(proto_object.agent());
    case v1::ObjectDefinition::kWall:
      return ConvertWallConfig(proto_object.wall());
    case v1::ObjectDefinition::kConverter:
      return ConvertConverterConfig(proto_object.converter());
    case v1::ObjectDefinition::kAssembler:
      return ConvertAssemblerConfig(proto_object.assembler());
    case v1::ObjectDefinition::kChest:
      return ConvertChestConfig(proto_object.chest());
    case v1::ObjectDefinition::CONFIG_NOT_SET:
      throw std::runtime_error("Object definition missing configuration");
  }
  throw std::runtime_error("Unsupported object definition");
}

GlobalObsConfig ConvertGlobalObs(const v1::GlobalObsConfig& proto_cfg) {
  GlobalObsConfig cfg;
  cfg.episode_completion_pct = proto_cfg.episode_completion_pct();
  cfg.last_action = proto_cfg.last_action();
  cfg.last_reward = proto_cfg.last_reward();
  cfg.visitation_counts = proto_cfg.visitation_counts();
  return cfg;
}

std::shared_ptr<ClipperConfig> ConvertClipperConfig(const v1::ClipperConfig& proto_cfg) {
  auto recipes = ConvertRecipesVector(proto_cfg.unclipping_recipes());
  return std::make_shared<ClipperConfig>(std::move(recipes),
                                         proto_cfg.length_scale(),
                                         proto_cfg.cutoff_distance(),
                                         proto_cfg.clip_rate());
}

}  // namespace

GameConfig ConvertGameConfig(const v1::GameConfig& proto_cfg) {
  std::vector<std::string> resource_names(proto_cfg.resource_names().begin(), proto_cfg.resource_names().end());

  std::vector<std::pair<std::string, std::shared_ptr<ActionConfig>>> actions;
  actions.reserve(proto_cfg.actions_size());
  for (const auto& action_proto : proto_cfg.actions()) {
    actions.emplace_back(action_proto.name(), ConvertActionDefinition(action_proto));
  }

  std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  objects.reserve(proto_cfg.objects_size());
  for (const auto& object_proto : proto_cfg.objects()) {
    objects.emplace(object_proto.name(), ConvertObjectDefinition(object_proto));
  }

  std::unordered_map<int, std::string> tag_id_map;
  for (const auto& [tag_id, tag_name] : proto_cfg.tag_id_map()) {
    tag_id_map.emplace(static_cast<int>(tag_id), tag_name);
  }

  std::unordered_map<std::string, float> reward_estimates(proto_cfg.reward_estimates().begin(),
                                                          proto_cfg.reward_estimates().end());

  std::shared_ptr<ClipperConfig> clipper;
  if (proto_cfg.has_clipper()) {
    clipper = ConvertClipperConfig(proto_cfg.clipper());
  }

  GlobalObsConfig global_obs;
  if (proto_cfg.has_global_obs()) {
    global_obs = ConvertGlobalObs(proto_cfg.global_obs());
  }

  GameConfig config(proto_cfg.num_agents(),
                    proto_cfg.max_steps(),
                    proto_cfg.episode_truncates(),
                    static_cast<ObservationCoord>(proto_cfg.obs_width()),
                    static_cast<ObservationCoord>(proto_cfg.obs_height()),
                    resource_names,
                    proto_cfg.num_observation_tokens(),
                    global_obs,
                    actions,
                    objects,
                    proto_cfg.resource_loss_prob(),
                    tag_id_map,
                    proto_cfg.track_movement_metrics(),
                    proto_cfg.recipe_details_obs(),
                    proto_cfg.allow_diagonals(),
                    reward_estimates,
                    proto_cfg.inventory_regen_interval(),
                    clipper);

  return config;
}

std::vector<std::vector<std::string>> ConvertMapDefinition(const v1::MapDefinition& proto_map) {
  if (proto_map.height() == 0 || proto_map.width() == 0) {
    throw std::runtime_error("Map definition must include positive width and height");
  }

  std::vector<std::vector<std::string>> map(proto_map.height(),
                                            std::vector<std::string>(proto_map.width(), "empty"));

  for (const auto& cell : proto_map.cells()) {
    if (cell.row() >= proto_map.height() || cell.col() >= proto_map.width()) {
      throw std::runtime_error("Map cell coordinates out of bounds");
    }
    map[cell.row()][cell.col()] = cell.object_type();
  }

  return map;
}

}  // namespace mettagrid::rpc
