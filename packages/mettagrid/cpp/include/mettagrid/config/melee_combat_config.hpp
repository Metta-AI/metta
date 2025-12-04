#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_MELEE_COMBAT_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_MELEE_COMBAT_CONFIG_HPP_

// Melee combat settings - isolated to avoid circular dependencies
// This struct is used by both Agent and GameConfig
struct MeleeCombatConfig {
  bool enabled = false;
  int attack_vibe_id = -1;   // Vibe ID required to attack (-1 = disabled)
  int defense_vibe_id = -1;  // Vibe ID required to defend (-1 = disabled)
  int attack_item_id = -1;   // Resource ID required to attack (-1 = disabled)
  int defense_item_id = -1;  // Resource ID required to defend (-1 = disabled)
  bool attack_consumes_item = true;
  bool defense_consumes_item = false;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_MELEE_COMBAT_CONFIG_HPP_
