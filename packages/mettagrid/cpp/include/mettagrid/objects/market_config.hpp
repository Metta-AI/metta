#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <unordered_map>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/inventory_config.hpp"

// Configuration for a single market terminal (one per direction)
struct MarketTerminalConfig {
  bool sell = false;  // true = sell to market, false = buy from market
  int amount = 1;     // max amount to trade per transaction

  MarketTerminalConfig() = default;
  MarketTerminalConfig(bool sell, int amount) : sell(sell), amount(amount) {}
};

struct MarketConfig : public GridObjectConfig {
  MarketConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : GridObjectConfig(type_id, type_name, initial_vibe),
        initial_inventory({}),
        inventory_config(),
        currency_resource_id(0) {}

  // Terminal configs by direction (Orientation enum values: 0=North, 1=South, 2=West, 3=East)
  std::unordered_map<int, MarketTerminalConfig> terminals;

  // Initial inventory for each resource type
  std::unordered_map<InventoryItem, int> initial_inventory;

  // Inventory configuration with limits
  InventoryConfig inventory_config;

  // The resource ID for the currency (e.g., hearts)
  InventoryItem currency_resource_id;

  // Maps vibe ID -> resource ID for trading (vibe names that match resource names)
  std::unordered_map<ObservationType, InventoryItem> vibe_to_resource;

  // Vibe names for debug output
  std::vector<std::string> vibe_names;
};

namespace py = pybind11;

inline void bind_market_terminal_config(py::module& m) {
  py::class_<MarketTerminalConfig>(m, "MarketTerminalConfig")
      .def(py::init<>())
      .def(py::init<bool, int>(), py::arg("sell") = false, py::arg("amount") = 1)
      .def_readwrite("sell", &MarketTerminalConfig::sell)
      .def_readwrite("amount", &MarketTerminalConfig::amount);
}

inline void bind_market_config(py::module& m) {
  bind_market_terminal_config(m);

  py::class_<MarketConfig, GridObjectConfig, std::shared_ptr<MarketConfig>>(m, "MarketConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &MarketConfig::type_id)
      .def_readwrite("type_name", &MarketConfig::type_name)
      .def_readwrite("tag_ids", &MarketConfig::tag_ids)
      .def_readwrite("terminals", &MarketConfig::terminals)
      .def_readwrite("initial_inventory", &MarketConfig::initial_inventory)
      .def_readwrite("inventory_config", &MarketConfig::inventory_config)
      .def_readwrite("currency_resource_id", &MarketConfig::currency_resource_id)
      .def_readwrite("vibe_to_resource", &MarketConfig::vibe_to_resource)
      .def_readwrite("vibe_names", &MarketConfig::vibe_names)
      .def_readwrite("initial_vibe", &MarketConfig::initial_vibe);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_CONFIG_HPP_
