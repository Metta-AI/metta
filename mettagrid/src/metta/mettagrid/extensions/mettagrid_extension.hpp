// extensions/mettagrid_extension.hpp
#ifndef EXTENSIONS_METTAGRID_EXTENSION_HPP_
#define EXTENSIONS_METTAGRID_EXTENSION_HPP_

#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>

#include "observation_encoder.hpp"
#include "types.hpp"

class MettaGrid;

using ExtensionStats = std::unordered_map<std::string, float>;

class MettaGridExtension {
public:
  virtual ~MettaGridExtension() = default;

  virtual void registerObservations(ObservationEncoder* /*enc*/) {}

  virtual void onInit(const MettaGrid* /*env*/) {}

  virtual void onReset(MettaGrid* /*env*/) {}

  virtual void onStep(MettaGrid* /*env*/) = 0;

  // stats
  virtual ExtensionStats getStats() const {
    return ExtensionStats();
  }

  // Metadata
  virtual std::string getName() const = 0;

protected:
  // Const access to agent observations
  std::span<const uint8_t> getAgentObservations(const MettaGrid* env, size_t agent_idx) const;

  // Mutable access to agent observations
  std::span<uint8_t> getAgentObservationsMutable(MettaGrid* env, size_t agent_idx);

  // Helper to get observation dimensions
  size_t getObservationSize(const MettaGrid* env) const;

  // Add other accessors as needed in the future
  // For example:
  // Grid* getGrid(MettaGrid* env);
  // EventManager* getEventManager(MettaGrid* env);
};

// Extension factory - no config needed
using ExtensionFactory = std::function<std::unique_ptr<MettaGridExtension>()>;

class ExtensionRegistry {
public:
  static ExtensionRegistry& instance() {
    static ExtensionRegistry instance;
    return instance;
  }

  void register_extension(const std::string& name, ExtensionFactory factory) {
    factories_[name] = factory;
  }

  std::unique_ptr<MettaGridExtension> create(const std::string& name) {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
      throw std::runtime_error("Unknown extension: " + name);
    }
    return it->second();
  }

private:
  std::map<std::string, ExtensionFactory> factories_;
};

// In mettagrid_extension.hpp, replace the REGISTER_EXTENSION macro with:

// Helper macro for registering extensions
#if defined(__clang__)
// Clang/AppleClang specific warning suppression
#define REGISTER_EXTENSION(name, class_name)                                                                     \
  _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wglobal-constructors\"")                 \
      _Pragma("clang diagnostic ignored \"-Wexit-time-destructors\"") namespace {                                \
    struct class_name##_Registrar {                                                                              \
      class_name##_Registrar() {                                                                                 \
        ExtensionRegistry::instance().register_extension(name, []() { return std::make_unique<class_name>(); }); \
      }                                                                                                          \
    };                                                                                                           \
    class_name##_Registrar class_name##_registrar_instance;                                                      \
  }                                                                                                              \
  _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
// GCC doesn't have -Wglobal-constructors, but might warn about other things
#define REGISTER_EXTENSION(name, class_name)                                                                          \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wpragmas\"") /* Ignore unknown pragma warnings */ \
      namespace {                                                                                                     \
    struct class_name##_Registrar {                                                                                   \
      class_name##_Registrar() {                                                                                      \
        ExtensionRegistry::instance().register_extension(name, []() { return std::make_unique<class_name>(); });      \
      }                                                                                                               \
    };                                                                                                                \
    class_name##_Registrar class_name##_registrar_instance;                                                           \
  }                                                                                                                   \
  _Pragma("GCC diagnostic pop")
#endif

#endif  // EXTENSIONS_METTAGRID_EXTENSION_HPP_
