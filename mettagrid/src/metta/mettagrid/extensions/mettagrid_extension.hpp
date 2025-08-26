// extensions/mettagrid_extension.hpp
#ifndef EXTENSIONS_METTAGRID_EXTENSION_HPP_
#define EXTENSIONS_METTAGRID_EXTENSION_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "types.hpp"

class MettaGrid;

namespace py = pybind11;

class MettaGridExtension {
public:
  virtual ~MettaGridExtension() = default;

  // Core lifecycle methods
  virtual void onInit(const MettaGrid* /*env*/) {}
  virtual void onReset(MettaGrid* /*env*/) {}
  virtual void onStep(MettaGrid* /*env*/) = 0;

  // stats
  virtual py::dict getStats() const {
    return py::dict();
  }

  // Metadata
  virtual std::string getName() const = 0;

protected:
  // Protected accessor methods that derived classes can use
  // These work because MettaGridExtension is a friend of MettaGrid
  // Returns non-const reference to allow modifications
  py::array_t<uint8_t>& getObservations(MettaGrid* env);

  // Const version for read-only access when you have a const MettaGrid*
  const py::array_t<uint8_t>& getObservations(const MettaGrid* env) const;

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
#else
// Other compilers - no warning suppression
#define REGISTER_EXTENSION(name, class_name)                                                                   \
  namespace {                                                                                                  \
  struct class_name##_Registrar {                                                                              \
    class_name##_Registrar() {                                                                                 \
      ExtensionRegistry::instance().register_extension(name, []() { return std::make_unique<class_name>(); }); \
    }                                                                                                          \
  };                                                                                                           \
  class_name##_Registrar class_name##_registrar_instance;                                                      \
  }
#endif

#endif  // EXTENSIONS_METTAGRID_EXTENSION_HPP_
