#ifndef TYPES_HPP
#define TYPES_HPP

// ============================================================================
// NUMPY TYPE NAME MACROS
// ============================================================================

// Define core types used for numpy arrays
typedef uint8_t numpy_bool_t;
typedef uint8_t c_observations_type;
typedef numpy_bool_t c_terminals_type;
typedef numpy_bool_t c_truncations_type;
typedef float c_rewards_type;
typedef uint8_t c_actions_type;
typedef numpy_bool_t c_masks_type;
typedef numpy_bool_t c_success_type;

// Type names to use in Python - these must match the C++ types above
#define NUMPY_OBSERVATIONS_TYPE "uint8"  // match c_observations_type
#define NUMPY_TERMINALS_TYPE "uint8"     // match c_terminals_type
#define NUMPY_TRUNCATIONS_TYPE "uint8"   // match c_truncations_type
#define NUMPY_REWARDS_TYPE "float32"     // match c_rewards_type
#define NUMPY_ACTIONS_TYPE "uint8"       // match c_actions_type
#define NUMPY_MASKS_TYPE "uint8"         // match c_masks_type
#define NUMPY_SUCCESS_TYPE "uint8"       // match c_success_type

#endif  // TYPES_HPP
