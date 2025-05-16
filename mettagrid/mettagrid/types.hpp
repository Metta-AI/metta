#ifndef TYPES_HPP
#define TYPES_HPP

// ============================================================================
// NUMPY TYPE NAME MACROS
// ============================================================================

// Type names to use in Python - these must match the C++ types above
#define NUMPY_OBSERVATIONS_TYPE "uint8"  // match c_observations_type
#define NUMPY_TERMINALS_TYPE "uint8"     // match c_terminals_type
#define NUMPY_TRUNCATIONS_TYPE "uint8"   // match c_truncations_type
#define NUMPY_REWARDS_TYPE "float32"     // match c_rewards_type
#define NUMPY_ACTIONS_TYPE "uint8"       // match c_actions_type
#define NUMPY_MASKS_TYPE "uint8"         // match c_masks_type
#define NUMPY_SUCCESS_TYPE "uint8"       // match c_success_type

#endif  // TYPES_HPP
