#ifndef TYPES_HPP
#define TYPES_HPP

// Define core types used for shred memory arrays
typedef uint8_t c_observations_type;
typedef bool c_terminals_type;
typedef bool c_truncations_type;
typedef float c_rewards_type;
typedef int c_actions_type;

// Type names to use in Python - these must match the C++ types above
#define NUMPY_OBSERVATIONS_TYPE "uint8"  // match c_observations_type
#define NUMPY_TERMINALS_TYPE "uint8"     // match c_terminals_type
#define NUMPY_TRUNCATIONS_TYPE "uint8"   // match c_truncations_type
#define NUMPY_REWARDS_TYPE "float32"     // match c_rewards_type
#define NUMPY_ACTIONS_TYPE "uint8"       // match c_actions_type

#endif  // TYPES_HPP
