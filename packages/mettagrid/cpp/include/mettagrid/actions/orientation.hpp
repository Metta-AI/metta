#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ORIENTATION_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ORIENTATION_HPP_

#include <cstddef>

enum Orientation {
  North = 0,
  South = 1,
  West = 2,
  East = 3,
  Northwest = 4,
  Northeast = 5,
  Southwest = 6,
  Southeast = 7,
};

// Short aliases
constexpr Orientation N = North;
constexpr Orientation S = South;
constexpr Orientation W = West;
constexpr Orientation E = East;
constexpr Orientation NW = Northwest;
constexpr Orientation NE = Northeast;
constexpr Orientation SW = Southwest;
constexpr Orientation SE = Southeast;

// Movement deltas for each orientation
constexpr int ORIENTATION_DELTAS_X[8] = {
    0,   // North
    0,   // South
    -1,  // West
    1,   // East
    -1,  // Northwest
    1,   // Northeast
    -1,  // Southwest
    1    // Southeast
};

constexpr int ORIENTATION_DELTAS_Y[8] = {
    -1,  // North
    1,   // South
    0,   // West
    0,   // East
    -1,  // Northwest
    -1,  // Northeast
    1,   // Southwest
    1    // Southeast
};

// Utility functions
inline size_t getOrientationCount(bool allow_diagonals) {
  return allow_diagonals ? 8 : 4;
}

inline bool isDiagonal(Orientation o) {
  return o >= Northwest;
}

inline bool isValidOrientation(Orientation o, bool allow_diagonals) {
  return o >= North && o <= (allow_diagonals ? Southeast : East);
}

// Get the opposite orientation
inline Orientation getOpposite(Orientation orient) {
  switch (orient) {
    case North:
      return South;
    case South:
      return North;
    case West:
      return East;
    case East:
      return West;
    case Northwest:
      return Southeast;
    case Northeast:
      return Southwest;
    case Southwest:
      return Northeast;
    case Southeast:
      return Northwest;
    default:
      return orient;
  }
}

// Get the orientation 90 degrees clockwise
inline Orientation getClockwise(Orientation orient) {
  switch (orient) {
    case North:
      return East;
    case East:
      return South;
    case South:
      return West;
    case West:
      return North;
    case Northeast:
      return Southeast;
    case Southeast:
      return Southwest;
    case Southwest:
      return Northwest;
    case Northwest:
      return Northeast;
    default:
      return orient;
  }
}

// Get delta coordinates for a given orientation
inline void getOrientationDelta(Orientation orient, int& dx, int& dy) {
  dx = ORIENTATION_DELTAS_X[orient];
  dy = ORIENTATION_DELTAS_Y[orient];
}

inline const char* OrientationNames[8] = {
    "N",   // 0
    "S",   // 1
    "W",   // 2
    "E",   // 3
    "NW",  // 4
    "NE",  // 5
    "SW",  // 6
    "SE"   // 7
};

inline const char* OrientationFullNames[8] = {
    "north",      // 0
    "south",      // 1
    "west",       // 2
    "east",       // 3
    "northwest",  // 4
    "northeast",  // 5
    "southwest",  // 6
    "southeast"   // 7
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ORIENTATION_HPP_
