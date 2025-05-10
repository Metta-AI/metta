#ifndef GRID_HPP
#define GRID_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "constants.hpp"
#include "grid_object.hpp"

/**
 * A 3D grid representation for storing game objects.
 * Each cell in the grid can contain objects on different layers.
 */
typedef std::vector<std::vector<std::vector<GridObjectId>>> GridType;

/**
 * The Grid class manages a 3D spatial representation of objects. It handles
 * object placement, removal, movement, and spatial queries.
 *
 * The grid is organized as a 3D structure:
 * - First dimension: rows (height)
 * - Second dimension: columns (width)
 * - Third dimension: layers
 *
 * Each position in the grid holds a GridObjectId that references an object
 * in the objects vector, or 0 for empty space.
 */
class Grid {
public:
  uint32_t width;                     // Width of the grid (columns)
  uint32_t height;                    // Height of the grid (rows)
  std::vector<Layer> typeToLayerMap;  // Maps object types to their respective layers
  Layer num_layers;                   // Total number of layers in the grid

  GridType grid;                     // The 3D grid structure that holds object IDs
  std::vector<GridObject*> objects;  // Storage for all objects, indexed by their ID

  /**
   * Constructs a Grid with specified dimensions and type-to-layer mapping.
   *
   * @param width The width (columns) of the grid
   * @param height The height (rows) of the grid
   * @param typeToLayerMap Vector that maps each object type ID to its corresponding layer
   */
  inline Grid(uint32_t width, uint32_t height, std::vector<Layer> typeToLayerMap)
      : width(width), height(height), typeToLayerMap(typeToLayerMap) {
    // Calculate the number of layers needed based on the mapping
    num_layers = *std::max_element(typeToLayerMap.begin(), typeToLayerMap.end()) + 1;

    // Initialize the grid with all positions set to 0 (empty)
    grid.resize(height, std::vector<std::vector<GridObjectId>>(width, std::vector<GridObjectId>(this->num_layers, 0)));

    // Reserve ID 0 for empty space
    objects.push_back(nullptr);
  }

  /**
   * Constructs a Grid with specified dimensions and a default type-to-layer mapping
   * based on the ObjectLayers defined in constants.hpp.
   *
   * @param width The width (columns) of the grid
   * @param height The height (rows) of the grid
   */
  inline Grid(uint32_t width, uint32_t height) : width(width), height(height) {
    // Get the maximum type ID to properly size our typeToLayerMap
    TypeId maxTypeId = 0;
    for (const auto& [type, layer] : ObjectLayers) {
      maxTypeId = std::max(maxTypeId, type);
    }

    // Initialize typeToLayerMap with default layers
    typeToLayerMap.resize(maxTypeId + 1, 0);  // Default all to layer 0

    // Set the layers according to ObjectLayers
    for (const auto& [type, layer] : ObjectLayers) {
      typeToLayerMap[type] = layer;
    }

    // Calculate the number of layers needed based on the mapping
    num_layers = *std::max_element(typeToLayerMap.begin(), typeToLayerMap.end()) + 1;

    // Initialize the grid with all positions set to 0 (empty)
    grid.resize(height, std::vector<std::vector<GridObjectId>>(width, std::vector<GridObjectId>(this->num_layers, 0)));

    // Reserve ID 0 for empty space
    objects.push_back(nullptr);
  }

  /**
   * Destructor: Cleans up all objects in the grid.
   */
  inline ~Grid() {
    for (uint64_t id = 1; id < objects.size(); ++id) {
      if (objects[id] != nullptr) {
        delete objects[id];
        objects[id] = nullptr;  // Set to nullptr after deletion for safety
      }
    }
  }

  /**
   * Adds an object to the grid at its current location.
   *
   * @param obj Pointer to the GridObject to add
   * @return True if the object was successfully added, false otherwise
   */
  inline bool add_object(GridObject* obj) {
    // Check if the object's location is valid
    if (obj->location.r >= height or obj->location.c >= width or obj->location.layer >= num_layers) {
      return false;
    }
    // Check if the target location is already occupied
    if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != 0) {
      return false;
    }

    // Assign a new ID to the object and add it to the objects vector
    obj->id = this->objects.size();
    this->objects.push_back(obj);
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
    return true;
  }

  /**
   * Removes an object from the grid (does not delete the object).
   *
   * @param obj Pointer to the GridObject to remove
   */
  inline void remove_object(GridObject* obj) {
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    // Don't delete the object here, just mark it as removed in the grid
    this->objects[obj->id] = nullptr;
    // The actual deletion is deferred to the destructor
  }

  /**
   * Removes an object from the grid by its ID.
   *
   * @param id The ID of the object to remove
   */
  inline void remove_object(GridObjectId id) {
    GridObject* obj = this->objects[id];
    this->remove_object(obj);
  }

  /**
   * Moves an object to a new location.
   *
   * @param id The ID of the object to move
   * @param dest The destination location
   * @return True if the move was successful, false otherwise
   */
  inline bool move_object(GridObjectId id, const GridLocation& dest) {
    // Check if the destination is valid
    if (dest.r >= height or dest.c >= width or dest.layer >= num_layers) {
      return false;
    }

    // Check if the destination is already occupied
    if (grid[dest.r][dest.c][dest.layer] != 0) {
      return false;
    }

    // Move the object
    GridObject* obj = objects[id];
    GridLocation src = obj->location;
    grid[dest.r][dest.c][dest.layer] = id;
    grid[src.r][src.c][src.layer] = 0;
    obj->location = dest;
    return true;
  }

  /**
   * Swaps the positions of two objects, maintaining their respective layers.
   *
   * @param id1 The ID of the first object
   * @param id2 The ID of the second object
   */
  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = objects[id1];
    GridLocation pos1 = obj1->location;
    Layer layer1 = pos1.layer;
    grid[pos1.r][pos1.c][pos1.layer] = 0;

    GridObject* obj2 = objects[id2];
    GridLocation pos2 = obj2->location;
    Layer layer2 = pos2.layer;
    grid[pos2.r][pos2.c][pos2.layer] = 0;

    // Keep the layer the same for each object
    obj1->location = pos2;
    obj1->location.layer = layer1;
    obj2->location = pos1;
    obj2->location.layer = layer2;

    // Update the grid
    grid[obj1->location.r][obj1->location.c][obj1->location.layer] = id1;
    grid[obj2->location.r][obj2->location.c][obj2->location.layer] = id2;
  }

  /**
   * Retrieves an object by its ID.
   *
   * @param obj_id The ID of the object to retrieve
   * @return Pointer to the GridObject, or nullptr if not found
   */
  inline GridObject* object(GridObjectId obj_id) {
    return objects[obj_id];
  }

  /**
   * Retrieves an object at a specific location.
   *
   * @param pos The location to check
   * @return Pointer to the GridObject at that location, or nullptr if none exists
   */
  inline GridObject* object_at(const GridLocation& pos) {
    if (pos.r >= height or pos.c >= width or pos.layer >= num_layers) {
      return nullptr;
    }
    if (grid[pos.r][pos.c][pos.layer] == 0) {
      return nullptr;
    }
    return objects[grid[pos.r][pos.c][pos.layer]];
  }

  /**
   * Retrieves an object of a specific type at a location.
   *
   * @param pos The location to check
   * @param type_id The type ID of the object to find
   * @return Pointer to the GridObject if found and of the correct type, nullptr otherwise
   */
  inline GridObject* object_at(const GridLocation& pos, TypeId type_id) {
    GridObject* obj = object_at(pos);
    if (obj != nullptr and obj->_type_id == type_id) {
      return obj;
    }
    return nullptr;
  }

  /**
   * Retrieves an object of a specific type at grid coordinates, using the typeToLayerMap
   * to determine which layer to check.
   *
   * @param r Row coordinate
   * @param c Column coordinate
   * @param type_id The type ID of the object to find
   * @return Pointer to the GridObject if found and of the correct type, nullptr otherwise
   */
  inline GridObject* object_at(GridCoord r, GridCoord c, TypeId type_id) {
    // Look in the appropriate layer for this type
    Layer layer = this->typeToLayerMap[type_id];
    GridObject* obj = object_at(GridLocation(r, c, layer));

    // Verify the object is of the expected type
    if (obj == nullptr || obj->_type_id != type_id) {
      return nullptr;
    }

    return obj;
  }

  /**
   * Gets the location of an object by its ID.
   *
   * @param id The ID of the object
   * @return The GridLocation of the object
   */
  inline const GridLocation location(GridObjectId id) {
    return objects[id]->location;
  }

  /**
   * Calculates a location relative to an origin based on orientation and distance.
   *
   * @param origin The starting location
   * @param orientation The direction (Up, Down, Left, Right)
   * @param distance How far to move in the primary direction
   * @param offset How far to move in the secondary direction
   * @return The calculated relative location
   */
  inline const GridLocation relative_location(const GridLocation& origin,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset) {
    int32_t targetR = origin.r;
    int32_t targetC = origin.c;

    // Calculate target coordinates based on orientation
    switch (orientation) {
      case Up:
        targetR = origin.r - distance;
        targetC = origin.c - offset;
        break;
      case Down:
        targetR = origin.r + distance;
        targetC = origin.c + offset;
        break;
      case Left:
        targetR = origin.r + offset;
        targetC = origin.c - distance;
        break;
      case Right:
        targetR = origin.r - offset;
        targetC = origin.c + distance;
        break;
    }

    // Ensure coordinates are within grid bounds
    targetR = std::max(0, targetR);
    targetC = std::max(0, targetC);

    return GridLocation(targetR, targetC, origin.layer);
  }

  /**
   * Calculates a location relative to an origin, with a specific type (and thus layer).
   *
   * @param origin The starting location
   * @param orientation The direction (Up, Down, Left, Right)
   * @param distance How far to move in the primary direction
   * @param offset How far to move in the secondary direction
   * @param type_id The type ID to determine the layer of the result
   * @return The calculated relative location
   */
  inline const GridLocation relative_location(const GridLocation& origin,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset,
                                              TypeId type_id) {
    // Calculate the base location
    GridLocation target = this->relative_location(origin, orientation, distance, offset);

    // Update the layer based on the specified type
    target.layer = this->typeToLayerMap[type_id];

    return target;
  }

  /**
   * Simplified version of relative_location that moves one unit in the given orientation.
   *
   * @param origin The starting location
   * @param orientation The direction to move
   * @return The calculated relative location
   */
  inline const GridLocation relative_location(const GridLocation& origin, Orientation orientation) {
    return this->relative_location(origin, orientation, 1, 0);
  }

  /**
   * Simplified version of relative_location that moves one unit in the given orientation
   * and sets the appropriate layer for the specified type.
   *
   * @param origin The starting location
   * @param orientation The direction to move
   * @param type_id The type ID to determine the layer of the result
   * @return The calculated relative location
   */
  inline const GridLocation relative_location(const GridLocation& origin, Orientation orientation, TypeId type_id) {
    return this->relative_location(origin, orientation, 1, 0, type_id);
  }

  /**
   * Checks if a grid cell is empty on all layers.
   *
   * @param row Row coordinate
   * @param col Column coordinate
   * @return True if the cell is empty on all layers, false otherwise
   */
  inline bool is_empty(uint32_t row, uint32_t col) {
    GridLocation pos;
    pos.r = row;
    pos.c = col;

    // Check all layers at this location
    for (int32_t layer = 0; layer < num_layers; ++layer) {
      pos.layer = layer;
      if (object_at(pos) != nullptr) {
        return false;
      }
    }
    return true;
  }
};

#endif  // GRID_HPP