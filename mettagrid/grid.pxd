# distutils: language=c++

from libcpp.vector cimport vector
from mettagrid.grid_object cimport Layer, TypeId, GridObjectId, GridObject
from mettagrid.grid_object cimport GridLocation, Orientation, GridCoord

cdef extern from "grid.hpp":
    cdef cppclass Grid:
        unsigned int width
        unsigned int height
        Layer num_layers

        vector[vector[vector[int]]] grid
        vector[GridObject*] objects

        Grid(unsigned int width, unsigned int height, vector[Layer] layer_for_type_id)
        void __dealloc__()

        const GridLocation location(GridObjectId id)
        const GridLocation location(unsigned int r, unsigned int c, Layer layer)
        const GridLocation relative_location(
            const GridLocation &loc, Orientation orientation,
            GridCoord distance, GridCoord offset)
        const GridLocation relative_location(
            const GridLocation &loc, Orientation orientation)
        const GridLocation relative_location(
            const GridLocation &loc, Orientation orientation, TypeId type_id)
        const GridLocation relative_location(
            const GridLocation &loc, Orientation orientation,
            GridCoord distance, GridCoord offset, TypeId type_id)

        char is_empty(unsigned int r, unsigned int c)

        char add_object(GridObject *obj)
        void remove_object(GridObject *obj)
        void remove_object(GridObjectId id)
        char move_object(GridObjectId id, const GridLocation &loc)
        void swap_objects(GridObjectId id1, GridObjectId id2)
        GridObject* object(GridObjectId obj_id)
        GridObject* object_at(const GridLocation &loc)
        GridObject* object_at(const GridLocation &loc, TypeId type_id)
        GridObject* object_at(GridCoord r, GridCoord c, TypeId type_id)
