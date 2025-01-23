#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <vector>
#include <string>

using namespace std;

typedef unsigned short Layer;
typedef unsigned short TypeId;
typedef unsigned int GridCoord;

class GridLocation {
    public:
        GridCoord r;
        GridCoord c;
        Layer layer;

        inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
        inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
        inline GridLocation(): r(0), c(0), layer(0) {}
};

enum Orientation {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3
};

typedef unsigned int GridObjectId;

class GridObject {
    public:
        GridObjectId id;
        GridLocation location;
        TypeId _type_id;

        inline void init(TypeId type_id, const GridLocation &loc) {
            this->_type_id = type_id;
            this->location = loc;
        }

        inline void init(TypeId type_id, GridCoord r, GridCoord c) {
            init(type_id, GridLocation(r, c, 0));
        }

        inline void init(TypeId type_id, GridCoord r, GridCoord c, Layer layer) {
            init(type_id, GridLocation(r, c, layer));
        }

};

#endif // GRID_OBJECT_HPP
