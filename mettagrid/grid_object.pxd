# distutils: language=c++

cdef extern from "grid_object.hpp":
    ctypedef unsigned short Layer
    ctypedef unsigned short TypeId

    ctypedef unsigned int GridCoord
    cdef cppclass GridLocation:
        GridCoord r
        GridCoord c
        Layer layer
        GridLocation()
        GridLocation(GridCoord r, GridCoord c, Layer l)
        GridLocation(GridCoord r, GridCoord c)

    ctypedef enum Orientation:
        Up = 0
        Down = 1
        Left = 2
        Right = 3

    ctypedef unsigned int GridObjectId

    cdef cppclass GridObject:
        GridObjectId id
        GridLocation location
        TypeId _type_id

        GridObject()
        void __dealloc__()
        void init(TypeId type_id, const GridLocation &loc)

