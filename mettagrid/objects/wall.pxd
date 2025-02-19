from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from mettagrid.objects.constants cimport GridLayer, ObjectType
from mettagrid.objects.metta_object cimport MettaObject, ObjectConfig

cdef cppclass Wall(MettaObject):
    inline Wall(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.WallT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]
