from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from .metta_object cimport MettaObject, ObjectConfig
from .usable cimport Usable
from .constants cimport GridLayer, ObjectType

cdef cppclass Altar(Usable):
    inline Altar(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]