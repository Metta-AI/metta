from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from .constants cimport GridLayer, ObjectType
from .metta_object cimport MettaObject, ObjectConfig
from .usable cimport Usable

cdef cppclass Generator(Usable):
    unsigned int r1

    inline Generator(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)
        this.r1 = cfg[b"initial_resources"]

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.r1
        obs[3] = this.ready and this.r1 > 0

    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]