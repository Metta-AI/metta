from mettagrid.mettagrid_config import MettaGridConfig


def f1():
    def inner():
        pass

    return 1


def f2() -> MettaGridConfig:
    raise NotImplementedError
