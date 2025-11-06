import mettagrid.config


def f1():
    def inner():
        pass

    return 1


def f2() -> mettagrid.config.MettaGridConfig:
    raise NotImplementedError
