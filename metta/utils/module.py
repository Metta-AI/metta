import importlib


def load_class(full_class_name: str):
    parts = full_class_name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid class name: {full_class_name}")
    module_name, class_name = parts
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def load_function(full_function_name: str):
    parts = full_function_name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid function name: {full_function_name}")
    module_name, function_name = parts
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func
