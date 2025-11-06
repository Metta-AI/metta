import metta.sweep.core


def test_categorical_parameter_config_creation():
    cat = metta.sweep.core.CategoricalParameterConfig(choices=["red", "blue", "butt"])
    assert cat.choices == ["red", "blue", "butt"]


def test_categorical_parameter_config_minimal():
    cat = metta.sweep.core.CategoricalParameterConfig(choices=["cpu", "cuda"])
    assert cat.choices == ["cpu", "cuda"]


def test_categorical_builder():
    param = metta.sweep.core.SweepParameters.categorical("model.color", ["red", "blue"])
    key, cfg = next(iter(param.items()))
    assert key == "model.color"
    assert cfg.choices == ["red", "blue"]
