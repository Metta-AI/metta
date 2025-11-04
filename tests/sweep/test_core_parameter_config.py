from metta.sweep.core import Distribution, ParameterConfig, SweepParameters


def test_parameter_config_creation():
    pc = ParameterConfig(min=0.001, max=0.1, distribution="log_normal", mean=0.01, scale="auto")
    assert pc.min == 0.001
    assert pc.max == 0.1
    assert pc.distribution == "log_normal"
    assert pc.mean == 0.01
    assert pc.scale == "auto"


def test_sweep_parameters_builder_logit_sanitization():
    # Ensure logit bounds are sanitized away from 0 and 1
    param = SweepParameters.param(
        name="trainer.losses.loss_configs.ppo.gae_lambda",
        distribution=Distribution.LOGIT_NORMAL,
        min=0.0,
        max=1.0,
        search_center=None,
        scale="auto",
    )
    cfg = next(iter(param.values()))
    assert cfg.min <= 1e-6
    assert cfg.max >= 1 - 1e-6
