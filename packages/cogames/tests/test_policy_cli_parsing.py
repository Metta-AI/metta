from pathlib import Path

import pytest

from cogames.cli.policy import _parse_policy_spec
from mettagrid.policy.loader import resolve_policy_class_path


def test_parse_policy_spec_with_class_only():
    spec = _parse_policy_spec("class=random")

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.data_path is None
    assert spec.proportion == 1.0
    assert spec.init_kwargs == {}


def test_parse_policy_spec_with_data_proportion_and_kwargs(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_text("dummy")

    spec = _parse_policy_spec(
        f"class=random,data={checkpoint},proportion=0.5,kw.alpha=0.1,kw.beta=value,kw.with-hyphen=ok"
    )

    assert spec.class_path == resolve_policy_class_path("random")
    assert spec.data_path == str(checkpoint.resolve())
    assert spec.proportion == 0.5
    assert spec.init_kwargs == {"alpha": "0.1", "beta": "value", "with_hyphen": "ok"}


@pytest.mark.parametrize(
    "raw_spec",
    [
        "",
        "data=only",
        "random:train_dir/model.pt",
        "random",
        "class=random,proportion=-1",
    ],
)
def test_parse_policy_spec_rejects_invalid_input(raw_spec: str):
    with pytest.raises(ValueError):
        _parse_policy_spec(raw_spec)
