from mettaconf import MettaConf


def write_config(filename, content):
    with open(str(filename), "w") as f:
        f.write(content)


def test_basic(tmpdir):
    filename = str(tmpdir / "1.yaml")
    write_config(
        filename,
        """
mykey:
    foo: 5
        """,
    )
    cfg = MettaConf.load(filename)
    assert cfg.mykey.foo == 5


def test_load_only(tmpdir):
    f1 = str(tmpdir / "1.yaml")
    f2 = str(tmpdir / "2.yaml")
    write_config(
        f1,
        """
mykey$load: ./2.yaml
        """,
    )
    write_config(
        f2,
        """
foo: 5
        """,
    )
    cfg = MettaConf.load(f1)
    assert cfg.mykey.foo == 5


def test_load_and_override(tmpdir):
    f1 = str(tmpdir / "1.yaml")
    f2 = str(tmpdir / "2.yaml")
    write_config(
        f1,
        """
mykey$load: ./2.yaml
mykey:
    bar: 10
    baz: 11
        """,
    )
    write_config(
        f2,
        """
foo: 5
bar: 6
        """,
    )
    cfg = MettaConf.load(f1)
    assert cfg.mykey.foo == 5
    assert cfg.mykey.bar == 10
    assert cfg.mykey.baz == 11
