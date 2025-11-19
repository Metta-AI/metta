from metta.gridworks.routes.schemas import get_schemas


def test_schema():
    schemas = get_schemas()
    assert schemas is not None
    assert len(schemas) > 0
