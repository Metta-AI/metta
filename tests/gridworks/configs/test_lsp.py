import pathlib

import pytest

import metta.gridworks.configs.lsp


@pytest.fixture
def lsp_client():
    client = metta.gridworks.configs.lsp.LSPClient()
    yield client
    client.shutdown()


# In one instance, client failed to initialize  -- lsp_client.recv_id(init_id) timed out
@pytest.mark.skip(reason="flaky")
def test_create_shutdown(lsp_client: metta.gridworks.configs.lsp.LSPClient):
    assert lsp_client is not None


@pytest.mark.skip(reason="flaky")
def test_get_file_symbols(lsp_client: metta.gridworks.configs.lsp.LSPClient):
    file_symbols = lsp_client.get_file_symbols(pathlib.Path("tests/gridworks/configs/fixtures/example.py"))

    assert file_symbols is not None
    assert len(file_symbols) == 3


@pytest.mark.skip(reason="flaky")
def test_get_hover(lsp_client: metta.gridworks.configs.lsp.LSPClient):
    hover = lsp_client.get_hover(pathlib.Path("tests/gridworks/configs/fixtures/example.py"), 3, 4)

    assert hover is not None
    assert hover["contents"]["value"] == "(function) def f1() -> Literal[1]"
