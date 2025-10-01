from pathlib import Path

import pytest

from metta.gridworks.configs.lsp import LSPClient


@pytest.fixture
def lsp_client():
    client = LSPClient()
    yield client
    client.shutdown()


@pytest.mark.skip(reason="flaky")
def test_create_shutdown(lsp_client: LSPClient):
    assert lsp_client is not None


def test_get_file_symbols(lsp_client: LSPClient):
    file_symbols = lsp_client.get_file_symbols(Path("tests/gridworks/configs/fixtures/example.py"))

    assert file_symbols is not None
    assert len(file_symbols) == 3


def test_get_hover(lsp_client: LSPClient):
    hover = lsp_client.get_hover(Path("tests/gridworks/configs/fixtures/example.py"), 3, 4)

    assert hover is not None
    assert hover["contents"]["value"] == "(function) def f1() -> Literal[1]"
