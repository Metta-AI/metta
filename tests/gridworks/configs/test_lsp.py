from pytest import fixture

from metta.gridworks.configs.lsp import LSPClient


@fixture
def lsp_client():
    client = LSPClient()
    yield client
    client.shutdown()


def test_create_shutdown(lsp_client: LSPClient):
    assert lsp_client is not None
