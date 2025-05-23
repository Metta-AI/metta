import builtins

from metta.map.utils import storage


def test_save_to_uri_local(tmp_path):
    file_path = tmp_path / "file.txt"
    storage.save_to_uri("hello", str(file_path))
    assert file_path.read_text() == "hello"


def test_save_to_uri_s3(monkeypatch):
    calls = []

    class DummyS3:
        def put_object(self, Bucket, Key, Body):
            calls.append((Bucket, Key, Body))

    monkeypatch.setattr(storage, "get_s3_client", lambda: DummyS3())
    monkeypatch.setattr(
        storage,
        "parse_file_uri",
        lambda uri: (_ for _ in ()).throw(AssertionError("parse_file_uri should not be called")),
    )
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("open should not be called")),
    )

    storage.save_to_uri("data", "s3://bucket/key")
    assert calls == [("bucket", "key", "data")]
