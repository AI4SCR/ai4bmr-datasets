import hashlib
from pathlib import Path
import importlib.util
import sys
import types

import pytest


class _DummyTqdm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def update(self, *_args, **_kwargs):
        return None


loguru_stub = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_, **__: None,
                                                                error=lambda *_, **__: None))


def _requests_get_stub(*_args, **_kwargs):
    raise RuntimeError("requests.get should be patched in tests")


requests_stub = types.SimpleNamespace(
    HTTPError=Exception,
    get=_requests_get_stub,
)
tqdm_stub = types.SimpleNamespace(tqdm=lambda *_, **__: _DummyTqdm())

sys.modules.setdefault("loguru", loguru_stub)
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("tqdm", tqdm_stub)

module_spec = importlib.util.spec_from_file_location(
    "ai4bmr_datasets.utils.download",
    Path(__file__).resolve().parents[2] / "src" / "ai4bmr_datasets" / "utils" / "download.py",
)
download_module = importlib.util.module_from_spec(module_spec)
assert module_spec.loader is not None
module_spec.loader.exec_module(download_module)

download_file_map = download_module.download_file_map
DownloadRecord = download_module.DownloadRecord


class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status code: {self.status_code}")

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]


def test_download_file_map_validates_checksum(tmp_path, monkeypatch):
    url = "https://example.org/file.bin"
    target = tmp_path / "file.bin"
    payload = b"checksum me"

    def fake_get(requested_url, headers, stream):
        assert requested_url == url
        assert stream is True
        return DummyResponse(payload)

    monkeypatch.setattr("requests.get", fake_get)

    download_file_map(
        [
            DownloadRecord(
                url=url,
                file_name=target.name,
                checksum=hashlib.sha256(payload).hexdigest(),
            ),
        ],
        download_dir=tmp_path,
    )

    assert target.read_bytes() == payload


def test_download_file_map_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    url = "https://example.org/file.bin"
    target = tmp_path / "file.bin"
    payload = b"checksum me"

    def fake_get(requested_url, headers, stream):
        return DummyResponse(payload)

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file_map(
            [
                DownloadRecord(
                    url=url,
                    file_name=target.name,
                    checksum="deadbeef",
                ),
            ],
            download_dir=tmp_path,
        )

    assert not target.exists()


def test_download_file_map_validates_existing_files(tmp_path):
    url = "https://example.org/file.bin"
    target = tmp_path / "file.bin"
    target.write_bytes(b"bad data")

    expected = hashlib.sha256(b"good data").hexdigest()

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file_map(
            [
                DownloadRecord(
                    url=url,
                    file_name=target.name,
                    checksum=expected,
                ),
            ],
            download_dir=tmp_path,
        )

    assert not target.exists()
