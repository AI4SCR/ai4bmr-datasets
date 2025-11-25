import hashlib
from pathlib import Path

import pytest
from ai4bmr_datasets.utils.download import DownloadRecord, download_file_map

# tmp_path = Path('/Volumes/T7/')

def test_download_file_map_validates_checksum(tmp_path, monkeypatch):
    url = "https://example.org/file.bin"
    target = Path(tmp_path) / "file.bin"
    payload = b"checksum me"
    target.write_bytes(payload)

    file_map = [
            DownloadRecord(
                url=url,
                file_name=target.name,
                checksum=hashlib.sha256(payload).hexdigest(),
            ),
        ]

    download_file_map(
        file_map=file_map,
        download_dir=tmp_path,
    )

    assert target.read_bytes() == payload


def test_download_file_map_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    url = "https://example.org/file.bin"
    target = Path(tmp_path) / "file.bin"
    payload = b"checksum me"
    target.write_bytes(payload)

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
