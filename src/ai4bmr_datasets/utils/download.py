from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hashlib

from loguru import logger


def unzip_recursive(zip_path: Path, extract_dir: Path = None):
    """
    Recursively unzips a given zip file to a specified directory.

    If `extract_dir` is not provided, it defaults to a directory named after the zip file
    (without the .zip extension) in the same parent directory as the zip file.
    The function checks if files have already been extracted to avoid redundant operations.
    It also handles nested zip files by recursively calling itself on any .zip files found
    within the extracted content.

    Args:
        zip_path (Path): The path to the zip file to be unzipped.
        extract_dir (Path, optional): The directory where the contents should be extracted.
                                      Defaults to None.
    """
    import zipfile
    from loguru import logger

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_names = zip_ref.namelist()
        extract_dir = extract_dir or zip_path.parent / zip_path.stem
        is_extracted = all((extract_dir / name).exists() for name in extracted_names)
        if is_extracted:
            logger.info(f"Skipping extraction for {zip_path.name}, files already exist.")
        else:
            logger.info(f"Unzipping {zip_path.name}")

            # zip_ref.extractall(extract_dir)
            for zip_info in zip_ref.infolist():
                if zip_info.filename == '/':
                    continue
                elif zip_info.filename.strip():  # Skip empty or whitespace-only filenames
                    zip_ref.extract(zip_info, extract_dir)

        # Recursively unzip any .zip files in extracted content
        for name in extracted_names:
            extracted_path = extract_dir / name
            if extracted_path.suffix == ".zip":
                unzip_recursive(extracted_path)


def _compute_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute the checksum for ``path`` using ``algorithm``.

    Parameters
    ----------
    path:
        File to hash.
    algorithm:
        Hash algorithm supported by :mod:`hashlib` (defaults to ``sha256``).

    Returns
    -------
    str
        Hex digest of the file contents.
    """

    hasher = hashlib.new(algorithm)
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class DownloadRecord:
    """Metadata describing a downloadable artifact."""

    url: str
    file_name: str
    checksum: str | None = None
    checksum_algorithm: str = "sha256"


def download_file_map(
        file_map: Iterable[DownloadRecord],
        download_dir: Path,
        force: bool = False,
):
    """
    Download multiple files described by :class:`DownloadRecord` instances.

    This function iterates through the ``file_map`` records, downloading each file to
    ``download_dir``. It includes a progress bar for each download and checks if a file
    already exists at the target path to skip re-downloading, unless ``force`` is set
    to ``True``. When checksum information is provided the downloaded (or existing)
    file is validated before being reused.

    Args:
        file_map (Iterable[DownloadRecord]): An iterable of download records describing
            the remote URLs and metadata for each artifact.
        download_dir (Path): Base directory where files will be stored. The final
            target path for each record is constructed from this directory and the
            record's ``file_name``.
        force (bool): If True, forces re-download of files even if they already exist.
    """
    from tqdm import tqdm
    import requests

    download_dir.mkdir(parents=True, exist_ok=True)

    for record in file_map:
        target_path = download_dir / record.file_name
        url = record.url
        if not target_path.exists() or force:
            try:
                logger.info(f"Downloading {url} → {target_path}")
                headers = {"User-Agent": "Mozilla/5.0"}
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('Content-Length', 0))
                    chunk_size = 8192

                    with open(target_path, 'wb') as f, tqdm(
                            desc=target_path.name,
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                logger.error(f"Failed to download {url}. Network connection might have been interrupted. Error: {e}")
                if target_path.exists():
                    target_path.unlink()
        else:
            logger.info(f"Skipping download of {target_path.name}, already exists.")

        if record.checksum:
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Expected downloaded file {target_path} for checksum verification"
                )

            expected_checksum = record.checksum.lower()
            actual_checksum = _compute_checksum(
                target_path,
                algorithm=record.checksum_algorithm,
            ).lower()

            if actual_checksum != expected_checksum:
                target_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {target_path} (expected {expected_checksum}, got {actual_checksum})."
                )
