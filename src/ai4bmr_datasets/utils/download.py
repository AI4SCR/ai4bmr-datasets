from pathlib import Path

from loguru import logger


def unzip_recursive(zip_path: Path, extract_dir: Path = None):
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


def download_file_map(file_map: dict[str, Path], force: bool = False):
    from tqdm import tqdm
    import requests

    for url, target_path in file_map.items():
        if not target_path.exists() or force:
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
        else:
            logger.info(f"Skipping download of {target_path.name}, already exists.")
