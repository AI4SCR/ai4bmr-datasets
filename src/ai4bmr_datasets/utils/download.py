from pathlib import Path

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