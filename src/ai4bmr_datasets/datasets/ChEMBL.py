from pathlib import Path
from ai4bmr_datasets.utils.download import download_file_map

import pandas as pd
from loguru import logger


class ChEMBL:

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/ChEMBL')
        self.raw_dir = self.base_dir / '01_raw'
        self.processed_dir = self.base_dir / '02_processed'
        self.chembl_version = "chembl_35"
        self.sqlite_file = self.raw_dir / self.chembl_version / f'{self.chembl_version}_sqlite' / f'{self.chembl_version}.db'
        self.db = None

    def prepare_data(self):
        self.download()

    def setup(self, query: str | None = None):
        import sqlite3

        query = query or """SELECT
    m.chembl_id, 
    s.canonical_smiles
FROM molecule_dictionary m
JOIN compound_structures s ON m.molregno = s.molregno
WHERE s.canonical_smiles IS NOT NULL
"""

        if self.db is None:
            db_path = self.sqlite_file
            if not db_path.exists():
                raise FileNotFoundError(f"{db_path} not found. Run `prepare_data()` first.")

            logger.info(f"Loading compound_structures from {db_path}")
            conn = sqlite3.connect(db_path)
            self.db = pd.read_sql_query(query, conn)
            conn.close()

    def download(self, force: bool = False):
        import tarfile

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        target_path = self.raw_dir / f"{self.chembl_version}_sqlite.tar.gz"
        file_map = {f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/{self.chembl_version}_sqlite.tar.gz": target_path}
        download_file_map(file_map=file_map, force=force)

        if not self.sqlite_file.exists():
            with tarfile.open(target_path, "r:gz") as tar:
                tar.extractall(path=self.raw_dir)

        logger.info("✅ Download and extraction completed.")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int) -> dict:
        return self.db.iloc[idx].to_dict()
