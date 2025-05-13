from pathlib import Path
from loguru import logger
from ai4bmr_core.utils.tidy import tidy_name

import pandas as pd

from .BaseIMCDataset import BaseIMCDataset


class BLCa(BaseIMCDataset):

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)

        # raw paths
        self.raw_metadata_path = self.base_dir / '01_raw' / 'metadata' / '02_processed' / 'labels.parquet'
        self.mcd_metadata_path = self.base_dir / "metadata" / "02_processed" / "mcd_metadata.parquet"

        # populated by `self.load()`
        self.sample_ids = None
        self.samples = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def prepare_data(self):
        self.create_clinical_metadata()
        self.create_metadata()
        pass

    def create_clinical_metadata(self):
        raw_dir = self.raw_dir / "metadata" / "01_raw"
        processed_dir = self.raw_dir / "metadata" / "02_processed"

        # ROI matching
        filename = "Analyses_cohort 2.xlsx"
        roi_matching = pd.read_excel(raw_dir / filename, dtype=str, header=4)
        cont = []
        for i, _type in enumerate(["tur-b", "primary-tumor", "lymph-node-metastases"]):
            cols = [f"Localisation.{i}", f"Case.{i}"] if i else ["Localisation", "Case"]
            dat = roi_matching[cols]
            dat.columns = ["sample_id", "patient_id"]
            dat = dat.assign(type=_type)
            cont.append(dat)

        ids = pd.concat(cont)
        ids = ids.dropna(subset="sample_id")
        ids.loc[:, "patient_id"] = ids.patient_id.ffill()
        assert ids.sample_id.duplicated().any() == False
        assert ids.patient_id.value_counts().max() <= 6

        # mcd data
        mcd_metadata_path = self.raw_dir / 'metadata' / '02_processed' / 'mcd_metadata.parquet'
        mcd_metadata = pd.read_parquet(mcd_metadata_path)
        # NOTE: rename acquisitions that were split
        mcd_metadata = mcd_metadata.assign(sample_id=mcd_metadata.Description)
        mcd_metadata.loc[
            mcd_metadata.Description.str.startswith("1C2b"), "sample_id"
        ] = "1C2b"
        mcd_metadata.loc[
            mcd_metadata.Description.str.startswith("1C3i"), "sample_id"
        ] = "1C3i"
        assert mcd_metadata.sample_id.duplicated().sum() == 2

        # create acquisition_id
        mcd_metadata = mcd_metadata.assign(
            acquisition_id=mcd_metadata.file_name_napari.map(lambda x: Path(x).stem)
        )

        sample_ids_with_not_in_analysis = set(mcd_metadata.sample_id) - set(
            ids.sample_id
        )
        sample_ids_with_no_acquisition = set(ids.sample_id) - set(
            mcd_metadata.sample_id
        )

        metadata = pd.merge(ids, mcd_metadata, how="outer")
        metadata[metadata.sample_id.isin(sample_ids_with_no_acquisition)].to_csv(
            processed_dir / "samples_with_no_acquisition.csv"
        )
        metadata[metadata.sample_id.isin(sample_ids_with_not_in_analysis)].to_csv(
            processed_dir / "sample_ids_with_not_in_analysis.csv"
        )

        sample_ids_to_remove = sample_ids_with_not_in_analysis.union(
            sample_ids_with_no_acquisition
        )
        metadata = metadata[~metadata.sample_id.isin(sample_ids_to_remove)]

        # clinical metadata
        clinical_metadata = pd.read_excel(
            raw_dir / "metadata_NAC_TMA_11_08.xlsx", dtype=str
        )
        clinical_metadata = clinical_metadata.iloc[:59, :]
        clinical_metadata = clinical_metadata.dropna(
            axis=1, how="all"
        )  # remove NaN columns
        # NOTE: some patients have a `Nr old` but everything else is NaN
        clinical_metadata = clinical_metadata[
            ~clinical_metadata.drop(columns="Nr old").isna().all(axis=1)
        ]
        # NOTE: `Nr. old` are for the matching with TMA and new ones for the genomic profiling
        clinical_metadata = clinical_metadata.assign(
            patient_id=clinical_metadata["Nr old"].astype(int).astype(str)
        )

        patient_ids_with_no_acquisition = set(clinical_metadata.patient_id) - set(
            metadata.patient_id
        )
        patient_ids_with_no_clinical_metadata = set(metadata.patient_id) - set(
            clinical_metadata.patient_id
        )

        clinical_metadata[
            clinical_metadata.patient_id.isin(patient_ids_with_no_acquisition)
        ].to_csv(processed_dir / "patient_ids_with_no_acquisition.csv")
        metadata[
            metadata.patient_id.isin(patient_ids_with_no_clinical_metadata)
        ].to_csv(processed_dir / "patient_ids_with_no_clinical_metadata.csv")

        patient_ids_to_remove = patient_ids_with_no_acquisition.union(
            patient_ids_with_no_clinical_metadata
        )
        clinical_metadata = clinical_metadata[
            ~clinical_metadata.patient_id.isin(patient_ids_to_remove)
        ]
        metadata = metadata[~metadata.patient_id.isin(patient_ids_to_remove)]

        mask = clinical_metadata.columns.str.contains("date", case=False)
        date_cols = clinical_metadata.columns[mask]
        date_cols = date_cols.tolist() + ["Start Chemo", "End Chemo"]

        clinical_metadata.loc[
            clinical_metadata["End Chemo"] == "31.02.2010", "End Chemo"
        ] = pd.NaT
        for col in date_cols:
            clinical_metadata.loc[:, col] = pd.to_datetime(clinical_metadata[col])

        # metadata
        metadata_full = pd.merge(metadata, clinical_metadata, how="left")
        assert len(metadata) == len(metadata_full)

        # TODO: add your own columns

        # convert to parquet compatible types
        metadata_full = metadata_full.rename(columns=dict(sample_id="roi_id", acquisition_id='sample_id'))
        metadata_full = metadata_full.convert_dtypes()
        metadata_full = metadata_full.set_index("sample_id")
        metadata_full.to_parquet(self.clinical_metadata_path, engine='fastparquet')


    def create_metadata(self):
        # %%
        raw_annotations_path = self.raw_dir / 'metadata' / '02_processed' / 'labels.parquet'
        version_name = self.get_version_name(image_version="filtered", mask_version="cleaned")
        save_dir = self.metadata_dir / version_name
        if save_dir.exists():
            return
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
            labels = pd.read_parquet(raw_annotations_path)
            labels = labels.convert_dtypes()
            labels = labels.astype("category")
            labels.index.names = ["sample_id", "object_id"]

            for grp_name, grp_data in labels.groupby("sample_id"):
                path = save_dir / f"{grp_name}.parquet"
                grp_data.to_parquet(path, engine='fastparquet')

    def create_images(self):
        pass

    def create_masks(self):
        pass

    def create_metadata(self):
        pass

    def create_features_spatial(self):
        pass

    def create_features_intensity(self):
        pass

    def create_features(self):
        self.create_features_spatial()
        self.create_features_intensity()

    def create_panel(self):
        pass

    @property
    def name(self):
        return "BLCa"

    @property
    def id(self):
        return "BLCa"

    @property
    def doi(self):
        return "unpublished"


