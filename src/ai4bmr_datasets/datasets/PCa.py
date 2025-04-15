from pathlib import Path
from loguru import logger
from ai4bmr_core.utils.tidy import tidy_name

import pandas as pd

from .BaseIMCDataset import BaseIMCDataset


class PCa(BaseIMCDataset):

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)

        # raw paths
        self.raw_clinical_metadata_path = (
            base_dir / "01_raw" / "metadata" / "ROI_matching_blockID.xlsx"
        )
        self.raw_tma_annotations_path = (
            base_dir / "01_raw" / "metadata" / "tma-annotations.xlsx"
        )

        self.raw_tma_tidy_path = base_dir / "01_raw" / "metadata" / "TMA_tidy.txt"
        self.raw_annotations_path = Path(
            "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/metadata/02_processed/labels.parquet"
        )

        # populated by `self.load()`
        self.sample_ids = None
        self.samples = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # TODO: this breaks if there is no self.samples
        sample_id = self.samples.index[idx]
        return {
            "sample_id": sample_id,
            "sample": self.samples.loc[sample_id],
            "image": self.images[sample_id],
            "mask": self.masks[sample_id],
            "panel": self.panel,
            "intensity": self.intensity.loc[sample_id],
            "spatial": self.spatial.loc[sample_id],
        }

    def process(self):
        self.create_clinical_metadata()
        self.create_tma_annotations()
        self.create_metadata()
        self.create_annotations()

    def create_masks_filtered_by_annotations(self):
        import numpy as np
        from tifffile import imread
        from ai4bmr_core.utils.saving import save_mask
        import pandas as pd

        def filter_mask(mask: np.ndarray, object_ids: set[int]):
            mapping = lambda x: x if x in object_ids else 0
            return np.vectorize(mapping)(mask)

        mask_dir = self.get_masks_dir(mask_version="cleaned")
        annotations = pd.read_parquet(self.raw_annotations_path)

        new_mask_dir = self.get_masks_dir(mask_version="filtered")
        new_mask_dir.mkdir(parents=True, exist_ok=True)

        sample_ids = set(annotations.index.get_level_values("sample_name"))
        for i, sample_id in enumerate(sample_ids):
            logger.info(f"[{i+1}/{len(sample_ids)}] Processing sample {sample_id}")
            object_ids = set(annotations.loc[sample_id].index)
            mask = imread(mask_dir / f"{sample_id}.tiff")
            mask = filter_mask(mask, object_ids)
            save_mask(mask=mask, save_path=new_mask_dir / f"{sample_id}.tiff")

    def create_annotations(self):
        annotations_dir = self.get_annotations_dir(
            image_version="filtered", mask_version="filtered"
        )
        if annotations_dir.exists():
            return
        else:
            annotations_dir.mkdir(parents=True, exist_ok=True)
            labels = pd.read_parquet(self.raw_annotations_path)
            labels = pd.read_parquet(
                "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/metadata/02_processed/labels.parquet"
            )
            for grp_name, grp_data in labels.groupby("sample_name"):
                grp_data.index.names = ["sample_id", "object_id"]
                grp_data = grp_data.convert_dtypes()
                grp_data = grp_data.astype("category")
                annotations_path = annotations_dir / f"{grp_name}.parquet"
                grp_data.to_parquet(annotations_path)

    def create_images(self):
        pass

    def create_masks(self):
        pass

    def create_metadata(self):
        pass

    def create_graphs(self):
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
        return "PCa"

    @property
    def id(self):
        return "PCa"

    @property
    def doi(self):
        return "unpublished"

    def create_clinical_metadata(self):
        """
        This computes the metadata for each object in the dataset. The results can still contain objects that have been
        removed after the clustering, i.e. all elements that where used in the clustering are included in the metadata.
        """
        if self.clinical_metadata_path.exists():
            return

        # %% read metadata
        roi_match_blockId = pd.read_excel(self.raw_clinical_metadata_path, dtype=str)
        roi_match_blockId = roi_match_blockId.drop(columns=["patient level code"])
        roi_match_blockId = roi_match_blockId.rename(
            columns={"Description (slidecode_coordinates_uniquecoreID)": "description"}
        )

        slide_code = roi_match_blockId["description"].str.split("_").str[0].str.strip()
        tma_coordinates = (
            roi_match_blockId["description"].str.split("_").str[1].str.strip()
        )
        tma_sample_ids = (
            roi_match_blockId["description"].str.split("_").str[2].str.strip()
        )

        roi_match_blockId = roi_match_blockId.assign(
            tma_sample_id=tma_sample_ids,
            slide_code=slide_code,
            tma_coordinates=tma_coordinates,
        )

        roi_match_blockId.loc[
            roi_match_blockId.tma_sample_id == "150 - split", "tma_sample_id"
        ] = "150"
        assert roi_match_blockId.tma_sample_id.str.contains("split").sum() == 0

        patient_ids = (
            roi_match_blockId["Original block number"].str.split("_").str[0].str.strip()
        )
        roi_match_blockId = roi_match_blockId.assign(PAT_ID=patient_ids)

        # NOTE: these are the LP samples (i.e. test runs)
        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=["PAT_ID"])

        assert not roi_match_blockId.slide_code.isna().any()
        assert not roi_match_blockId.tma_coordinates.isna().any()
        assert not roi_match_blockId.tma_sample_id.isna().any()

        tma_id = roi_match_blockId.slide_code + "_" + roi_match_blockId.tma_coordinates
        roi_match_blockId["tma_id"] = tma_id

        tma_tidy = pd.read_csv(self.raw_tma_tidy_path, sep="\t", dtype=str)
        tma_tidy = tma_tidy.drop(columns=["DATE_OF_BIRTH", "INITIALS"])

        tma_tidy.loc[:, "PAT_ID"] = tma_tidy.PAT_ID.astype(str).str.strip()

        # NOTE: the `PATIENT_ID` of the first patient is NA for no obvious reason.
        assert tma_tidy.PATIENT_ID.isna().sum() == 1
        assert tma_tidy["AGE_AT_SURGERY"].isna().any() == False

        metadata = pd.merge(roi_match_blockId, tma_tidy, how="left")
        metadata = metadata.sort_values(["PAT_ID"])

        # NOTE: all patients in the roi_match_block are present in the metadata
        assert metadata.PAT_ID.isna().sum() == 0
        # NOTE: 3 ROIs from 2 patients have no meta data
        assert metadata.DONOR_BLOCK_ID.isna().sum() == 3
        assert metadata[metadata.DONOR_BLOCK_ID.isna()].PAT_ID.nunique() == 2

        # NOTE: for 3 ROIs we do not have patient metadata
        #   thus from the 546 images we have, only 543 are used in downstream analysis
        metadata = metadata[metadata.DONOR_BLOCK_ID.notna()]

        # verify
        # TODO: remove after verification by collaborators
        for i in range(metadata.shape[0]):
            id_ = metadata["tma_sample_id"].values[i]
            ids_ = metadata[
                [
                    "UNIQUE_TMA_SAMPLE_ID_1",
                    "UNIQUE_TMA_SAMPLE_ID_2",
                    "UNIQUE_TMA_SAMPLE_ID_3",
                    "UNIQUE_TMA_SAMPLE_ID_4",
                ]
            ].iloc[i]
            ids_ = set(ids_)
            if id_ not in ids_:
                pass
                print(id_, metadata.PAT_ID.iloc[i])

        filter_ = metadata["Original block number"] == metadata["DONOR_BLOCK_ID"]
        # NOTE: these mismatches are only because we have high and low gleason for the same patient
        mismatches = metadata[
            [
                "PAT_ID",
                "Gleason pattern (TMA core)",
                "Original block number",
                "DONOR_BLOCK_ID",
            ]
        ][~filter_]

        # %% tidy
        metadata = metadata.astype(str)

        metadata.columns = [
            tidy_name(c, split_camel_case=False) for c in metadata.columns
        ]

        metadata["date_of_surgery"] = pd.to_datetime(metadata["date_of_surgery"])

        metadata["age_at_surgery"] = metadata["age_at_surgery"].astype(int)

        mapping = {
            "High Gleason pattern": "high",
            "Low Gleason pattern": "low",
            "Only 1 Gleason pattern": "only_1",
            "unkown": "unknown",
        }
        metadata["gleason_pattern_tma_core"] = metadata.gleason_pattern_tma_core.map(
            mapping
        )

        metadata["psa_at_surgery"] = metadata.psa_at_surgery.astype(float)
        assert metadata["psa_at_surgery"].isna().sum() == 0

        metadata["last_fu"] = metadata["last_fu"].astype(int)
        assert metadata.last_fu.isna().sum() == 0

        metadata["cause_of_death"] = metadata.cause_of_death.map(
            {"0": "alive", "1": "non-PCa_death", "2": "PCa_death"}
        )
        assert metadata.cause_of_death.isna().sum() == 0

        metadata["os_status"] = metadata.os_status.map({"0": "alive", "1": "dead"})
        assert metadata.os_status.isna().sum() == 0

        metadata["psa_progr"] = metadata.psa_progr.astype(int)
        assert metadata.psa_progr.isna().sum() == 0

        metadata["psa_progr_time"] = metadata.psa_progr_time.astype(int)
        assert metadata.psa_progr_time.isna().sum() == 0

        metadata["clinical_progr"] = metadata.clinical_progr.astype(int)
        assert metadata.clinical_progr.isna().sum() == 0

        metadata["clinical_progr_time"] = metadata.clinical_progr_time.astype(int)
        assert metadata.clinical_progr_time.isna().sum() == 0

        metadata["disease_progr"] = metadata.disease_progr.astype(int)
        assert metadata.disease_progr.isna().sum() == 0

        metadata["disease_progr_time"] = metadata.disease_progr_time.astype(int)
        assert metadata.disease_progr_time.isna().sum() == 0

        metadata["recurrence_loc"] = metadata.recurrence_loc.map(
            {
                "0": "no_recurrence",
                "1": "local",
                "2": "metastasis",
                "3": "local_and_metastasis",
            }
        )
        assert metadata.recurrence_loc.isna().sum() == 0

        mapping = {
            i: "recurrence" for i in ["local", "local_and_metastasis", "metastasis"]
        }
        metadata["recurrence"] = metadata["recurrence_loc"].astype(str).replace(mapping)

        # NOTE: we do not have the mapping at this point
        # metadata['cgrading_biopsy'] = metadata.cgrading_biopsy.astype(int)

        metadata["cgs_pat_1"] = metadata.cgs_pat_1.astype(int)
        assert metadata.cgs_pat_1.isna().sum() == 0

        metadata["cgs_pat_2"] = metadata.cgs_pat_2.astype(int)
        assert metadata.cgs_pat_2.isna().sum() == 0

        metadata["cgs_score"] = metadata.cgs_score.astype(int)
        assert metadata.cgs_score.isna().sum() == 0

        # metadata['gs_grp'] = metadata.gs_grp.map({'1': '<=6', '2': '3+4=7', '3': '4+3=7', '4': '8', '5': '9/10'})
        # assert metadata.gs_grp.isna().sum() == 0

        metadata["pgs_score"] = metadata.pgs_score.astype(int)
        assert metadata.pgs_score.isna().sum() == 0

        metadata["ln_status"] = metadata.ln_status.astype(int)
        assert metadata.ln_status.isna().sum() == 0

        metadata["surgical_margin_status"] = metadata.surgical_margin_status.astype(
            float
        )
        assert metadata.surgical_margin_status.astype(float).isna().sum() == 87

        metadata["d_amico_risk"] = metadata["d_amico"].map(
            {"Intermediate Risk": "intermediate", "High Risk": "high"}
        )
        metadata = metadata.drop("d_amico", axis=1)

        metadata = metadata.convert_dtypes()

        # %%
        # vals = metadata.cause_of_death.astype(int).values
        # cats = sorted(set(vals))
        # metadata['cause_of_death'] = pd.Categorical(vals, categories=cats, ordered=True)
        # assert metadata.cause_of_death.isna().sum() == 0

        # %%
        sample_names = [Path(i).stem for i in metadata.file_name_napari]
        metadata = metadata.assign(sample_id=sample_names)
        metadata = metadata.set_index("sample_id")

        # %%
        cat_cols = [
            "file_name_napari",
            "original_block_number",
            "gleason_pattern_tma_core",
            "tma_sample_id",
            "slide_code",
            "pat_id",
            "patient_id",
            "donor_block_id",
            "unique_tma_sample_id_1",
            "unique_tma_sample_id_2",
            "unique_tma_sample_id_3",
            "unique_tma_sample_id_4",
            "cause_of_death",
            "os_status",
            "psa_progr",
            "clinical_progr",
            "disease_progr",
            "recurrence",
            "recurrence_loc",
            "cgrading_biopsy",
            "cgs_pat_1",
            "cgs_pat_2",
            "cgs_score",
            "gs_grp",
            "ct_stage",
            "pt_stage",
            "pgs_score",
            "ln_status",
            "surgical_margin_status",
            "adj_adt",
            "adj_radio",
            "d_amico_risk",
        ]
        metadata = metadata.astype({k: "category" for k in cat_cols})

        # %%
        self.sample_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.sample_metadata_path, engine="fastparquet")

    def create_tma_annotations(self):
        samples = pd.read_parquet(self.sample_metadata_path, engine="fastparquet")
        df = pd.read_excel(self.raw_tma_annotations_path, sheet_name="PCa_ROIs_final")
        # set(samples.index) - set(df.sample_name)

        df.columns = df.columns.map(tidy_name)
        df = df.rename(columns={"unnamed_12": "notes"})
        df["notes"] = df.notes.astype(str).str.strip()

        filter_ = df.sample_name == "sample_name"
        df = df[~filter_]

        filter_ = df.sample_name.notna()
        df = df[filter_]

        mapping = {"y": "yes", "n": "no", "/": "unclear_annotation", "nan": pd.NA}
        df = df.map(lambda x: mapping.get(x, x))

        # df.gs_pat_1.unique()
        mapping = {
            # nan, 4,
            "tumor not evident": "no tumor",
            # 3,
            # 'no tumor',
            # 'not enough material',
            # 'no tissue',
            # 'not sure if tumor',
            "Tumor??": "not sure if tumor",
            # 'not enough tumor',
            # 'too much loss',
            # 5,
            "no evident tumor": "no tumor",
            "n": "unclear annotation",
            "no": "unclear annotation",
        }
        df["gs_pat_1"] = df.gs_pat_1.map(lambda x: mapping.get(x, x))

        # df.gs_pat_2.unique()
        mapping = {
            # nan, 4, 3,
            "4 (3 is also an option)": 4,
            # 5,
            "/": "unclear_annotation",
            "4(few)": 4,
        }
        df["gs_pat_2"] = df.gs_pat_2.map(lambda x: mapping.get(x, x))

        mapping = {"yes,": "yes", "nyes": "yes"}
        df["glandular_atrophy_pin"] = df.glandular_atrophy_pin.map(
            lambda x: mapping.get(x, x)
        )

        mapping = {"yed": "yes"}
        df["non_stromogenic_smc_abundant"] = df.non_stromogenic_smc_abundant.map(
            lambda x: mapping.get(x, x)
        )

        mapping = {"yno": "unclear_annotation"}
        df["inflammation"] = df.inflammation.map(lambda x: mapping.get(x, x))

        mapping = {"n3": "no"}
        df["cribriform"] = df.cribriform.map(lambda x: mapping.get(x, x))

        tidy_col_vals = [
            "gs_pat_1",
            "gs_pat_2",
            "stromogenic_smc_loss_reactive_stroma_present",
            "non_stromogenic_smc_abundant",
            "inflammation",
            "glandular_atrophy_pin",
            "cribriform",
        ]
        for col in tidy_col_vals:
            df[col] = df[col].astype(str).map(tidy_name)

        filter_ = df.isin(
            [
                "not_enough_material",
                "unclear_annotation",
                "no_tissue",
                "not_sure_if_tumor",
                "not_enough_tumor",
                "too_much_loss",
            ]
        ).sum(axis=1)
        filter_ = filter_ > 0
        df = df[~filter_]

        mapping = {"nan": pd.NA}
        df = df.map(lambda x: mapping.get(x, x))

        filter_ = df.gs_pat_1.notna()
        df = df[filter_]

        mapping = {"nan": pd.NA, None: pd.NA}
        df = df.map(lambda x: mapping.get(x, x))

        df = df.convert_dtypes()
        df = df.rename(columns={"sample_name": "sample_id"}).set_index("sample_id")
        # df.to_csv('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-datasets/tests/tma_anno.csv')

        # sample_id_to_tma_id = samples['tma_id'].to_dict()
        # df['tma_id'] = df.index.map(sample_id_to_tma_id)
        # assert not df.tma_id.isna().any()

        # NOTE: check that the data in samples and the data in the tma annotations are the same
        shared_sample_ids = list(set(df.index).intersection(set(samples.index)))
        for i in ["tma_coordinates", "tma_sample_id"]:
            s1 = set(samples.loc[shared_sample_ids][i].astype(str).to_dict().items())
            s2 = set(df.loc[shared_sample_ids][i].astype(str).to_dict().items())
            assert s1 == s2

        set(samples).intersection(set(df))
        shared_cols = ["description", "tma_sample_id", "tma_coordinates"]
        df = df.drop(columns=shared_cols)

        # note: for some reason we cannot store boolean as a category, we use yes/no instead
        df["is_tumor"] = "yes"
        df.loc[df.gs_pat_1 == "no_tumor", "is_tumor"] = "no"

        mapping = {"no_tumor": pd.NA, "3": 3, "4": 4, "5": 5}
        df["gs_pat_1"] = df.gs_pat_1.map(lambda x: mapping.get(x, x))

        mapping = {"no_tumor": pd.NA, "3": 3, "4": 4, "5": 5}
        df["gs_pat_2"] = df.gs_pat_2.map(lambda x: mapping.get(x, x))

        df["gleason_score"] = pd.NA
        gleason_score = df.gs_pat_1.astype(str) + "+" + df.gs_pat_2.astype(str)

        filter_ = df.gs_pat_1.notna()
        df.loc[filter_, "gleason_score"] = gleason_score[filter_]

        gleason_score_sum = df.gs_pat_1[filter_].astype(int) + df.gs_pat_2[
            filter_
        ].astype(int)
        df["gleason_score_sum"] = pd.NA
        df.loc[filter_, "gleason_score_sum"] = gleason_score_sum

        df["gleason_grp"] = pd.NA

        filter_ = df.gleason_score_sum <= 6
        df.loc[filter_, "gleason_grp"] = 1

        filter_ = df.gleason_score == "3+4"
        df.loc[filter_, "gleason_grp"] = 2

        filter_ = df.gleason_score == "4+3"
        df.loc[filter_, "gleason_grp"] = 3

        filter_ = df.gleason_score_sum == 8
        df.loc[filter_, "gleason_grp"] = 4

        filter_ = df.gleason_score_sum >= 9
        df.loc[filter_, "gleason_grp"] = 5

        df = df.convert_dtypes()

        cat_cols = [
            "annotation_roi_he",
            "gs_pat_1",
            "gs_pat_2",
            "stromogenic_smc_loss_reactive_stroma_present",
            "non_stromogenic_smc_abundant",
            "inflammation",
            "glandular_atrophy_pin",
            "cribriform",
            "is_tumor",
            "gleason_grp",
            "gleason_score_sum",
            "gleason_score",
        ]

        df = df.astype({k: "category" for k in cat_cols})
        samples = pd.concat([samples, df], axis=1)
        samples.to_parquet(self.sample_metadata_path, engine="fastparquet")
