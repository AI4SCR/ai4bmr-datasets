import re
from pathlib import Path

import numpy as np
import pandas as pd
from ai4bmr_core.utils.saving import save_image, save_mask
from ai4bmr_core.utils.tidy import tidy_name
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tifffile import imread
from loguru import logger

from .BaseIMCDataset import BaseIMCDataset


class TNBC(BaseIMCDataset):
    """
    Download data from https://www.angelolab.com/mibi-data, unzip and place in `base_dir/01_raw`.
    Save Ionpath file as tnbc.
    The supplementary_table_1_path is downloaded as Table S1 from the paper.
    """

    # TODO: check why patient_id 30 is missing in `data` ( update: due to noise)
    #   but we have patient ids up to 44 in `raw_sca`.
    #   we only have images for patient ids up to 41

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.version_name = "published"

        # raw data paths
        self.raw_masks_dir = self.raw_dir / "TNBC_shareCellData"
        self.raw_images_dir = self.raw_dir / "tnbc"
        self.supplementary_table_1_path = (
            self.raw_dir / "1-s2.0-S0092867418311000-mmc1.xlsx"
        )  # downloaded as Table S1 from paper
        self.raw_sca_path = self.raw_dir / "TNBC_shareCellData" / "cellData.csv"
        self.patient_class_path = (
            self.raw_dir / "TNBC_shareCellData" / "patient_class.csv"
        )

        # containers for processing of features
        self.observations_default = set()
        self.observations_unfiltered = set()

    def prepare_data(self):
        self.create_panel()
        self.create_metadata()
        self.create_images()
        self.create_masks()
        self.create_features()

    def create_panel(self):
        logger.info("Creating panel")

        panel_1 = pd.read_excel(self.supplementary_table_1_path, header=3)[:31]
        panel_1.columns = panel_1.columns.str.strip()
        panel_1 = panel_1.rename(columns={"Antibody target": "target"})
        panel_2 = pd.read_excel(self.supplementary_table_1_path, header=37)
        panel_2.columns = panel_2.columns.str.strip()
        panel_2 = panel_2.rename(columns={"Label": "target", "Titer": "Titer (µg/mL)"})
        panel = pd.concat([panel_1, panel_2])

        panel = panel.rename(columns={"Mass channel": "mass_channel"})

        # note: Biotin has not Start,Stop,NoiseT
        #   for PD-L1, we replace the Mass channel and Isotope with the values in Biotin
        panel.loc[panel.target == "PDL1-biotin", "mass_channel"] = panel.loc[
            panel.target == "Biotin", "mass_channel"
        ].iloc[0]
        panel.loc[panel.target == "PDL1-biotin", "Isotope"] = panel.loc[
            panel.target == "Biotin", "Isotope"
        ].iloc[0]
        panel = panel[panel.Start.notna()]

        target_map = {"PDL1-biotin": "PD-L1", "Ki-67": "Ki67"}
        panel = panel.assign(target=panel.target.replace(target_map))

        panel = panel.sort_values("mass_channel", ascending=True)
        # panel = panel.assign(page=range(len(panel)))
        panel = panel.assign(
            target_original_name=panel.target, target=panel.target.map(tidy_name)
        )

        # extract original page numbers for markers
        img_path = sorted(
            filter(
                lambda f: not f.name.startswith("."), self.raw_images_dir.glob("*.tiff")
            )
        )[0]
        metadata = self.get_tiff_metadata(img_path)
        metadata = metadata.rename(columns={"target": "target_from_tiff"}).reset_index()

        panel = panel.merge(metadata, on="mass_channel", how="left")
        assert panel.isna().any().any() == False
        panel = panel.sort_values("page").reset_index(drop=True)

        panel.index.name = "channel_index"
        panel = panel.convert_dtypes()

        panel_path = self.get_panel_path(image_version=self.version_name)
        panel_path.parent.mkdir(exist_ok=True, parents=True)
        panel.to_parquet(panel_path, engine='fastparquet')

    @staticmethod
    def get_tiff_metadata(path):
        import tifffile

        tag_name = "PageName"
        metadata = pd.DataFrame()
        metadata.index.name = "page"
        with tifffile.TiffFile(path) as tif:
            for page_num, page in enumerate(tif.pages):
                page_name_tag = page.tags.get(tag_name)
                metadata.loc[page_num, tag_name] = page_name_tag.value
        metadata = metadata.assign(PageName=metadata.PageName.str.strip())
        metadata = metadata.PageName.str.extract(
            r"(?P<target>\w+)\s+\((?P<mass_channel>\d+)"
        )

        metadata.mass_channel = metadata.mass_channel.astype("int")
        metadata = metadata.convert_dtypes()

        return metadata

    def create_images(self):
        logger.info("Creating images")

        panel_path = self.get_panel_path(image_version=self.version_name)
        panel = pd.read_parquet(panel_path)

        images_dir = self.get_image_version_dir(self.version_name)
        images_dir.mkdir(exist_ok=True, parents=True)

        image_paths = sorted(
            filter(
                lambda f: not f.name.startswith("."), self.raw_images_dir.glob("*.tiff")
            )
        )
        for img_path in image_paths:
            logger.debug(f"processing image {img_path.stem}")
            sample_id = re.search(r"Point(\d+).tiff", img_path.name).groups()[0]

            if (images_dir / f"{sample_id}.tiff").exists():
                continue

            img = imread(img_path)
            img = img[panel.page.values]  # extract only the layers of the panel
            assert len(img) == len(panel)

            save_image(img, images_dir / f"{sample_id}.tiff")

    def create_masks(self):
        from PIL import Image

        logger.info("Filtering masks")

        masks_default_dir = self.get_mask_version_dir(mask_version=self.version_name)
        masks_default_dir.mkdir(exist_ok=True, parents=True)

        masks_unfiltered_dir = self.get_mask_version_dir(mask_version="unfiltered")
        masks_unfiltered_dir.mkdir(exist_ok=True, parents=True)

        mask_paths = sorted(list(self.raw_masks_dir.glob(r"p*_labeledcellData.tiff")))

        data = pd.read_csv(self.raw_sca_path)
        data = data.rename(
            columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"}
        )
        data.sample_id = data.sample_id.astype(str)
        data = data.set_index("sample_id")[["object_id"]]

        sample_ids = set(data.index.get_level_values("sample_id"))

        for mask_path in mask_paths:
            sample_id = re.match(r"p(\d+)", mask_path.stem).groups()
            assert len(sample_id) == 1
            sample_id = sample_id[0]

            logger.info(f"processing sample {sample_id}")

            path_default = masks_default_dir / f"{sample_id}.tiff"
            path_unfiltered = masks_unfiltered_dir / f"{sample_id}.tiff"

            if not path_unfiltered.exists():
                # load segmentation from processed data
                mask_unfiltered = imread(mask_path)

                # load segmentation mask from raw data
                segm = Image.open(self.raw_images_dir
                    / f"TA459_multipleCores2_Run-4_Point{sample_id}"
                    / "segmentation_interior.png")
                segm = np.array(segm)

                # note: set region with no segmentation to background
                mask_unfiltered[segm == 0] = 0
                assert 1 not in mask_unfiltered.flat

                save_mask(mask_unfiltered, path_unfiltered)
            else:
                mask_unfiltered = imread(path_unfiltered)

            # we record all observations to filter the raw single cell annotations data
            self.observations_unfiltered.update(
                set((sample_id, i) for i in set(mask_unfiltered.flat))
            )

            # NOTE: for the default masks we only compute the masks for the samples that are in the processed data
            if sample_id not in sample_ids:
                continue

            if not path_default.exists():
                # NOTE: filter masks by objects in data
                #   - 0: cell boarders
                #   - 1: background
                #   - 2..N: cells
                #   (19, 4812) is background in the mask and removed after removing objects not in `data`
                mask_default = imread(mask_path)

                objs_in_data = set(data.loc[sample_id, :].object_id)
                removed_objects = set(mask_default.flat) - set(objs_in_data) - {0, 1}

                map = {i: 0 for i in removed_objects}
                vfunc = np.vectorize(lambda x: map.get(x, x))
                mask_default = vfunc(mask_default)
                mask_default[mask_default == 1] = 0

                assert set(mask_default.flat) - objs_in_data == {0}
                assert 1 not in mask_default.flat

                save_mask(mask_default, path_default)
            else:
                mask_default = imread(path_unfiltered)

            # we record all observations to filter the raw single cell annotations data
            self.observations_default.update(
                set((sample_id, i) for i in set(mask_default.flat))
            )

            if False:
                # diagnostic plot
                # FIXME: what do to with this?
                # note: map objects in mask that are not present in data to 2
                objs_in_segm = set(mask[segm == 1])
                map = {i: 2 for i in objs_in_segm - objs_in_data}
                assert objs_in_data - objs_in_segm == set()
                map.update({i: 1 for i in objs_in_data.intersection(objs_in_segm)})
                z = mask.copy()
                z[segm != 1] = 0
                vfunc = np.vectorize(lambda x: map.get(x, x))
                z = vfunc(z)

                fig, axs = plt.subplots(2, 2, dpi=400)
                axs = axs.flat

                axs[0].imshow(mask > 1, interpolation=None, cmap="grey")
                axs[0].set_title("Segmentation from processed data", fontsize=8)

                axs[1].imshow(mask_filtered > 0, interpolation=None, cmap="grey")
                axs[1].set_title("Segmentation from raw data", fontsize=8)

                axs[2].imshow(z, interpolation=None)
                axs[2].set_title("Objects missing in `cellData.csv`", fontsize=8)

                axs[3].imshow(mask_filtered_v2 > 0, interpolation=None, cmap="grey")
                axs[3].set_title(
                    "Segmentation from processed\nfiltered with `cellData.csv`",
                    fontsize=8,
                )

                for ax in axs:
                    ax.set_axis_off()
                fig.tight_layout()
                fig.savefig(self.misc / f"{sample_id}.png")
                plt.close(fig)

    def process_clinical_metadata(self):
        logger.info("Creating metadata [samples]")
        samples = pd.read_csv(self.patient_class_path, header=None)
        samples.columns = ["patient_id", "cancer_type_id"]
        samples = samples.assign(sample_id=samples["patient_id"]).set_index("sample_id")
        samples.index = samples.index.astype(str)

        cancer_type_map = {0: "mixed", 1: "compartmentalized", 2: "cold"}
        samples = samples.assign(
            cancer_type=samples["cancer_type_id"].map(cancer_type_map)
        )

        path = self.clinical_metadata_path
        path.parent.mkdir(exist_ok=True, parents=True)
        samples.to_parquet(path, engine='fastparquet')

    def process_annotations(self):
        logger.info("Creating metadata [annotations]")

        data = pd.read_csv(self.raw_sca_path)

        metadata_cols = [
            "SampleID",
            "cellSize",
            "tumorYN",
            "tumorCluster",
            "Group",
            "immuneCluster",
            "immuneGroup",
        ]
        metadata = data[metadata_cols]

        metadata = metadata.rename(columns={"SampleID": "sample_id"})
        metadata.sample_id = metadata.sample_id.astype(str)
        metadata = metadata.set_index("sample_id")

        grp_map = {
            1: "unidentified",
            2: "immune",
            3: "endothelial",
            4: "mesenchymal_like",
            5: "tumor",
            6: "keratin_positive_tumor",
        }
        metadata = metadata.assign(group_name=metadata["Group"].map(grp_map))

        immune_grp_map = {
            1: "Tregs",
            2: "CD4 T",
            3: "CD8 T",
            4: "CD3 T",
            5: "NK",
            6: "B",
            7: "Neutrophils",
            8: "Macrophages",
            9: "DC",
            10: "DC/Mono",
            11: "Mono/Neu",
            12: "Other immune",
        }
        metadata = metadata.assign(
            immune_group_name=metadata["immuneGroup"].map(immune_grp_map)
        )

        col_map = dict(
            cellSize="cell_size",
            tumorYN="tumor_yn",
            tumorCluster="tumor_cluster_id",
            Group="group_id",
            immuneCluster="immune_cluster_id",
            immuneGroup="immune_group_id",
        )
        metadata = metadata.rename(columns=col_map)
        metadata = metadata.assign(tumor_yn=metadata.tumor_yn.astype("bool"))

        label_name = metadata.group_name.copy()
        filter = metadata.immune_group_name.notna()
        label_name[filter] = metadata.immune_group_name[filter]
        metadata = metadata.assign(label_name=label_name)
        label_id_map = dict(zip(label_name.unique(), range(len(label_name.unique()))))
        metadata = metadata.assign(label_id=metadata.label_name.map(label_id_map))

        metadata = metadata.convert_dtypes()

        for grp_name, grp_data in metadata.groupby("sample_id"):
            path = self.metadata_dir / self.version_name / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

    def create_metadata(self):
        self.process_clinical_metadata()
        self.process_annotations()

    def create_graphs(self):
        pass

    def create_features_spatial(self):
        logger.info("Creating features [spatial]")

        samples = pd.read_parquet(self.clinical_metadata_path)
        spatial = pd.read_csv(self.raw_sca_path)

        index_columns = ["sample_id", "object_id"]
        feat_cols_names = ["cellSize"]
        spatial = spatial.rename(
            columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"}
        )
        spatial.sample_id = spatial.sample_id.astype(str)
        spatial = spatial.set_index(index_columns)

        spatial = spatial[feat_cols_names]
        spatial.columns = spatial.columns.map(tidy_name)
        spatial = spatial.convert_dtypes()

        valid_obs = set(spatial.index).intersection(self.observations_default)
        default = spatial.loc[list(valid_obs), :]

        valid_samples = set(default.index.get_level_values("sample_id")).intersection(
            samples.index
        )
        default = default.loc[list(valid_samples)]

        for grp_name, grp_data in default.groupby("sample_id"):
            dir = self.spatial_dir / self.version_name
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

        del default

        valid_obs = set(spatial.index).intersection(self.observations_unfiltered)
        unfiltered = spatial.loc[list(valid_obs), :]

        valid_samples = set(
            unfiltered.index.get_level_values("sample_id")
        ).intersection(samples.index)
        unfiltered = unfiltered.loc[list(valid_samples)]

        for grp_name, grp_data in unfiltered.groupby("sample_id"):
            dir = self.intensity_dir / self.version_name
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

    def create_features_intensity(self):
        logger.info("Creating features [intensities]")

        samples = pd.read_parquet(self.clinical_metadata_path)
        panel_path = self.get_panel_path(image_version=self.version_name)
        panel = pd.read_parquet(panel_path)
        intensity = pd.read_csv(self.raw_sca_path)

        index_columns = ["sample_id", "object_id"]
        intensity = intensity.rename(
            columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"}
        )
        intensity.sample_id = intensity.sample_id.astype(str)
        intensity = intensity.set_index(index_columns)

        intensity.columns = intensity.columns.map(tidy_name)

        intensity = intensity[panel.target]
        assert intensity.isna().sum().sum() == 0

        intensity = intensity.convert_dtypes()

        # filter for samples that are in both processed data and patient data
        # NOTE: patient 30 has been excluded from the analysis due to noise
        #   and there are patients 42,43,44 in the processed single cell data but we do not have images for them
        valid_obs = set(intensity.index).intersection(self.observations_default)
        default = intensity.loc[list(valid_obs), :]

        valid_samples = set(default.index.get_level_values("sample_id")).intersection(
            samples.index
        )
        default = default.loc[list(valid_samples)]

        for grp_name, grp_data in default.groupby("sample_id"):
            dir = self.intensity_dir / self.version_name
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

    @property
    def name(self):
        return "TNBC"

    @property
    def id(self):
        return "TNBC-v2"

    @property
    def doi(self):
        return "10.1016/j.cell.2018.08.039"


# %%
# TODO: we still need to implement the single cell value extraction from the images based on the masks!
# at the moment we simply use the processed data from the publication.
def reproduce_paper_figure_2E(self) -> Figure:
    """Try to reproduce the figure from the paper. Qualitative comparison looks good if data is loaded from"""

    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import colorcet as cc

    data = self.load()
    df = data["data"]

    # %%
    cols = [
        "FoxP3",
        "CD4",
        "CD3",
        "CD56",
        "CD209",
        "CD20",
        "HLA-DR",
        "CD11c",
        "CD16",
        "CD68",
        "CD11b",
        "MPO",
        "CD45",
        "CD31",
        "SMA",
        "Vimentin",
        "Beta catenin",
        "Pan-Keratin",
        "p53",
        "EGFR",
        "Keratin17",
        "Keratin6",
    ]
    pdat = df[cols]
    pdat = pdat[pdat.index.get_level_values("sample_id") <= 41]

    pdat, row_colors = pdat.align(data["metadata"], join="left", axis=0)
    pdat = pdat.assign(
        group_name=pd.Categorical(
            row_colors.group_name,
            categories=[
                "immune",
                "endothelial",
                "Mmsenchymal_like",
                "tumor",
                "keratin_positive_tumor",
                "unidentified",
            ],
        )
    )
    pdat = pdat.assign(
        immune_group_name=pd.Categorical(
            row_colors.immune_group_name,
            categories=[
                "Tregs",
                "CD4 T",
                "CD8 T",
                "CD3 T",
                "NK",
                "B",
                "Neutrophils",
                "Macrophages",
                "DC",
                "DC/Mono",
                "Mono/Neu",
                "Other immune",
            ],
        )
    )
    pdat = pdat.set_index(["group_name", "immune_group_name"], append=True)

    color_map = {
        "immune": np.array((187, 103, 30, 0)) / 255,
        "endothelial": np.array((253, 142, 142, 0)) / 255,
        "Mmsenchymal_like": np.array((1, 56, 54, 0)) / 255,
        "tumor": np.array((244, 201, 254, 0)) / 255,
        "keratin_positive_tumor": np.array((128, 215, 230, 0)) / 255,
        "unidentified": np.array((255, 255, 255, 0)) / 255,
    }
    row_colors = row_colors.assign(group_name=row_colors.group_name.map(color_map))

    color_map = {
        k: cc.glasbey_category10[i]
        for i, k in enumerate(row_colors.immune_group_name.unique())
    }
    row_colors = row_colors.assign(
        immune_group_name=row_colors.immune_group_name.map(color_map)
    )
    row_colors = row_colors[["group_name", "immune_group_name"]]

    pdat = pdat.sort_values(["group_name", "immune_group_name"])

    # %% filter
    pdat = pdat[pdat.index.get_level_values("group_name") == "immune"]
    pdat = pdat[
        [
            "FoxP3",
            "CD4",
            "CD3",
            "CD56",
            "CD209",
            "CD20",
            "HLA-DR",
            "CD11c",
            "CD16",
            "CD68",
            "CD11b",
            "MPO",
            "CD45",
        ]
    ]

    # %%
    pdat = pd.DataFrame(
        MinMaxScaler().fit_transform(pdat), index=pdat.index, columns=pdat.columns
    )
    cg = sns.clustermap(
        pdat,
        cmap="bone",
        row_colors=row_colors,
        col_cluster=False,
        row_cluster=False,
        # figsize=(20, 20)
    )
    cg.ax_heatmap.set_facecolor("black")
    cg.ax_heatmap.set_yticklabels([])
    cg.figure.tight_layout()
    cg.figure.show()
    return cg.figure
