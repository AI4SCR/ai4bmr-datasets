import json
import subprocess
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import openslide
import pandas as pd
import tifffile
import torch
from ai4bmr_core.utils.tidy import tidy_name
from ai4bmr_learn.data_models.Coordinate import SlideCoordinate
from ai4bmr_learn.datasets.items import SlidePatches, get_patch
from ai4bmr_learn.plotting.patches import visualize_coords
from ai4bmr_learn.utils.device import batch_to_device
from ai4bmr_learn.utils.images import filter_coords
from ai4bmr_learn.utils.slides import get_coordinates_dict, get_mpp, get_mpp_and_resolution, get_seg_model, \
    get_slide_patcher_params, get_thumbnail, segment_slide
from loguru import logger
from openslide import OpenSlide
from skimage.io import imsave
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm
from trident.patch_encoder_models import encoder_factory
from ai4bmr_learn.plotting.umap import plot_umap


class BEAT:
    INVALID_SAMPLE_IDS = {
        # '1GUG.0M.2',  # barcode
        # '1GVQ.06.2',  # barcode
        # '1GUG.0M.0',  # mpp_x != mpp_y
        # '1GVQ.06.1',  # mpp_x != mpp_y
    }
    PATH_TO_TRIDENT = Path("/users/mensmeng/workspace/dermatology/TRIDENT_pipeline/new_install/trident/trident")

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned')
        self.raw_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/HnE/BEAT-MESO_rescan_40x_preXenium')
        self.processed_dir = self.base_dir / '02_processed'
        self.dataset_dir = self.processed_dir / 'datasets' / 'beat'
        self.tools_dir = self.base_dir / '03_tools'

        # GLOBAL PARAMS
        # self.target_mpp = 0.8624999831542972
        self.target_mpp = 0.
        self.overlap = 0.25

    def prepare_tools(self):
        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        if bfconvert.exists():
            logger.info(f"bfconvert found at {bfconvert}. Skipping installation.")
            return

        logger.info("⚙️Installing Bio-Formats tools.")
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        install_cmd = f"""
        wget https://downloads.openmicroscopy.org/bio-formats/8.2.0/artifacts/bftools.zip -O {self.tools_dir}/bftools.zip
        unzip {self.tools_dir}/bftools.zip -d {self.tools_dir}
        {self.tools_dir / "bftools" / "bfconvert"} -version || exit
        """

        install_cmd = textwrap.dedent(install_cmd).strip()
        subprocess.run(install_cmd, shell=True, check=True)

    def prepare_metadata(self):
        def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
            df = df[df['tech_id'] == 'HE']
            df = df[df['include'] == True]

            ## remove columns where all values are NA
            df = df.dropna(axis=1, how='all')
            assert df.index.is_unique
            assert df.sample_path.is_unique
            assert df.sample_path.isna().sum() == 0
            return df
        


        clinical = pd.read_csv('/users/mensmeng/workspace/beat_meso/metadata_beat_meso_final_rescanned.csv')
        clinical = filter_metadata(clinical)
        clinical = clinical.set_index('sample_id')

        save_path = self.dataset_dir / 'metadata.parquet'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        clinical.to_parquet(save_path, engine='fastparquet')

    def get_metadata(self):
        return pd.read_parquet(self.dataset_dir / 'metadata.parquet', engine='fastparquet')

    def get_sample_ids(self):
        clinical = self.get_metadata()
        sample_ids = set(clinical.index.tolist()) - self.INVALID_SAMPLE_IDS
        return sample_ids

    def get_wsi_path(self, sample_id: str):
        clinical = self.get_metadata()
        wsi_path = clinical[clinical.index == sample_id].sample_path.item()
        assert Path(wsi_path).exists()
        return wsi_path
    
    def get_wsi_name(self, sample_id: str):
        clinical = self.get_metadata()
        wsi_name = clinical[clinical.index == sample_id].block_id.item()
        return wsi_name

    def check_wsi(self, sample_id: str, force: bool = False):

        if sample_id in self.INVALID_SAMPLE_IDS:
            return
        
        wsi_path = self.get_wsi_path(sample_id=sample_id)

        try:
            slide = openslide.OpenSlide(wsi_path)
            mpp_x, mpp_y = get_mpp_and_resolution(slide)
            assert np.isclose(mpp_x, mpp_y), f'mpp_x={mpp_x} != mpp_y={mpp_y}'
            logger.info(f'{sample_id} has mpp={mpp_x:.4f}')
        except Exception as e:
            logger.error(f'Error opening {wsi_path} for {sample_id}: {e}')
            self.INVALID_SAMPLE_IDS.add(sample_id)
            return
        

    def prepare_wsi_trident(self, sample_id: str, force: bool = False):

        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id)
        ### call job script and pass parameters


        ###### segmentation
        from utils.wsi.segment import segment_wsi
        seg_dir = self.dataset_dir / sample_id / 'contours' / 'trident'
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_wsi(wsi_path=wsi_path, wsi_name=wsi_name, seg_model='hest', seg_dir=seg_dir, force_rerun=force)


        ###### patching
        

        pass


        
    ### legacy code, for custom segmentation
    def segment(self, sample_id: str, model_name: str = 'grandqc', target_mpp: float = 4.0):
        logger.info(f'🧩 Segmenting {sample_id} with {model_name}')

        seg_model = get_seg_model(model_name=model_name)  # hest, grandqc, grandqc_artifact
        sample_dir = self.dataset_dir / sample_id
        img_path = sample_dir / 'wsi.tiff'

        if not img_path.exists():
            return

        version_name = f'segment-mpp={target_mpp}-model_name={model_name}'
        save_coords_path = sample_dir / 'coords' / f'{version_name}.json'
        save_contours_path = sample_dir / 'contours' / f'{version_name}.parquet'

        if save_contours_path.exists() and save_contours_path.exists():
            logger.info(f'{sample_id} is already segmented')
            return

        slide = OpenSlide(img_path)

        try:
            mpp = get_mpp_and_resolution(slide)[0]
        except AssertionError:
            logger.error(f'Error retrieving slide mpp for {img_path}')
            return

        logger.info(f'{sample_dir.name} reports mpp={mpp:.4f}')

        if save_contours_path.exists():
            return

        segment_slide(slide,
                      seg_model=seg_model,
                      target_mpp=target_mpp,
                      min_area=10000,
                      save_contours_path=save_contours_path,
                      save_coords_path=save_coords_path)

    ### legacy code, for custom segmentation
    def create_thumbnail(self, sample_id: str):

        img_path = self.dataset_dir / sample_id / 'wsi.tiff'
        if not img_path.exists():
            return

        logger.info(f'Create thumbnail for {sample_id}')

        slide = OpenSlide(img_path)
        thumbnail, _ = get_thumbnail(slide=slide)

        save_path = img_path.parent / 'thumbnail.png'
        imsave(str(save_path), thumbnail)

    ### legacy code, for custom segmentation
    def filter_contours(self):
        # TODO: potentially post-process contours or improve filtering during segmentation.
        # NOTE: we only use contours that are larger than the median area of all found contours
        # q = np.quantile(contours.geometry.area, 0.5)
        # filter_ = contours.geometry.area > q
        # import matplotlib.pyplot as plt
        # plt.imshow(visualize_contours(contours=contours[filter_], slide=slide)).figure.show()
        pass
    

    ### legacy code, for custom patching
    def create_coords(self, sample_id: str, patch_size: int = 512, model_name: str = 'hest'):
        patch_stride = patch_size
        target_mpp = self.target_mpp
        overlap = self.overlap
        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'
        logger.info(f'Create coords version {coords_version} for {sample_id}.')

        wsi_path = self.dataset_dir / sample_id / 'wsi.tiff'
        coords_path = self.dataset_dir / sample_id / 'coords' / f"{coords_version}.json"

        if coords_path.exists():
            logger.info(f'Found coords at {coords_path}.')
            return

        contours_path = self.dataset_dir / sample_id / 'contours' / f"segment-mpp=4.0-model_name={model_name}.parquet"
        contours = gpd.read_parquet(contours_path)
        logger.info(f'{sample_id} has {len(contours)} contours.')

        slide = OpenSlide(str(wsi_path))
        params = get_slide_patcher_params(slide=slide, patch_size=patch_size, patch_stride=patch_stride,
                                          target_mpp=target_mpp)
        coords = get_coordinates_dict(**params)
        logger.info(f'{len(coords)} coordinates before filtering')
        coords = [SlideCoordinate(**i) for i in coords]

        logger.info('Filtering coordinates...')
        coords = filter_coords(coords=coords, contours=contours, overlap=overlap)

        coords_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coords_path, 'w') as f:
            coords_dict = [i.model_dump() for i in coords]
            json.dump(coords_dict, f)

    def visualize_patches(self, sample_id: str, patch_size: int = 512, num_patches: int = 100):
        logger.info(f'Visualizing patches for {sample_id}')

        patch_stride = patch_size
        target_mpp = self.target_mpp
        overlap = self.overlap

        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'
        coords_path = self.dataset_dir / sample_id / 'coords' / f'{coords_version}.json'
        assert coords_path.exists()

        wsi_path = self.dataset_dir / sample_id / 'wsi.tiff'
        assert wsi_path.exists()

        with open(coords_path, 'r') as f:
            coords = json.load(f)
            coords = [SlideCoordinate(**i) for i in coords]

        save_patches_dir = self.dataset_dir / sample_id / 'visualizations' / 'patches' / coords_version
        save_patches_dir.mkdir(parents=True, exist_ok=True)

        num_patches = min(num_patches, len(coords))
        idc = np.random.choice(range(len(coords)), size=num_patches, replace=False)
        for i in tqdm(idc):
            item = coords[i].model_dump()
            img = get_patch(item=item, to_tensor=False)
            save_path = save_patches_dir / f'id={coords[i].id}-uuid={coords[i].uuid}.png'
            imsave(save_path, np.array(img))

    def visualize_coords(self, sample_id: str, patch_size: int = 512):
        logger.info(f'Visualizing coords of {sample_id}')

        patch_stride = patch_size

        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={self.target_mpp:.4f}-overlap={self.overlap:.2f}'
        coords_path = self.dataset_dir / sample_id / 'coords' / f'{coords_version}.json'
        assert coords_path.exists()

        wsi_path = self.dataset_dir / sample_id / 'wsi.tiff'
        assert wsi_path.exists()
        slide = OpenSlide(wsi_path)

        with open(coords_path, 'r') as f:
            coords = json.load(f)
            coords = [SlideCoordinate(**i) for i in coords]

        img = visualize_coords(coords=coords, slide=slide)
        imsave(coords_path.with_suffix('.png'), img)

    ### legacy code, for custom patch embedding
    def create_patch_embeddings(self, sample_id: str, coords_version: str, model_name: str = 'uni_v1',
                                batch_size: int = 64, num_workers: int = 12):
        logger.info(f'Computing embeddings for {sample_id}')

        factory = encoder_factory(model_name)
        model = factory.model
        # TODO: check how to convert to v2.
        #  - save option would be to inspect manually and create the v2 transforms for each model
        #  - but seems to work fine
        transform = factory.eval_transforms
        precision = factory.precision

        coords_path = self.dataset_dir / sample_id / 'coords' / f'{coords_version}.json'
        assert coords_path.exists(), f'`coords_path` {coords_path} does not exist.'

        mean = transform.transforms[-1].mean
        std = transform.transforms[-1].std
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(224, interpolation=InterpolationMode.BILINEAR),
            v2.CenterCrop(224),  # note: Resize only resizes the shorter edge
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

        ds = SlidePatches(items_path=coords_path, transform=transform)
        ds.setup()
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = model.to('cuda').to(precision)

        save_embeddings_dir = self.dataset_dir / sample_id / 'embeddings' / 'patch' / f'model={model_name}' / coords_version
        save_embeddings_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
                images = batch.pop('image')
                images = images.to('cuda').to(precision)
                x = model(images)

                batch = batch_to_device(batch, device='cpu')
                batch['embedding'] = x.cpu()
                save_path = save_embeddings_dir / f'batch_idx={batch_idx}.pt'
                torch.save(x, save_path)


    ### legacy code, for custom slide embedding
    def create_slide_embedding(self, sample_id: str, coords_version: str, model_name: str = 'uni_v1',
                               batch_size: int = 64, num_workers: int = 12):
        import h5py
        import numpy as np
        from pathlib import Path
        from ai4bmr_learn.plotting.umap import plot_umap

        embeddings_dir = Path(
            "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/wsi_embed_subset/20x_256px_0px_overlap/features_uni_v1/")
        paths = sorted(embeddings_dir.glob('*.h5'))
        data = []
        for path in paths:
            with h5py.File(path, "r") as f:
                # print("Keys in file:", list(f.keys()))  # top-level datasets
                embeddings = f["features"][:]
            data.append(embeddings)
        data = np.concat(data, axis=0)
        print("Shape:", data.shape)

        ax = plot_umap(data=data)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    ### legacy code, for custom slide embedding
    def get_sample_embeddings(self, sample_id: str, model_name: str, coords_version: str, max_patches: int = 1000):
        embeddings_dir = self.dataset_dir / sample_id / 'embeddings' / 'patch' / f'model={model_name}' / coords_version
        assert embeddings_dir.exists(), f'`embeddings_dir` {embeddings_dir} does not exist.'

        data = []
        for path in sorted(embeddings_dir.glob('*.pt')):
            data.append(torch.load(path).cpu())
        data = torch.cat(data)

        idc = torch.randperm(data.shape[0])[:max_patches]
        data = data[idc]

        return data
    ### legacy code, for custom slide embedding
    def create_wsi_umap(self, sample_id: str, model_name: str, coords_version: str, max_patches: int = 1000):
        logger.info(f'Computing UMAP for {sample_id}')

        fname = f'model={model_name}-{coords_version}.png'
        save_path = self.dataset_dir / sample_id / 'visualizations' / 'umaps' / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            return

        data = self.get_sample_embeddings(sample_id=sample_id,
                                          model_name=model_name,
                                          coords_version=coords_version,
                                          max_patches=max_patches)

        if len(data) <= 15:
            logger.error(f"{sample_id} has only {len(data)} patches. Can't compute UMAP.")
            return

        ax = plot_umap(data=data)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def create_patches_umap(self, model_name: str, coords_version: str, max_patches_per_slide: int = 100):
        logger.info(f'Computing UMAP for patches')

        fname = f'model={model_name}-{coords_version}-max_patches_per_slide={max_patches_per_slide}.png'
        save_path = self.base_dir / '04_outputs' / 'umaps' / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            logger.info(f'UMAP already exists.')
            return

        data = []
        labels = []
        sample_ids = self.get_sample_ids()
        for sample_id in tqdm(sample_ids):
            tmp = self.get_sample_embeddings(sample_id=sample_id,
                                       model_name=model_name,
                                       coords_version=coords_version,
                                       max_patches=max_patches_per_slide)
            data.append(tmp)
            labels.extend([sample_id] * len(tmp))

        data = torch.concat(data)
        labels = np.array(labels)

        ax = plot_umap(data=data, labels=labels, show_legend=False)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def create_slides_umap(self):

        embeddings_dir: Path = Path(
            '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/wsi_embed_subset/20x_512px_0px_overlap/slide_features_titan')
        save_path = embeddings_dir.parent / 'umap-slides.png'

        data = []
        for path in sorted(embeddings_dir.glob('*.pt')):
            data.append(torch.load(path))
        data = torch.cat(data).cpu()

        ax = plot_umap(data=data)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def add_stats_to_metadata(self):

        sample_dirs = sorted([i for i in self.dataset_dir.iterdir() if i.is_dir()])
        stats = []
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name

            if sample_id in self.INVALID_SAMPLE_IDS:
                continue

            slide = OpenSlide(sample_dir / 'wsi.tiff')
            width, height = slide.dimensions
            mpp = get_mpp(slide=slide)
            record = {'sample_id': sample_id, 'width': width, 'height': height, 'mpp': mpp}

            for patch_size in [448]:
                with open(
                        sample_dir / 'coords' / f'patch_size={patch_size}-stride={patch_size}-mpp=0.8625-overlap=0.25.json') as f:
                    coords = json.load(f)
                    record[f'num_coords_patch_size={patch_size}'] = len(coords)

            stats.append(record)

        stats = pd.DataFrame(stats)
        stats = stats.set_index('sample_id')

        metadata_path = self.dataset_dir / 'metadata.parquet'
        metadata = pd.read_parquet(metadata_path)
        metadata = pd.concat([metadata, stats], axis=1)
        metadata.to_parquet(metadata_path)

    def prepare_data(self, sample_id: str, force: bool = False):
        self.prepare_tools()
        self.prepare_metadata()

        self.prepare_wsi(sample_id=sample_id, force=force)
        self.post_process_tiff_flags(sample_id=sample_id)
        self.create_thumbnail(sample_id=sample_id)

        self.segment(sample_id=sample_id, model_name='hest', target_mpp=4.0)
        self.segment(sample_id=sample_id, model_name='grandqc', target_mpp=4.0)

        for patch_size in [224, 448]:
            self.create_coords(sample_id=sample_id, patch_size=patch_size)
            self.visualize_coords(sample_id=sample_id, patch_size=patch_size)
            self.visualize_patches(sample_id=sample_id, patch_size=patch_size)

        patch_size = patch_stride = 448
        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={self.target_mpp:.4f}-overlap={self.overlap:.2f}'
        self.create_patch_embeddings(sample_id=sample_id, coords_version=coords_version, model_name='uni_v1')


# sample_id = '1FP2.0E.0'
# sample_id = '1FU2.06.0'
# coords_version = 'patch_size=448-stride=448-mpp=0.8625-overlap=0.25'
# model_name = 'uni_v1'
# max_patches_per_slide: int = 100
# beat = self = BEAT()