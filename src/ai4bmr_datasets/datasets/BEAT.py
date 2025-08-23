import json
import json
import subprocess
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import openslide
import pandas as pd
import torch
from ai4bmr_core.utils.tidy import tidy_name
from ai4bmr_learn.data_models.Coordinate import SlideCoordinate
from ai4bmr_learn.datasets.items import SlidePatches, get_patch
from ai4bmr_learn.plotting.patches import visualize_coords
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


class BEAT:

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat')
        self.raw_dir = self.base_dir / '01_raw'
        self.processed_dir = self.base_dir / '02_processed'
        self.dataset_dir = self.processed_dir / 'datasets' / 'beat'
        self.tools_dir = self.base_dir / '03_tools'

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

    def prepare_clinical(self):
        """Questions
        Is this n=12 or n=14? I assume -14 can be ignored
        2024-05-24_n.1 - n.14/malexan3-24-05-2024-012-14.czi

        What are the corresponding numbers for `2024-05-31_n98-n100`

        """

        clinical = pd.read_excel(self.raw_dir / 'H_E_Images/Image_ID_vs_patient_blockIDs.xlsx')
        clinical.columns = clinical.columns.map(tidy_name)
        clinical = clinical.convert_dtypes()

        assert clinical.sample_barcode.value_counts().max() == 1

        dir_to_range = {
            '2024-05-28 slide n.15 - n.50': list(range(15, 50 + 1)),
            'Slide N°130-222': list(range(130, 222 + 1)),
            '2024-05-24_n.1 - n.14': list(range(1, 14 + 1)),
            '2024-05-30_n51 - n.97': list(range(51, 97 + 1)),
            '2024-06-05_n101-n129': list(range(101, 129 + 1)),
            '2024-05-31_n98-n100': list(range(98, 100 + 1))
        }

        import re
        wsi_paths = list(self.base_dir.rglob('*.ndpi')) + list(self.base_dir.rglob('*.czi'))
        records = []
        for wsi_path in wsi_paths:

            match = re.search(r'malexan3-31-05-2024-(\d{3})', str(wsi_path))
            if match is not None:
                index = int(match.group(1)) - 60
                n = dir_to_range[wsi_path.parent.name][index]
                records.append({'wsi_path': wsi_path, 'n': n})
                continue

            match = re.search(r'malexan3-\d{2}-\d{2}-\d{4}-(\d{3})\.czi$', str(wsi_path))
            if match is not None:
                index = int(match.group(1))
                if wsi_path.parent.name == '2024-06-05_n101-n129':
                    index -= 31
                n = dir_to_range[wsi_path.parent.name][index]
                records.append({'wsi_path': wsi_path, 'n': n})
                continue

            match = re.search(r'malexan3-\d{2}-\d{2}-\d{4}-(\d{3})-\d{1,2}\.czi$', str(wsi_path))
            if match is not None:
                index = int(match.group(1))
                n = dir_to_range[wsi_path.parent.name][index]
                records.append({'wsi_path': wsi_path, 'n': n})
                continue

            match = re.search(r'beat-meso (\d+) -', str(wsi_path).lower())
            match = match or re.search(r'best-meso (\d+) -', str(wsi_path).lower())
            n = int(match.group(1))
            records.append({'wsi_path': wsi_path, 'n': n})

        assert len(wsi_paths) == len(records)
        assert len(wsi_paths) == 183

        metadata = pd.DataFrame.from_records(records)
        metadata = metadata.merge(clinical)
        metadata['sample_id'] = metadata['sample_barcode'].str.strip() + '.' + metadata.groupby(
            'sample_barcode').cumcount().astype(str)
        metadata = metadata.convert_dtypes().astype({'wsi_path': str})
        metadata = metadata.set_index('sample_id')

        save_path = self.processed_dir / 'metadata' / 'clinical.parquet'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(save_path, engine='fastparquet')

    def get_wsi_path(self, sample_id: str):
        clinical = self.get_clinical_metadata()
        wsi_path = clinical[clinical.index == sample_id].wsi_path.item()
        assert Path(wsi_path).exists()
        return wsi_path

    def get_clinical_metadata(self):
        return pd.read_parquet(self.processed_dir / 'metadata' / 'clinical.parquet', engine='fastparquet')

    def prepare_wsi(self, sample_id: str, force: bool = False):

        if sample_id in ['1GUG.0M.2',  '1GVQ.06.2']:
            return

        self.prepare_tools()

        dataset_dir = self.dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)

        clinical = self.get_clinical_metadata()
        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        wsi_path = self.get_wsi_path(sample_id)

        intermediate_path = dataset_dir / sample_id / 'intermediate.tiff'
        save_path = dataset_dir / sample_id / 'wsi.tiff'

        if save_path.exists() and not force:
            logger.info(f"File {save_path} already exists. Skipping conversion.")
            return
        else:
            logger.info(f'Converting {wsi_path} to {sample_id}.')
            return

        assert False
        job_name = f"{sample_id}-wsi2tiff"
        logger.info(f"Converting {wsi_path} to {save_path} using Bio-Formats (job_name={job_name})")
        num_cpus = 16

        sbatch_command = f"""
        sbatch \\
        --job-name={job_name} \\
        --output=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/{job_name}-%j.log \\
        --error=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/{job_name}-%j.err \\
        --time=24:00:00 \\
        --mem=128G \\
        --cpus-per-task={num_cpus} \\
        <<'EOF'
        #!/bin/bash
        
        source /users/amarti51/miniconda3/bin/activate
        conda activate beat
        
        INPUT="{wsi_path}"
        INTERMEDIATE="{intermediate_path}"
        OUTPUT="{save_path}"
        BFCONVERT="{bfconvert}"
        
        mkdir -p "$(dirname "{save_path}")"

        # Convert NDPI/other WSI to intermediate TIFF
        # /usr/bin/time -v "$BFCONVERT" -nogroup -bigtiff -compression LZW -tilex 512 -tiley 512 -series 0 -pyramid-resolutions 4 -pyramid-scale 2 "$INPUT" "$INTERMEDIATE"
        echo "* Exporting full-res TIFF..."
        /usr/bin/time -v "$BFCONVERT" -nogroup -bigtiff -series 0 "$INPUT" "$INTERMEDIATE"

        # Convert intermediate TIFF to OpenSlide-compatible pyramid TIFF
        echo "* Building OpenSlide-compatible pyramid TIFF..."
        export VIPS_CONCURRENCY={num_cpus - 1}
        /usr/bin/time -v vips tiffsave "$INTERMEDIATE" "$OUTPUT" --pyramid --tile --tile-width 512 --tile-height 512 --bigtiff --compression lzw
        
        rm "$INTERMEDIATE"
        
        EOF
        """
        sbatch_command = textwrap.dedent(sbatch_command).strip()

        subprocess.run(sbatch_command, shell=True, check=True)

    def post_process_tiff_flags(self, sample_id: str):

        wsi_path = self.get_wsi_path(sample_id)
        slide = openslide.OpenSlide(wsi_path)

        try:
            mpp = get_mpp(slide)
        except AssertionError:
            logger.error(f'Error retriving slide mpp for {wsi_path}')
            return

        if str(mpp).startswith('4.422'):
            cmd = f"""
            # set XResolution (tag 282) to 22611
            tiffset -s 282 22611 "{wsi_path}"
            
            # set YResolution (tag 283) to 22611
            tiffset -s 283 22611 "{wsi_path}"
            
            # set ResolutionUnit (tag 296) to 3 (centimeter)
            tiffset -s 296 3 "{wsi_path}"
            """
            cmd = textwrap.dedent(cmd).strip()
            subprocess.run(cmd, shell=True, check=True)

    def segment(self, sample_id: str, model_name: str = 'grandqc', target_mpp: float = 4.0):

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

        logger.info(f'🧩 segmenting {sample_id}')

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

    def create_thumbnail(self, sample_id: str):

        img_path = self.dataset_dir / sample_id / 'wsi.tiff'
        if not img_path.exists():
            return

        logger.info(f'Create thumbnail for {sample_id}')

        slide = OpenSlide(img_path)
        thumbnail, _ = get_thumbnail(slide=slide)

        save_path = img_path.parent / 'thumbnail.png'
        imsave(str(save_path), thumbnail)

    def prepare_data(self, force: bool = False):
        self.prepare_tools()
        self.prepare_clinical()

        self.prepare_wsi(force=force)
        self.post_process_tiff_flags()

        self.segment(model_name='hest', target_mpp=4)
        self.segment(model_name='grandqc', target_mpp=4)

        self.create_coords(patch_size=224)
        self.create_coords(patch_size=512)

        patch_size = patch_stride = 512
        target_mpp = 0.8625
        overlap = 0.25
        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'
        self.create_embeddings(coords_version=coords_version, model_name='uni_v1')

    def filter_contours(self):
        # TODO: potentially post-process contours or improve filtering during segmentation.
        # NOTE: we only use contours that are larger than the median area of all found contours
        # q = np.quantile(contours.geometry.area, 0.5)
        # filter_ = contours.geometry.area > q
        # import matplotlib.pyplot as plt
        # plt.imshow(visualize_contours(contours=contours[filter_], slide=slide)).figure.show()
        pass

    def create_coords(self, sample_id: str, patch_size: int = 512, model_name: str = 'hest'):
        patch_stride = patch_size
        target_mpp = 0.8624999831542972  # TODO: check this value for the used model
        overlap = 0.25
        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'

        wsi_path = self.dataset_dir / sample_id / 'wsi.tiff'
        coords_path = self.dataset_dir / sample_id / 'coords' / f"{coords_version}.json"

        if coords_path.exists():
            logger.info(f'{coords_path} exists.')
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
        target_mpp = 0.8624999831542972
        overlap = 0.25

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
        idc = np.random.choice(range(len(coords)), size=num_patches, replace=False)
        for i in tqdm(idc):
            item = coords[i].model_dump()
            img = get_patch(item=item, to_tensor=False)
            save_path = save_patches_dir / f'id={coords[i].id}-uuid={coords[i].uuid}.png'
            imsave(save_path, np.array(img))

    def visualize_coords(self, sample_id: str, patch_size: int = 512):
        logger.info(f'Visualizing coords of {sample_id}')

        patch_stride = patch_size
        target_mpp = 0.8624999831542972
        overlap = 0.25

        coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'
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

    def create_embeddings(self, sample_id: str, coords_version: str, model_name: str = 'uni_v1',
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

        save_embeddings_dir = self.dataset_dir / sample_id / 'embeddings' / f'model={model_name}' / coords_version
        save_embeddings_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
                images = batch.pop('image')
                images = images.to('cuda').to(precision)
                x = model(images)

                batch['embedding'] = x.cpu()
                save_path = save_embeddings_dir / f'batch_idx={batch_idx}.pt'
                torch.save(x, save_path)

    def create_umaps(self, embeddings_version: ''):
        pass

sample_id = '1FP2.0E.0'
beat = self = BEAT()
