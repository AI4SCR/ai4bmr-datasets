import shutil
from openslide import OpenSlide
from pathlib import Path

import pandas as pd
from loguru import logger


class BEAT:

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat')
        self.raw_dir = self.base_dir / '01_raw'
        self.processed_dir = self.base_dir / '02_processed'
        self.datasets_dir = self.processed_dir / 'datasets'
        self.tools_dir = self.base_dir / '03_tools'

    def prepare_tools(self):
        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        if bfconvert.exists():
            logger.info(f"bfconvert found at {bfconvert}. Skipping installation.")
            return

        import textwrap
        import subprocess
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
        import pandas as pd
        from ai4bmr_core.utils.tidy import tidy_name

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

    def prepare_wsi(self, force: bool = False):
        import textwrap
        import subprocess

        self.prepare_tools()

        dataset_dir = self.datasets_dir / 'beat'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        clinical = pd.read_parquet(self.processed_dir / 'metadata' / 'clinical.parquet', engine='fastparquet')
        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        wsi_paths = list(self.base_dir.rglob('*.ndpi')) + list(self.base_dir.rglob('*.czi'))
        for wsi_path in wsi_paths:

            save_name = clinical[clinical.wsi_path == str(wsi_path)].index.item()
            if save_name in [
                '1GUG.0M.2',  # barcode
                '1GVQ.06.2',  # barcode
            ]:
                continue
            
            intermediate_path = dataset_dir / save_name / 'intermediate.tiff'
            save_path = dataset_dir / save_name / 'wsi.tiff'

            if save_path.exists() and not force:
                logger.info(f"File {save_path} already exists. Skipping conversion.")

            job_name = f"proj=beat-task=wsi2tiff-{save_name}"
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

    def post_process_tiff_flags(self):
        from ai4bmr_learn.utils.slides import get_mpp
        import openslide
        import textwrap
        import subprocess

        wsi_paths = list(self.datasets_dir.rglob('*wsi.tiff'))
        for wsi_path in wsi_paths:
            slide = openslide.OpenSlide(wsi_path)
            
            try:
                mpp = get_mpp(slide)
            except AssertionError:
                logger.error(f'Error retriving slide mpp for {wsi_path}')
                continue

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

    def segment(self, sample_dirs: list[Path] | None = None, model_name: str = 'grandqc', target_mpp: float = 4.0, source_mpp: float = None):
        from ai4bmr_learn.utils.slides import segment_slide, get_seg_model, get_mpp_and_resolution
        from openslide import OpenSlide

        seg_model = get_seg_model(model_name=model_name)  # hest, grandqc, grandqc_artifact
        sample_dirs = sample_dirs or [i for i in (self.datasets_dir / 'beat').iterdir() if i.is_dir()]
        for i, sample_dir in enumerate(sample_dirs, start=1):

            img_path = sample_dir / 'wsi.tiff'
            if not img_path.exists():
                continue

            version_name = f'segment-mpp={target_mpp}-model_name={model_name}' + (
                f'-src_mpp={source_mpp}' if source_mpp else '')
            save_coords_path = sample_dir / 'coords' / f'{version_name}.json'
            save_contours_path = sample_dir / 'contours' / f'{version_name}.parquet'

            if save_contours_path.exists() and save_contours_path.exists():
                logger.info(f'[{i}/{len(sample_dirs)}] {sample_dir} is already segmented')
                continue

            logger.info(f'[{i}/{len(sample_dirs)}] 🧩 segmenting {sample_dir}')

            slide = OpenSlide(img_path)

            try:
                mpp = get_mpp_and_resolution(slide)[0]
            except AssertionError:
                logger.error(f'Error retriving slide mpp for {img_path}')
                continue

            logger.info(
                f'{sample_dir.name} reports mpp={mpp:.4f}' + (f', assuming mpp={source_mpp}' if source_mpp else ''))

            segment_slide(slide,
                          seg_model=seg_model,
                          target_mpp=target_mpp,
                          source_mpp=source_mpp,
                          save_contours_path=save_contours_path,
                          save_coords_path=save_coords_path)

    def create_thumbnail(self):
        from skimage.io import imsave
        from ai4bmr_learn.utils.slides import get_thumbnail
        from openslide import OpenSlide

        sample_dirs = [i for i in (self.datasets_dir / 'beat').iterdir() if i.is_dir()]
        for i, sample_dir in enumerate(sample_dirs, start=1):

            img_path = sample_dir / 'wsi.tiff'
            if not img_path.exists():
                continue

            logger.info(f'Create thumbnail for {sample_dir.name}')

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

        self.create_patch_embeddings()

    def setup(self):
        pass

    def create_patch_embeddings(self, model_name: str = 'uni_v1', batch_size: int = 64, num_workers: int = 12):
        from trident.patch_encoder_models import encoder_factory
        from ai4bmr_learn.utils.slides import get_coordinates_dict, get_slide_patcher_params, get_mpp
        from ai4bmr_learn.utils.images import filter_coords
        from ai4bmr_learn.data_models.Coordinate import SlideCoordinate
        from ai4bmr_learn.datasets.Patches import Patches, get_patch
        from ai4bmr_learn.plotting.patches import visualize_coords
        from openslide import OpenSlide
        import geopandas as gpd
        from dataclasses import asdict
        import json
        from skimage.io import imsave
        import numpy as np
        from torchvision.transforms import v2
        import torch
        from torch.utils.data import DataLoader
        from ai4bmr_learn.utils.device import batch_to_device
        from tqdm import tqdm

        factory = encoder_factory(model_name)
        model = factory.model
        transform = factory.eval_transforms  # TODO: check how to convert to v2
        precision = factory.precision

        wsi_paths = sorted(self.datasets_dir.rglob('*wsi.tiff'))
        for wsi_path in wsi_paths:
            logger.info(f'Computing patch embeddings for {wsi_path.parent.name}...')

            slide = OpenSlide(str(wsi_path))
            contours_path = wsi_path.parent / 'contours' / "segment-mpp=4-model_name=hest.parquet"
            contours = gpd.read_parquet(contours_path)

            patch_size = patch_stride = 224
            target_mpp = 0.8624999831542972  # TODO: check this value for the used model
            overlap = 0.25
            params = get_slide_patcher_params(slide=slide, patch_size=patch_size, patch_stride=patch_stride, target_mpp=target_mpp)
            coords = get_coordinates_dict(**params)
            coords = [SlideCoordinate(**i) for i in coords]
            coords = filter_coords(coords=coords, contours=contours, overlap=overlap)

            coords_version = f'patch_size={patch_size}-stride={patch_stride}-mpp={target_mpp:.4f}-overlap={overlap:.2f}'
            save_coords_path = wsi_path.parent / 'coords' / f'{coords_version}.json'
            save_coords_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_coords_path, 'w') as f:
                    coords_dict = [asdict(i) for i in coords]
                    json.dump(coords_dict, f)
                
            img = visualize_coords(coords=coords,slide=slide)
            imsave(save_coords_path.with_suffix('.png'), img)

            save_patches_dir = wsi_path.parent / 'patches' / coords_version
            save_patches_dir.mkdir(parents=True, exist_ok=True)
            for i in np.random.choice(range(len(coords)), size=100, replace=False):
                img = get_patch(coord=coords[i], as_tensor=False)
                save_path = save_patches_dir / f'id={coords[i].id}-uuid={coords[i].uuid}.png'
                imsave(save_path, np.array(img))

            mean = transform.transforms[-1].mean
            std = transform.transforms[-1].std
            transform = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
            ds = Patches(coords=coords, transform=transform)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            model = model.to('cuda').to(precision)

            save_embeddings_dir = wsi_path.parent / 'embeddings' / coords_version
            save_embeddings_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
                    images = batch.pop('image')
                    images = images.to('cuda').to(precision)
                    x = model(images)
                    
                    batch['embedding'] = x.cpu()
                    save_path = save_embeddings_dir / f'batch_idx={batch_idx}.pt'
                    torch.save(x, save_path)
            break
