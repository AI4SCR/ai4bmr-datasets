import shutil
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
        self.log_dir = self.base_dir / '04_logs'

    def prepare_tools(self):
        bfconvert = self.tools_dir / "bftools" /"bfconvert"

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
        {self.tools_dir / "bftools" /"bfconvert"} -version || exit
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
        metadata['sample_id'] = metadata['sample_barcode'].str.strip() + '.' + metadata.groupby('sample_barcode').cumcount().astype(str)
        metadata = metadata.convert_dtypes().astype({'wsi_path': str})
        metadata = metadata.set_index('sample_id')

        save_path = self.processed_dir / 'metadata' / 'clinical.parquet'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(save_path, engine='fastparquet')

    def prepare_wsi(self, force: bool = False):
        import textwrap
        import subprocess

        self.prepare_tools()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = self.datasets_dir / 'beat'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        clinical = pd.read_parquet(self.processed_dir / 'metadata' / 'clinical.parquet', engine='fastparquet')
        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        wsi_paths = list(self.base_dir.rglob('*.ndpi')) + list(self.base_dir.rglob('*.czi'))
        for wsi_path in wsi_paths:

            save_name = clinical[clinical.wsi_path == str(wsi_path)].index.item()
            save_path = dataset_dir / save_name / 'wsi.ome.tiff'
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.exists() and not force:
                logger.info(f"File {save_path} already exists. Skipping conversion.")

            job_name = f"wsi2tiff-{save_name}"
            logger.info(f"Converting {wsi_path} to {save_path} using Bio-Formats (job_name={job_name})")

            sbatch_command = f"""
            sbatch \\
            --job-name={job_name} \\
            --output=/users/amarti51/logs/{job_name}-%A-%a.log \\
            --error=/users/amarti51/logs/{job_name}-%A-%a.err \\
            --time=08:00:00 \\
            --mem=128G \\
            --cpus-per-task=12 \\
            <<'EOF'
            #!/bin/bash

            INPUT="{wsi_path}"
            OUTPUT="{save_path}"
            BFCONVERT="{bfconvert}"

            /usr/bin/time -v "$BFCONVERT" -nogroup -bigtiff -compression LZW -tilex 512 -tiley 512 "$INPUT" "$OUTPUT"
            EOF
            """
            sbatch_command = textwrap.dedent(sbatch_command).strip()

            subprocess.run(sbatch_command, shell=True, check=True)

    def segment(self, model_name: str = 'grandqc', target_mpp: float = 4.0):
        from ai4bmr_learn.utils.slides import segment_slide, get_seg_model
        from openslide import OpenSlide

        seg_model = get_seg_model(model_name=model_name)  # hest, grandqc, grandqc_artifact
        sample_dirs = [i for i in (self.datasets_dir / 'beat').iterdir() if i.is_dir()]
        for sample_dir in sample_dirs:

            save_coords_path = sample_dir / 'coords' / f'segment-mpp={target_mpp}-model_name={model_name}.parquet'
            save_contours_path = sample_dir / 'contours' / f'segment-mpp={target_mpp}-model_name={model_name}.parquet'
            img_path = sample_dir / 'wsi.ome.tiff'

            segment_slide(slide,
                          seg_model=seg_model,
                          target_mpp=target_mpp,
                          save_contours_path=save_contours_path, save_coords_path=save_coords_path)

    def setup(self):
        pass

# self = BEAT()