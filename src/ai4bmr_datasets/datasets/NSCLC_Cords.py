from .BaseIMCDataset import BaseIMCDataset
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import tifffile


class NSCLC(BaseIMCDataset):


    def __init__(self, base_dir: Path):


        super().__init__(base_dir)

        # TO DO get logger 
        #self.logger = get_logger('TNBCv2', verbose=1)
        self.version_name = 'published'

        self.raw_masks_dir = self.raw_dir / 'Cell_masks'
        self.raw_images_mcd = self.raw_dir / 'raw/mcd'


        self.supplementary_table_1_path = self.raw_dir / '1-s2.0-S0092867418311000-mmc1.xlsx'  # downloaded as Table S1 from paper
        self.raw_sca_path = self.raw_dir / 'TNBC_shareCellData' / 'cellData.csv'
        self.patient_class_path = self.raw_dir / 'TNBC_shareCellData' / 'patient_class.csv'


    @staticmethod
    def get_tiff_metadata(path):
        import tifffile
        tag_name = 'PageName'
        metadata = pd.DataFrame()
        metadata.index.name = 'page'
        with tifffile.TiffFile(path) as tif:
            for page_num, page in enumerate(tif.pages):
                page_name_tag = page.tags.get(tag_name)
                import pdb; pdb.set_trace()
                metadata.loc[page_num, tag_name] = page_name_tag.value
        metadata = metadata.assign(PageName=metadata.PageName.str.strip())
        metadata = metadata.PageName.str.extract(r'(?P<target>\w+)\s+\((?P<mass_channel>\d+)')

        metadata.mass_channel = metadata.mass_channel.astype('int')
        metadata = metadata.convert_dtypes()

        return metadata

panel_86 = pd.read_csv('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/86_B_panel.csv')

import pdb; pdb.set_trace()

import tifffile
import re


def extract_metadata(path):
    metadata = pd.DataFrame()
    metadata.index.name = "page"

    with tifffile.TiffFile(path) as tif:
        for page_num, page in enumerate(tif.pages):
            description_tag = page.tags.get("ImageDescription")
            description = description_tag.value if description_tag else ""
            #import pdb; pdb.set_trace()

            metadata.loc[page_num, "ImageDescription"] = description

            match_channels = re.search(r'channels=(\d+)', description)
            metadata.loc[page_num, "channel"] = int(match_channels.group(1)) if match_channels else None

    metadata = metadata.convert_dtypes()

    return metadata



path = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img/20210129_LC_NSCLC_TMA_88_A_073.tiff'
metadata_df = extract_metadata(path)
print(metadata_df)

import pdb; pdb.set_trace()



# read attributes of the image file
image = Image.open('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img/20210129_LC_NSCLC_TMA_88_A_073.tiff')

exif_data = image.getexif()

for tag_id, value in exif_data.items():
    tag = TAGS.get(tag_id, tag_id)
    print(f"{tag}: {value}")

print(' ------- ')

with tifffile.TiffFile('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img/20210129_LC_NSCLC_TMA_88_A_073.tiff') as tif:
    for tag in tif.pages[0].tags.values():
        print(f"{tag.name}: {tag.value}")

import pdb; pdb.set_trace()

    


NSCLS_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/')
NSCLC_dataset = NSCLC(base_dir=NSCLS_path)
    
img_path = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img/20210129_LC_NSCLC_TMA_88_A_073.tiff'
NSCLC_dataset.get_tiff_metadata(img_path)





