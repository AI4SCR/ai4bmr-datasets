from skimage.measure import regionprops_table
import pandas as pd
from ..datamodels.Image import Image, Mask


def get_intensity(image: Image, mask: Mask, properties: tuple = ('label', 'intensity_mean')):
    df = pd.DataFrame(regionprops_table(mask.data, image.data, properties=properties))
    metadata = image.metadata

    if metadata and 'target' in metadata:
        df.columns = df.columns.map(metadata['target'].to_dict())
        df.index.name = 'object_id'
        df = df.assign(sample_id=image.id).set_index('sample_id', append=True)

    return df
