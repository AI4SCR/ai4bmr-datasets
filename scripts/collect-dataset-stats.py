import pandas as pd

from ai4bmr_datasets import Danenberg2022, Cords2024, Jackson2020, Keren2018
from pathlib import Path

datasets_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets')
cont = []
for ds_cls in [Danenberg2022, Cords2024, Jackson2020, Keren2018]:
    base_dir = datasets_dir / ds_cls.name
    assert base_dir.exists()

    ds = ds_cls(base_dir=base_dir)
    image_version = mask_version = feature_version = metadata_version = 'published'
    if ds.name == 'Danenberg2022':
        mask_version = 'published_cell'

    ds.setup(image_version=image_version, mask_version=mask_version,
             feature_version=feature_version, load_intensity=True,
             metadata_version=metadata_version, load_metadata=True)

    stats = dict(
        name=ds.name,
        num_images=len(ds.images),
        num_masks=len(ds.masks),
        num_markers=len(ds.panel),
        num_annotated_cells=len(ds.metadata),
        num_clinical_samples=len(ds.clinical)
    )
    cont.append(stats)

stats = pd.DataFrame(cont)
print(stats.to_markdown())

# %%
