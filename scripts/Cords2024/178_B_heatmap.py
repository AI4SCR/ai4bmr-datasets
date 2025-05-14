from ai4bmr_datasets.datasets.Cords2024 import Cords2024
from pathlib import Path
from ai4bmr_datasets.utils.imc.normalize import normalize
import pandas as pd

image_version='published'
mask_version='published'
metadata_version='published'
ds  = Cords2024(base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024'))
# ds.setup(image_version='published', mask_version='published', metadata_version=metadata_version,
#          load_metadata=True, load_intensity=True)

prefix = '178_B'
# prefix = '88_C'
feature_version = ds.get_version_name(image_version='published', mask_version='published')

intensity_dir = ds.intensity_dir / feature_version
intensity = pd.concat(
    [
        pd.read_parquet(i, engine="fastparquet")
        for i in intensity_dir.glob("*.parquet")
        if i.name.startswith(prefix)
    ]
)
intensity = normalize(intensity)

metadata_dir = ds.metadata_dir / 'published'
metadata = pd.concat(
    [
        pd.read_parquet(i, engine="fastparquet")
        for i in metadata_dir.glob("*.parquet")
        if i.name.startswith(prefix)
    ]
)

metadata.index = pd.MultiIndex.from_frame(metadata.index.to_frame().astype(dict(object_id=int)))

dat, meta = intensity.align(metadata, axis='index', join='inner')
assert dat.isna().any().any() == False
assert meta.isna().any().any() == False

# i, m = intensity.align(metadata, axis='index', join='outer')
# i.isna().any(axis=1).sum()
# m.isna().any(axis=1).sum()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from matplotlib.patches import Patch

save_dir = Path('/users/amarti51/prometex/data/ai4bmr-datasets/Cords2024')
save_dir.mkdir(parents=True, exist_ok=True)

col_names = list(set(intensity.columns) - set(['iridium', 'iridium_1']))
cmap = cc.glasbey_bw
num_colors = len(cmap)
var_name = 'cell_type'

row_colors = meta[var_name]
color_map = {k: cmap[i % num_colors] for i, k in enumerate(set(row_colors))}

legend_handles = [Patch(facecolor=color_map[k], label=k) for k in color_map]

# %%
pdat = pd.concat((dat, meta), axis=1).groupby(var_name)[col_names].mean()
row_colors = pdat.index.get_level_values(var_name).to_list()
row_colors = pd.DataFrame({var_name: list(map(color_map.get, row_colors))}, index=row_colors)
cg = sns.clustermap(data=pdat, row_colors=row_colors, linewidth=.1, linecolor='gray')
cg.ax_cbar.legend(handles=legend_handles, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))

cg.figure.show()
cg.figure.savefig(save_dir / f'clustermap-{prefix}-mean.png')
plt.close(cg.figure)

# %%
pdat = dat[col_names]
num_cells = min(50_000, len(pdat))
pdat = pdat.sample(num_cells, random_state=0)

row_colors = meta[var_name]
row_colors = pd.DataFrame({var_name: list(map(color_map.get, row_colors))}, index=row_colors.index)
cg = sns.clustermap(data=pdat, row_colors=row_colors)
cg.ax_cbar.legend(handles=legend_handles, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))

cg.figure.suptitle(f'{num_cells} cells from {prefix}')
cg.figure.show()
cg.figure.savefig(save_dir / f'clustermap-{prefix}.png')
plt.close(cg.figure)

