from ai4bmr_datasets.datasets.TNBC import TNBC

dataset = TNBC(verbose=1)
data = dataset.load()

image = data['images'][14]
masks = data['masks'][14]

import numpy as np

np.sort(image.data.sum(axis=(1, 2)))

df = data['data']

# %%
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

cols = ['FoxP3', 'CD4', 'CD3', 'CD56', 'CD209', 'CD20', 'HLA-DR', 'CD11c', 'CD16', 'CD68', 'CD11b', 'MPO', 'CD45',
        'CD31',
        'SMA', 'Vimentin', 'Beta catenin', 'Pan-Keratin', 'p53', 'EGFR', 'Keratin17', 'Keratin6']
pdat = df[cols]
pdat = pdat[pdat.index.get_level_values('sample_id') <= 41]

pdat, row_colors = pdat.align(data['metadata'], join='left', axis=0)
pdat = pdat.assign(group_name=pd.Categorical(row_colors.group_name,
                                             categories=['immune', 'endothelial', 'Mmsenchymal_like', 'tumor',
                                                         'keratin_positive_tumor', 'unidentified']))
pdat = pdat.set_index('group_name', append=True)
color_map = {'immune': np.array((187, 103, 30, 0)) / 255,
             'endothelial': np.array((253, 142, 142, 0)) / 255,
             'Mmsenchymal_like': np.array((1, 56, 54, 0)) / 255,
             'tumor': np.array((244, 201, 254, 0)) / 255,
             'keratin_positive_tumor': np.array((128, 215, 230, 0)) / 255,
             'unidentified': np.array((255, 255, 255, 0)) / 255,
             }

row_colors = row_colors['group_name'].map(color_map)
pdat = pdat.sort_values('group_name')
pdat = pd.DataFrame(MinMaxScaler().fit_transform(pdat), index=pdat.index, columns=pdat.columns)
cg = sns.clustermap(pdat,
                    cmap='bone',
                    row_colors=row_colors,
                    col_cluster=False,
                    row_cluster=False,
                    # figsize=(20, 20)
                    )
cg.ax_heatmap.set_facecolor('black')
cg.ax_heatmap.set_yticklabels([])
cg.figure.tight_layout()
cg.figure.show()
cg.figure.savefig('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/datasets/TNBC/tnbc.png', dpi=300)

# not_used = ['PD1',  'PD-L1',
#        'Ki67', 'CD209', 'CD11c', 'CD138',  'CD8', 'IDO',
#        , 'CD63', 'CD45RO', 'CD20',  'HLA-DR',
#           'phospho-S6',
#        'HLA_Class_1', 'dsDNA',  'H3K9ac', 'H3K27me3']
