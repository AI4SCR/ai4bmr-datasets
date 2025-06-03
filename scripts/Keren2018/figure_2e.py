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
