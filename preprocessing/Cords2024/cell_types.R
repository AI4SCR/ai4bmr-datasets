library(arrow)

path = '/Users/adrianomartinelli/Downloads/SingleCellExperiment Objects/sce_all_annotated.rds'
data = readRDS(path)
data

coldat = colData(data)
colnames(coldat)
cell_types = coldat[, c("cell_category", "cell_type", "cell_subtype")]
cell_types = as.data.frame(cell_types)
cell_types['object_id'] = rownames(cell_types)
write_parquet(
  cell_types,
  "/Users/adrianomartinelli/Downloads/SingleCellExperiment Objects/cell_types.parquet",
)


# rowData(data)
intensity = assay(data, 'counts')
intensity['object_id'] = rownames(intensity)
intensity = as.data.frame(intensity)
write_parquet(
  intensity,
  "/Users/adrianomartinelli/Downloads/SingleCellExperiment Objects/intensities.parquet",
)


