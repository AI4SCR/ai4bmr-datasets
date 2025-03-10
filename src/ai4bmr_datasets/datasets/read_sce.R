setwd('Documents/data/NSCLC/SingleCellExperiment Objects/')

library(basilisk)
library(scRNAseq)
library(zellkonverter)
library(reticulate)

# load sce from .rds file
sce <- readRDS('sce_all_annotated.rds') 
#sce_subset  <- sce[, 1:10000]

unique_patient_IDs <- unique(sce$Patient_ID)

# store sce as an anndata in the .h5ad format
writeH5AD(sce = sce, file='sce.h5ad')


## useful to generate the anndata until here



roundtrip <- basiliskRun(fun = function(sce) {
  # Convert SCE to AnnData:
  adata <- SCE2AnnData(sce)
  
  # Maybe do some work in Python on 'adata':
  # BLAH BLAH BLAH
  
  # Convert back to an SCE:
  AnnData2SCE(adata)
}, env = zellkonverterAnnDataEnv(), sce = sce_subset)

adata <- SCE2AnnData(sce_subset)

py_run_string("adata.write('adata_file.h5ad')")

writeH5AD(adata, "adata_file.h5ad")

