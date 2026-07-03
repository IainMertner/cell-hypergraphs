# cell-hypergraphs

## Pipeline

* H&E slides from TCGA-BRCA
* Cell segmentation and classification with CellVit++
* Five graph construction arms:
    - Two field baselines (kNN and Delaunay)
    - Two controls (hypergraph and its flattened pairwise counterpart)
    - Main hypergraph method (control hypergraph with added DeepSets aggregation layer)
* Cancer genomics tasks:
    - Cell-level subtype prediction using 30% masking